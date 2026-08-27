"""PoC: one full BitLinear from ternlight `.bin` (LN + act quant + Triton pack matmul).

Compares against training/pack/unpack.bitlinear_forward semantics.

  PYTHONPATH=. python3 -m flash_ternary.poc_ternlight_layer \\
      --bin training/pack/out/en-minilm-int4.bin \\
      --layer 0 --name fc1
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch

from .bin_loader import load_bin
from .ternlight_layer import BitLinearTritonFromBin, bitlinear_forward_ref

_REPO_ROOT = Path(os.environ.get("TERNLIGHT_REPO") or Path(__file__).resolve().parents[3])


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, n=40, warmup=12):
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(n):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return ts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bin", default=str(_REPO_ROOT / "training" / "pack" / "out" / "en-minilm-int4.bin")
    )
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--name", default="fc1")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq", type=int, default=1)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--list", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA required")
        return 1

    header, _body, slices = load_bin(args.bin)
    print(
        f"bin={args.bin}\n"
        f"  header: layers={header.n_layers} d_model={header.d_model} "
        f"ffn={header.ffn_dim} vocab={header.vocab_size} rope={header.pos_encoding}"
    )
    if args.list:
        for s in slices:
            print(
                f"  L{s.layer}.{s.name}: in={s.in_features} out={s.out_features} "
                f"bias={s.has_bias} pack={s.weights_bytes}B"
            )
        return 0

    layer = BitLinearTritonFromBin.from_bin(
        args.bin, layer=args.layer, name=args.name, device="cuda"
    )
    print(f"loaded: {layer.extra_repr()}")

    torch.manual_seed(args.seed)
    rows = args.batch * args.seq
    x = torch.randn(rows, layer.in_features, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        y_t = layer(x)
        y_r = layer.forward_ref(x)
        # also pack/unpack reference from training/pack if importable
        y_u = None
        try:
            pack_dir = _REPO_ROOT / "training" / "pack"
            if str(pack_dir) not in sys.path:
                sys.path.insert(0, str(pack_dir))
            from unpack import bitlinear_forward as bl_fwd  # type: ignore

            W = layer.w_quant_matrix().to(device=x.device)
            bias = None if layer.bias is None else layer.bias.to(device=x.device)
            y_u = bl_fwd(x, W.to(torch.int8), layer.w_scale, bias)
        except Exception as e:
            print(f"(optional unpack.py ref skipped: {e})")

    def _report(tag, a, b):
        d = (a - b).abs()
        rel = d.max().item() / (b.abs().max().item() + 1e-6)
        print(
            f"  {tag}: max|e|={d.max().item():.4e} mean|e|={d.mean().item():.4e} "
            f"rel_max={rel:.4e}"
        )
        return rel

    print("correctness (full BitLinear: LN + act quant + matmul + scale):")
    r1 = _report("triton vs expand_ref", y_t, y_r)
    ok = r1 < 2e-2 or (y_t - y_r).abs().max().item() < 0.5
    if y_u is not None:
        r2 = _report("triton vs pack/unpack.py", y_t, y_u)
        r3 = _report("expand_ref vs unpack.py", y_r, y_u)
        ok = ok and r2 < 2e-2
        ok = ok and r3 < 1e-5  # pure torch paths should match tightly
    if not ok:
        print("FAIL correctness")
        return 1
    print("OK correctness")

    pack_mib = layer.packed_bytes / 1024**2
    fp16_mib = layer.fp16_weight_bytes / 1024**2
    print(
        f"weight memory: pack={pack_mib:.3f} MiB  fp16_equiv={fp16_mib:.3f} MiB  "
        f"compression={fp16_mib / max(pack_mib, 1e-9):.1f}x"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        layer(x)
    _sync()

    ts_t = _timed(lambda: layer(x), n=args.iters, warmup=15)
    ts_r = _timed(lambda: layer.forward_ref(x), n=max(15, args.iters // 2), warmup=8)
    peak = torch.cuda.max_memory_allocated() / 1024**2
    mt, mr = statistics.median(ts_t), statistics.median(ts_r)
    print(
        f"speed rows={rows}: triton_full={mt:.3f} ms  expand_ref={mr:.3f} ms  "
        f"speedup={mr / mt:.2f}x  peak_alloc≈{peak:.1f} MiB"
    )

    # matmul-only micro (skip LN) for kernel floor
    from .ternlight_layer import bitlinear_pre_act
    from .kernel_triton import fused_ternary_matmul

    x_q, x_s = bitlinear_pre_act(x)
    x_h = x_q.half().contiguous()
    Wp = layer.weight_packed

    def matmul_only():
        return fused_ternary_matmul(x_h, Wp, layer.in_features)

    for _ in range(5):
        matmul_only()
    ts_m = _timed(matmul_only, n=args.iters, warmup=15)
    print(f"  of which Triton matmul-only (precomputed act): {statistics.median(ts_m):.3f} ms")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
