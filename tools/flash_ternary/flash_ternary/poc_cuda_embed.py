"""End-to-end CUDA pack embed vs UnpackedModel (expand) reference.

  PYTHONPATH=. python3 -m flash_ternary.poc_cuda_embed \\
    --bin training/pack/out/en-minilm-int4.bin
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(os.environ.get("TERNLIGHT_REPO") or Path(__file__).resolve().parents[3])


def _sync():
    torch.cuda.synchronize()


def _timed(fn, n=20, warmup=5):
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
    p.add_argument(
        "--tokenizer",
        default=str(_REPO_ROOT / "training" / "data" / "en" / "spm-en-50k"),
        help="HF tokenizer dir (must match bin vocab)",
    )
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA required")
        return 1

    from .cuda_embed import PackCudaEmbed

    pack_dir = _REPO_ROOT / "training" / "pack"
    if str(pack_dir) not in sys.path:
        sys.path.insert(0, str(pack_dir))
    from unpack import UnpackedModel  # type: ignore

    texts = [
        "ternary quantization embedding BitLinear",
        "FlashTernary fused matmul pack 2-bit weights",
        "JMTEB japanese embedding evaluation harness",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print(f"loading pack CUDA embed from {args.bin} ...")
    t0 = time.perf_counter()
    model = PackCudaEmbed.from_bin(args.bin, device="cuda")
    print(f"  load {time.perf_counter()-t0:.2f}s  "
          f"bitlinear_pack_total={model.packed_bitlinear_bytes()/1024**2:.2f} MiB  "
          f"d_model={model.d_model} heads={model.n_heads} out={model.output_dim} rope={model.pos_rope}")

    print("loading UnpackedModel reference (CPU expand) ...")
    t0 = time.perf_counter()
    ref = UnpackedModel.from_bin(args.bin)
    print(f"  load {time.perf_counter()-t0:.2f}s")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
    ids_cpu = enc["input_ids"]
    mask_cpu = enc["attention_mask"]
    ids = ids_cpu.cuda()
    mask = mask_cpu.cuda()

    with torch.inference_mode():
        y_cuda = model(ids, mask).float().cpu()
        y_ref = ref.forward(ids_cpu, mask_cpu).float()

    # cosine per row
    cos = (y_cuda * y_ref).sum(-1) / (
        y_cuda.norm(dim=-1) * y_ref.norm(dim=-1) + 1e-9
    )
    err = (y_cuda - y_ref).abs()
    print("\n=== correctness vs UnpackedModel ===")
    print(f"  max|e|={err.max().item():.4e}  mean|e|={err.mean().item():.4e}")
    print(f"  cosine: min={cos.min().item():.6f} mean={cos.mean().item():.6f} "
          f"max={cos.max().item():.6f}")
    if cos.min().item() < 0.999:
        print("FAIL: cosine too low")
        return 1
    print("OK correctness (cosine ≥ 0.999)")

    # speed
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        model(ids, mask)
    _sync()
    ts = _timed(lambda: model(ids, mask), n=args.iters, warmup=8)
    peak = torch.cuda.max_memory_allocated() / 1024**2

    # CPU expand ref speed (same batch)
    ts_ref = []
    for _ in range(5):
        ref.forward(ids_cpu, mask_cpu)
    for _ in range(max(10, args.iters // 2)):
        t0 = time.perf_counter()
        ref.forward(ids_cpu, mask_cpu)
        ts_ref.append((time.perf_counter() - t0) * 1000)

    print("\n=== speed (batch of 4 texts) ===")
    print(f"  pack_cuda median={statistics.median(ts):.2f} ms  "
          f"({statistics.median(ts)/len(texts):.2f} ms/text)  peak_alloc≈{peak:.0f} MiB")
    print(f"  unpack_cpu  median={statistics.median(ts_ref):.2f} ms  "
          f"({statistics.median(ts_ref)/len(texts):.2f} ms/text)")
    print(f"  speedup vs expand-CPU: {statistics.median(ts_ref)/statistics.median(ts):.1f}x")

    # single-text warm
    one_ids = ids[:1]
    one_mask = mask[:1]
    for _ in range(5):
        model(one_ids, one_mask)
    ts1 = _timed(lambda: model(one_ids, one_mask), n=args.iters, warmup=10)
    print(f"  pack_cuda single: median={statistics.median(ts1):.2f} ms")

    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
