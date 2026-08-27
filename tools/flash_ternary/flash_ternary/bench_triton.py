"""Correctness + speed: Triton pack-CUDA vs dense fp16 vs expand-then-linear.

Usage:
  cd tools/flash_ternary   # from the repo root
  PYTHONPATH=. python3 -m flash_ternary.bench_triton
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from .ops_cuda import TernaryLinearTriton, pack_ternary_torch, ternarize_absmean, unpack_ternary_torch
from .kernel_triton import fused_ternary_matmul


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, n=50, warmup=10):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=32)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--N", type=int, default=1024)
    p.add_argument("--shapes", default="8x512x512,32x1024x1024,8x4096x4096,64x2048x2048")
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--warmup", type=int, default=12)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM free={free/1024**2:.0f} / total={total/1024**2:.0f} MiB")

    shapes = []
    for part in args.shapes.split(","):
        m, k, n = part.strip().lower().split("x")
        shapes.append((int(m), int(k), int(n)))

    print("\n=== correctness (random ternary) ===")
    for M, K, N in [(8, 256, 128), (32, 1024, 1024), (7, 512, 768)]:
        if K % 4:
            continue
        W = torch.randint(-1, 2, (K, N), device=device).to(torch.float16)
        Bp = pack_ternary_torch(W)
        A = torch.randn(M, K, device=device, dtype=torch.float16)
        # first call compiles kernel
        C = fused_ternary_matmul(A, Bp, K)
        ref = A @ W
        err = (C.float() - ref.float()).abs()
        rel = err.max().item() / (ref.float().abs().max().item() + 1e-6)
        print(
            f"  {M}x{K}x{N}: max|e|={err.max().item():.3e} rel={rel:.3e} "
            f"packed={Bp.numel()}B vs fp16={K*N*2}B ({K*N*2/Bp.numel():.1f}x)"
        )
        assert rel < 1e-2 or err.max().item() < 0.2, "correctness failed"

    # TernaryLinear module
    lin = torch.nn.Linear(1024, 1024, bias=True).cuda().half()
    with torch.no_grad():
        lin.weight.copy_(ternarize_absmean(lin.weight).half())
    tlin = TernaryLinearTriton.from_float(lin).cuda()
    x = torch.randn(16, 1024, device=device, dtype=torch.float16)
    y_t = tlin(x)
    y_r = tlin.forward_ref(x)
    err = (y_t.float() - y_r.float()).abs().max().item()
    rel = err / (y_r.float().abs().max().item() + 1e-6)
    print(
        f"  TernaryLinearTriton vs unpack-ref: max|e|={err:.3e} rel={rel:.3e}  {tlin.extra_repr()}"
    )
    assert rel < 1e-2 or err < 0.25

    print("\n=== speed + VRAM (warm) ===")
    print(
        f"{'shape':16s} {'triton_ms':>10s} {'dense_ms':>10s} {'expand_ms':>10s} "
        f"{'speedup':>8s} {'pack_MiB':>9s} {'fp16_MiB':>9s}"
    )

    for M, K, N in shapes:
        if K % 4:
            print(f"skip {M}x{K}x{N} (K%4!=0)")
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        W = torch.randint(-1, 2, (K, N), device=device).to(torch.float16)
        Bp = pack_ternary_torch(W).contiguous()
        A = torch.randn(M, K, device=device, dtype=torch.float16).contiguous()

        # compile / warmup triton once per shape
        for _ in range(3):
            fused_ternary_matmul(A, Bp, K)
        _sync()

        # peak for pack path
        torch.cuda.reset_peak_memory_stats()
        _ = fused_ternary_matmul(A, Bp, K)
        _sync()
        peak_pack = torch.cuda.max_memory_allocated() / 1024**2

        ts_t = _timed(lambda: fused_ternary_matmul(A, Bp, K), n=args.iters, warmup=args.warmup)
        # dense fp16 matmul with already-ternary W (ideal dense floor)
        ts_d = _timed(lambda: A @ W, n=args.iters, warmup=args.warmup)
        # expand-then-matmul each time (worst case of unpack in HBM)
        def expand_mm():
            Wu = unpack_ternary_torch(Bp, K).to(device=device, dtype=torch.float16)
            return A @ Wu

        ts_e = _timed(expand_mm, n=max(10, args.iters // 2), warmup=5)

        mt, md, me = statistics.median(ts_t), statistics.median(ts_d), statistics.median(ts_e)
        pack_mib = Bp.numel() / 1024**2
        fp16_mib = K * N * 2 / 1024**2
        print(
            f"{M}x{K}x{N:<10} {mt:10.3f} {md:10.3f} {me:10.3f} "
            f"{md/mt:8.2f}x {pack_mib:9.3f} {fp16_mib:9.3f}"
        )

    print("\n=== TernaryLinearTriton vs nn.Linear (ternarized weights, expand) ===")
    for in_f, out_f, bs in [(512, 512, 32), (1024, 1024, 32), (2048, 2048, 16)]:
        torch.cuda.empty_cache()
        base = torch.nn.Linear(in_f, out_f, bias=True).cuda().half()
        with torch.no_grad():
            base.weight.copy_(ternarize_absmean(base.weight).half())
        tlayer = TernaryLinearTriton.from_float(base).cuda()
        x = torch.randn(bs, in_f, device=device, dtype=torch.float16)
        for _ in range(5):
            tlayer(x)
            base(x)
        ts_t = _timed(lambda: tlayer(x), n=30, warmup=10)
        # expand weight once then linear (still stores full W)
        W_full = unpack_ternary_torch(tlayer.weight_packed.cpu(), in_f).t().cuda().half()
        ts_l = _timed(
            lambda: F.linear(x, W_full, tlayer.bias),
            n=30,
            warmup=10,
        )
        print(
            f"  Linear {in_f}→{out_f} bs={bs}: triton={statistics.median(ts_t):.3f} ms  "
            f"expand+linear={statistics.median(ts_l):.3f} ms  "
            f"pack={tlayer.packed_bytes/1024**2:.3f}MiB vs fp16={tlayer.fp16_bytes/1024**2:.3f}MiB"
        )

    print("\nALL OK")


if __name__ == "__main__":
    main()
