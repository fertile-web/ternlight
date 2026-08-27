"""Speed + accuracy benchmark across backends.

Usage:
  PYTHONPATH=. python3 -m flash_ternary.bench
  PYTHONPATH=. python3 -m flash_ternary.bench --shapes 32x1024x1024,8x4096x4096
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from . import generate_random_ternary, pack_ternary, ternary_matmul
from .matmul import available_backends, rust_simd_level


@dataclass
class Row:
    backend: str
    shape: str
    latency_ms: float
    gops: float
    max_abs: float
    mean_abs: float
    rel_max: float
    ok: bool


def _ref_dense_f64(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """High-precision reference: f64 dense matmul."""
    return (A.astype(np.float64) @ W.astype(np.float64)).astype(np.float64)


def _metrics(C: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    d = np.abs(C.astype(np.float64) - ref)
    max_abs = float(d.max())
    mean_abs = float(d.mean())
    scale = float(np.max(np.abs(ref))) + 1e-12
    rel_max = max_abs / scale
    return max_abs, mean_abs, rel_max


def _bench_one(
    backend: str,
    A: np.ndarray,
    Bp: np.ndarray,
    ref: np.ndarray,
    K: int,
    iters: int,
    warmup: int,
) -> Row:
    M, _ = A.shape
    N = Bp.shape[1]
    shape = f"{M}x{K}x{N}"
    try:
        for _ in range(warmup):
            C = ternary_matmul(A, Bp, backend=backend, K=K)
        t0 = time.perf_counter()
        for _ in range(iters):
            C = ternary_matmul(A, Bp, backend=backend, K=K)
        dt = (time.perf_counter() - t0) / iters
        max_abs, mean_abs, rel_max = _metrics(C, ref)
        # accept tiny numerical noise
        ok = max_abs < 1e-3 * (1.0 + float(np.max(np.abs(ref))))
        gops = (2.0 * M * N * K) / dt / 1e9
        return Row(backend, shape, dt * 1e3, gops, max_abs, mean_abs, rel_max, ok)
    except Exception as e:
        return Row(backend, shape, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), False)


def run(
    shapes: list[tuple[int, int, int]],
    backends: list[str],
    iters: int,
    warmup: int,
    seed: int,
) -> list[Row]:
    rng = np.random.default_rng(seed)
    rows: list[Row] = []
    for M, K, N in shapes:
        W = generate_random_ternary(K, N, rng)
        Bp = pack_ternary(W)
        A = rng.standard_normal((M, K), dtype=np.float32)
        ref = _ref_dense_f64(A, W)
        for b in backends:
            rows.append(_bench_one(b, A, Bp, ref, K, iters, warmup))
    return rows


def _parse_shapes(s: str) -> list[tuple[int, int, int]]:
    out = []
    for part in s.split(","):
        part = part.strip().lower()
        if not part:
            continue
        m, k, n = part.split("x")
        out.append((int(m), int(k), int(n)))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="flash_ternary speed+accuracy bench")
    p.add_argument(
        "--shapes",
        default="8x256x256,32x1024x1024,16x2048x2048,8x4096x4096",
        help="comma list of MxKxN",
    )
    p.add_argument(
        "--backends",
        default="auto",
        help="comma list, or 'auto' for all available",
    )
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    shapes = _parse_shapes(args.shapes)
    if args.backends == "auto":
        backends = available_backends()
    else:
        backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    print(f"simd_hw={rust_simd_level()}")
    print(f"backends={backends}")
    print(f"shapes={shapes}  iters={args.iters} warmup={args.warmup}")
    print()

    rows = run(shapes, backends, args.iters, args.warmup, args.seed)

    # table
    hdr = f"{'backend':<14} {'shape':<16} {'ms':>10} {'GOPS':>8} {'max|e|':>12} {'mean|e|':>12} {'rel_max':>10} {'ok':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.backend:<14} {r.shape:<16} {r.latency_ms:10.3f} {r.gops:8.1f} "
            f"{r.max_abs:12.3e} {r.mean_abs:12.3e} {r.rel_max:10.3e} {'Y' if r.ok else 'N':>4}"
        )

    # summary per backend (geo-mean speedup vs rust-scalar if present, else vs first)
    print()
    basenames = {r.backend for r in rows}
    baseline = "rust-scalar" if "rust-scalar" in basenames else ("numpy" if "numpy" in basenames else next(iter(basenames)))
    print(f"speedup vs {baseline} (higher=faster):")
    for b in backends:
        ratios = []
        for M, K, N in shapes:
            shape = f"{M}x{K}x{N}"
            base = next((r for r in rows if r.backend == baseline and r.shape == shape), None)
            cur = next((r for r in rows if r.backend == b and r.shape == shape), None)
            if base and cur and base.latency_ms > 0 and cur.latency_ms > 0 and base.ok and cur.ok:
                ratios.append(base.latency_ms / cur.latency_ms)
        if ratios:
            g = float(np.exp(np.mean(np.log(ratios))))
            print(f"  {b:<14}  geomean ×{g:.2f}  (n={len(ratios)})")

    failed = [r for r in rows if not r.ok]
    if failed:
        print(f"\nFAIL: {len(failed)} rows")
        return 1
    print("\nALL OK (accuracy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
