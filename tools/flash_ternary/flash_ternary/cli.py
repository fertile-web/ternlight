"""CLI: check / bench with --backend numpy|torch|rust."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import available_backends, generate_random_ternary, pack_ternary, ternary_matmul, unpack_ternary
from .matmul import pack_ternary_rust


def cmd_check(args: argparse.Namespace) -> int:
    backends = [args.backend] if args.backend != "all" else available_backends()
    rng = np.random.default_rng(args.seed)
    K, N, M = args.K, args.N, args.M
    W = generate_random_ternary(K, N, rng)
    Bp = pack_ternary(W)
    Wu = unpack_ternary(Bp, K)
    if not np.array_equal(W.astype(np.int8), Wu.astype(np.int8)):
        print("FAIL pack/unpack roundtrip")
        return 1
    print(f"OK pack/unpack  compression={W.nbytes / Bp.nbytes:.1f}x vs f32")

    A = rng.standard_normal((M, K), dtype=np.float32)
    ref = A @ W  # W is already {-1,0,+1} f32
    A_i8 = np.clip(np.rint(A), -128, 127).astype(np.float32)
    ref_i8 = A_i8 @ W

    for b in backends:
        # rust auto/simd kernels treat A as BitLinear x_quant (i8).
        b_f32 = "rust-scalar" if b in ("rust", "rust-simd") else b
        try:
            C = ternary_matmul(A, Bp, backend=b_f32, K=K)
        except Exception as e:
            print(f"FAIL backend={b}: {e}")
            return 1
        err = float(np.max(np.abs(C - ref)))
        ok = err < 1e-3 * (1 + np.max(np.abs(ref)))
        print(f"{'OK' if ok else 'FAIL'} matmul backend={b} (f32)  max_abs_err={err:.3e}")
        if not ok:
            return 1
        if b in ("rust", "rust-simd", "rust-scalar"):
            try:
                C8 = ternary_matmul(A_i8, Bp, backend="rust-simd", K=K)
            except Exception as e:
                print(f"FAIL backend={b} simd/i8: {e}")
                return 1
            err8 = float(np.max(np.abs(C8 - ref_i8)))
            ok8 = err8 < 1e-3 * (1 + np.max(np.abs(ref_i8)))
            print(f"{'OK' if ok8 else 'FAIL'} matmul backend=rust-simd (i8 A)  max_abs_err={err8:.3e}")
            if not ok8:
                return 1
            Bp_r = pack_ternary_rust(W)
            if not np.array_equal(Bp, Bp_r):
                print("FAIL rust pack != numpy pack")
                return 1
            print("OK rust pack == numpy pack")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    K, N, M = args.K, args.N, args.M
    W = generate_random_ternary(K, N, rng)
    Bp = pack_ternary(W)
    A = rng.standard_normal((M, K), dtype=np.float32)
    # warmup
    for _ in range(2):
        ternary_matmul(A, Bp, backend=args.backend, K=K)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        C = ternary_matmul(A, Bp, backend=args.backend, K=K)
    dt = (time.perf_counter() - t0) / args.iters
    # ternary MAC count ≈ 2*M*N*K (mul+add); report GOPS loosely
    gops = (2.0 * M * N * K) / dt / 1e9
    print(
        f"backend={args.backend} M={M} K={K} N={N}  "
        f"latency={dt*1e3:.2f} ms  ~{gops:.1f} GOPS  out={C.shape}"
    )
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    print("available:", ", ".join(available_backends()))
    try:
        from .matmul import rust_simd_level

        print("rust_simd_hw:", rust_simd_level())
    except Exception as e:
        print("rust_simd_hw: (n/a)", e)
    return 0


def cmd_cos_gate(args: argparse.Namespace) -> int:
    """Delegate to cos_gate.main with remaining argv-style namespace."""
    from . import cos_gate

    argv: list[str] = []
    if args.bin:
        argv += ["--bin", str(args.bin)]
    if args.tok:
        argv += ["--tok", str(args.tok)]
    if args.ckpt:
        argv += ["--ckpt", str(args.ckpt)]
    if args.with_ckpt:
        argv.append("--with-ckpt")
    if args.cos_min is not None:
        argv += ["--cos-min", str(args.cos_min)]
    if args.no_cuda:
        argv.append("--no-cuda")
    if args.allow_skip:
        argv.append("--allow-skip")
    if args.backend:
        for b in args.backend:
            argv += ["--backend", b]
    return cos_gate.main(argv)


def cmd_compare(args: argparse.Namespace) -> int:
    from .bench import main as bench_main

    # strip leading "--" if user passed: compare -- --shapes ...
    rest = list(args.bench_args or [])
    if rest and rest[0] == "--":
        rest = rest[1:]
    return bench_main(rest)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="flash_ternary",
        description="Portable FlashTernary (CPU): pack + fused ternary matmul",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list backends")
    pl.set_defaults(func=cmd_list)

    pc = sub.add_parser("check", help="correctness vs dense matmul")
    pc.add_argument(
        "--backend",
        default="numpy",
        help="numpy | torch | rust | rust-scalar | rust-simd | all (default: numpy)",
    )
    pc.add_argument("--M", type=int, default=8)
    pc.add_argument("--K", type=int, default=256)
    pc.add_argument("--N", type=int, default=128)
    pc.add_argument("--seed", type=int, default=0)
    pc.set_defaults(func=cmd_check)

    pb = sub.add_parser("bench", help="microbenchmark (single backend)")
    pb.add_argument(
        "--backend",
        default="numpy",
        choices=["numpy", "torch", "rust", "rust-scalar", "rust-simd"],
    )
    pb.add_argument("--M", type=int, default=32)
    pb.add_argument("--K", type=int, default=1024)
    pb.add_argument("--N", type=int, default=1024)
    pb.add_argument("--iters", type=int, default=20)
    pb.add_argument("--seed", type=int, default=0)
    pb.set_defaults(func=cmd_bench)

    pcmp = sub.add_parser("compare", help="speed+accuracy table (see python -m flash_ternary.bench -h)")
    pcmp.add_argument("bench_args", nargs=argparse.REMAINDER, help="forwarded to flash_ternary.bench")
    pcmp.set_defaults(func=cmd_compare)

    pg = sub.add_parser(
        "cos-gate",
        help="pack embed cos≥threshold vs UnpackedModel (and optional pytorch ckpt)",
    )
    pg.add_argument("--bin", default=None)
    pg.add_argument("--tok", default=None)
    pg.add_argument("--ckpt", default=None)
    pg.add_argument("--with-ckpt", action="store_true")
    pg.add_argument("--cos-min", type=float, default=None)
    pg.add_argument("--backend", action="append", default=None)
    pg.add_argument("--no-cuda", action="store_true")
    pg.add_argument("--allow-skip", action="store_true")
    pg.set_defaults(func=cmd_cos_gate)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
