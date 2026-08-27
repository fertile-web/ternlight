"""Fused ternary matmul backends: numpy | torch | rust | rust-scalar | rust-simd."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .quant import unpack_ternary

Backend = Literal["numpy", "torch", "rust", "rust-scalar", "rust-simd", "triton"]

# C ABI mode: 0=auto, 1=scalar, 2=simd
_RUST_MODE = {
    "rust": 0,
    "rust-auto": 0,
    "rust-scalar": 1,
    "rust-simd": 2,
}


def available_backends() -> list[str]:
    out = ["numpy"]
    try:
        import torch  # noqa: F401

        out.append("torch")
        if torch.cuda.is_available():
            try:
                import triton  # noqa: F401

                out.append("triton")
            except ImportError:
                pass
    except ImportError:
        pass
    if _rust_lib_path() is not None:
        out.extend(["rust", "rust-scalar", "rust-simd"])
    return out


def rust_simd_level() -> str:
    """Hardware SIMD level reported by the rust cdylib."""
    if _rust_lib_path() is None:
        return "unbuilt"
    lib = _load_rust()
    import ctypes

    lib.flash_ternary_simd_level.restype = ctypes.c_char_p
    s = lib.flash_ternary_simd_level()
    return s.decode() if s else "unknown"


def ternary_matmul(
    A: NDArray | object,
    B_packed: NDArray | object,
    *,
    backend: Backend | str = "numpy",
    K: int | None = None,
) -> NDArray[np.float32]:
    """C = A @ dequant(B_packed).

    Args:
        A: (M, K) float activations
        B_packed: (K//4, N) uint8
        backend: numpy | torch | rust | rust-scalar | rust-simd
        K: original K (defaults to A.shape[1])
    """
    backend = str(backend).lower()
    if backend == "numpy":
        return _matmul_numpy(A, B_packed, K=K)
    if backend == "torch":
        return _matmul_torch(A, B_packed, K=K)
    if backend == "triton":
        return _matmul_triton(A, B_packed, K=K)
    if backend in _RUST_MODE:
        return _matmul_rust(A, B_packed, K=K, mode=_RUST_MODE[backend])
    raise ValueError(f"unknown backend={backend!r}; choose from {available_backends()}")


def _as_numpy_A(A) -> NDArray[np.float32]:
    if hasattr(A, "detach"):
        A = A.detach().cpu().numpy()
    return np.asarray(A, dtype=np.float32)


def _as_numpy_B(B) -> NDArray[np.uint8]:
    if hasattr(B, "detach"):
        B = B.detach().cpu().numpy()
    return np.asarray(B, dtype=np.uint8)


def _matmul_numpy(A, B_packed, K: int | None) -> NDArray[np.float32]:
    """Fused-in-concept: unpack in small K-tiles then matmul (CPU reference).

    Full HBM-style fusion is GPU-specific; on CPU we tile K and never materialize
    the full dequant matrix when K is large (streaming unpack).
    """
    A = _as_numpy_A(A)
    Bp = _as_numpy_B(B_packed)
    if A.ndim != 2 or Bp.ndim != 2:
        raise ValueError("A and B_packed must be 2D")
    M, K_a = A.shape
    Kp, N = Bp.shape
    K = K if K is not None else K_a
    if K_a != K or Kp != K // 4:
        raise ValueError(f"shape mismatch A{A.shape} B{Bp.shape} K={K}")

    # Tile along K to avoid allocating full (K,N) when large
    tile = 256
    C = np.zeros((M, N), dtype=np.float32)
    for k0 in range(0, K, tile):
        k1 = min(k0 + tile, K)
        # need packed rows covering [k0, k1)
        p0, p1 = k0 // 4, (k1 + 3) // 4
        # unpack only this slice
        W_slice = unpack_ternary(Bp[p0:p1], original_K=(p1 - p0) * 4)
        # trim to [k0,k1) relative to p0*4
        rel0 = k0 - p0 * 4
        rel1 = rel0 + (k1 - k0)
        W = W_slice[rel0:rel1]  # (k1-k0, N)
        C += A[:, k0:k1] @ W
    return C


def _matmul_torch(A, B_packed, K: int | None) -> NDArray[np.float32]:
    import torch

    if not isinstance(A, torch.Tensor):
        A_t = torch.as_tensor(np.asarray(A, dtype=np.float32))
    else:
        A_t = A.detach().to(dtype=torch.float32, device="cpu")
    if not isinstance(B_packed, torch.Tensor):
        Bp = torch.as_tensor(np.asarray(B_packed, dtype=np.uint8))
    else:
        Bp = B_packed.detach().to(dtype=torch.uint8, device="cpu")

    M, K_a = A_t.shape
    Kp, N = Bp.shape
    K = K if K is not None else K_a
    if K_a != K or Kp != K // 4:
        raise ValueError(f"shape mismatch A{tuple(A_t.shape)} B{tuple(Bp.shape)} K={K}")

    # vectorized unpack (same as FlashTernary quant.unpack)
    x = Bp.to(torch.int32)
    enc = torch.stack(
        [(x >> 0) & 3, (x >> 2) & 3, (x >> 4) & 3, (x >> 6) & 3],
        dim=1,
    )  # (Kp, 4, N)
    W = (enc == 1).to(torch.float32) - (enc == 2).to(torch.float32)
    W = W.reshape(K, N)
    C = A_t @ W
    return C.numpy()


def _matmul_triton(A, B_packed, K: int | None) -> NDArray[np.float32]:
    """CUDA pack-maintained path (weights stay uint8 on device)."""
    import torch
    from .kernel_triton import fused_ternary_matmul
    from .ops_cuda import pack_ternary_torch

    if not torch.cuda.is_available():
        raise RuntimeError("triton backend requires CUDA")

    if isinstance(A, torch.Tensor):
        A_t = A.detach()
    else:
        A_t = torch.as_tensor(np.asarray(A, dtype=np.float32))
    if isinstance(B_packed, torch.Tensor):
        Bp = B_packed.detach()
    else:
        Bp = torch.as_tensor(np.asarray(B_packed, dtype=np.uint8))

    # If B is not packed uint8 of right shape, try pack from (K,N) ternary
    if Bp.dtype != torch.uint8 or (K is not None and Bp.shape[0] != K // 4):
        if Bp.ndim == 2 and (K is None or Bp.shape[0] == K):
            Bp = pack_ternary_torch(Bp.float())
        else:
            raise ValueError("triton expects B_packed uint8 (K//4, N)")

    A_t = A_t.to(device="cuda", dtype=torch.float16).contiguous()
    Bp = Bp.to(device="cuda", dtype=torch.uint8).contiguous()
    K_use = K if K is not None else A_t.shape[1]
    C = fused_ternary_matmul(A_t, Bp, K_use)
    return C.float().cpu().numpy()


_rust_lib = None


def _rust_lib_path() -> Path | None:
    env = os.environ.get("FLASH_TERNARY_LIB")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    root = Path(__file__).resolve().parents[1] / "rust" / "target"
    candidates = [
        root / "release" / "libflash_ternary_cpu.so",
        root / "release" / "libflash_ternary_cpu.dylib",
        root / "release" / "flash_ternary_cpu.dll",
        root / "debug" / "libflash_ternary_cpu.so",
        root / "debug" / "libflash_ternary_cpu.dylib",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_rust():
    global _rust_lib
    if _rust_lib is not None:
        return _rust_lib
    import ctypes

    path = _rust_lib_path()
    if path is None:
        raise RuntimeError(
            "rust backend not built. Run: "
            "cargo build --release --manifest-path rust/Cargo.toml"
        )
    lib = ctypes.CDLL(str(path))
    lib.flash_ternary_matmul_f32_ex.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    lib.flash_ternary_matmul_f32_ex.restype = ctypes.c_int
    lib.flash_ternary_matmul_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.flash_ternary_matmul_f32.restype = ctypes.c_int
    lib.flash_ternary_pack.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.flash_ternary_pack.restype = ctypes.c_int
    lib.flash_ternary_simd_level.restype = ctypes.c_char_p
    _rust_lib = lib
    return lib


def _matmul_rust(A, B_packed, K: int | None, mode: int = 0) -> NDArray[np.float32]:
    import ctypes

    # Hot path: skip copies when caller already passed C-contig f32 / u8.
    if isinstance(A, np.ndarray) and A.dtype == np.float32 and A.flags["C_CONTIGUOUS"]:
        A_c = A
    else:
        A_c = np.ascontiguousarray(_as_numpy_A(A))
    if (
        isinstance(B_packed, np.ndarray)
        and B_packed.dtype == np.uint8
        and B_packed.flags["C_CONTIGUOUS"]
    ):
        Bp = B_packed
    else:
        Bp = np.ascontiguousarray(_as_numpy_B(B_packed))
    M, K_a = A_c.shape
    Kp, N = Bp.shape
    K = K if K is not None else K_a
    if K_a != K or Kp != K // 4:
        raise ValueError(f"shape mismatch A{A_c.shape} B{Bp.shape} K={K}")
    C = np.empty((M, N), dtype=np.float32)  # rust fills fully; skip zeroing
    lib = _load_rust()
    rc = lib.flash_ternary_matmul_f32_ex(
        A_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Bp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M,
        N,
        K,
        ctypes.c_uint32(mode),
    )
    if rc != 0:
        raise RuntimeError(f"rust matmul failed rc={rc}")
    return C


def pack_ternary_rust(W: NDArray) -> NDArray[np.uint8]:
    """Optional: pack via rust (parity check)."""
    import ctypes

    W = np.ascontiguousarray(np.asarray(W, dtype=np.float32))
    K, N = W.shape
    if K % 4 != 0:
        raise ValueError("K % 4 != 0")
    out = np.zeros((K // 4, N), dtype=np.uint8)
    lib = _load_rust()
    rc = lib.flash_ternary_pack(
        W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        K,
        N,
    )
    if rc != 0:
        raise RuntimeError(f"rust pack failed rc={rc}")
    return out
