"""
2-bit encoding for ternary weights {-1, 0, +1}.
Compatible with abusalih-dev/FlashTernary (MIT).

Encoding: +1 → 01,  0 → 00,  -1 → 10
Layout: 4 consecutive K values → one uint8 (bits 1-0 = w0, ..., 7-6 = w3).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pack_ternary(W: NDArray) -> NDArray[np.uint8]:
    """Pack (K, N) ternary → (K//4, N) uint8. K must be divisible by 4."""
    W = np.asarray(W)
    if W.ndim != 2:
        raise ValueError(f"expected 2D (K,N), got shape {W.shape}")
    K, N = W.shape
    if K % 4 != 0:
        raise ValueError(f"K={K} must be divisible by 4")

    enc = np.zeros((K, N), dtype=np.uint8)
    enc[W == 1] = 1
    enc[W == -1] = 2
    # leave 0 as 0
    enc = enc.reshape(K // 4, 4, N)
    packed = (
        enc[:, 0, :]
        | (enc[:, 1, :] << 2)
        | (enc[:, 2, :] << 4)
        | (enc[:, 3, :] << 6)
    )
    return packed.astype(np.uint8, copy=False)


def unpack_ternary(W_packed: NDArray[np.uint8], original_K: int | None = None) -> NDArray[np.float32]:
    """Unpack (K//4, N) uint8 → (K, N) float32 in {-1,0,+1}."""
    Wp = np.asarray(W_packed, dtype=np.uint8)
    if Wp.ndim != 2:
        raise ValueError(f"expected 2D packed, got {Wp.shape}")
    Kp, N = Wp.shape
    K = original_K if original_K is not None else Kp * 4
    if Kp != K // 4:
        raise ValueError(f"packed rows {Kp} != K//4 ({K // 4})")

    x = Wp.astype(np.int32)
    enc0 = (x >> 0) & 0x03
    enc1 = (x >> 2) & 0x03
    enc2 = (x >> 4) & 0x03
    enc3 = (x >> 6) & 0x03
    enc = np.stack([enc0, enc1, enc2, enc3], axis=1)  # (Kp, 4, N)
    # branchless: (enc==1) - (enc==2)
    W = (enc == 1).astype(np.float32) - (enc == 2).astype(np.float32)
    return W.reshape(K, N)


def generate_random_ternary(K: int, N: int, rng: np.random.Generator | None = None) -> NDArray[np.float32]:
    if rng is None:
        rng = np.random.default_rng(0)
    return rng.integers(-1, 2, size=(K, N)).astype(np.float32)
