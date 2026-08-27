"""Ternlight `.bin` wire layout ↔ FlashTernary GEMM layout.

Keep in sync with ``engine/src/ternary_codec.rs``:

- Wire BitLinear pack: ``(out_features, in_features // 4)`` uint8
- Flash GEMM pack:     ``(in_features // 4, out_features)`` uint8
- Codes: +1→01, 0→00, −1→10 (2 bits, 4 weights / byte along ``in``)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def wire_oi_to_flash_io(packed_oi: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Transpose ``(out, in//4)`` → ``(in//4, out)`` without changing codes."""
    p = np.asarray(packed_oi, dtype=np.uint8)
    if p.ndim != 2:
        raise ValueError(f"expected 2D wire pack, got {p.shape}")
    return np.ascontiguousarray(p.T)


def flash_io_to_wire_oi(packed_io: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Transpose ``(in//4, out)`` → ``(out, in//4)``."""
    p = np.asarray(packed_io, dtype=np.uint8)
    if p.ndim != 2:
        raise ValueError(f"expected 2D flash pack, got {p.shape}")
    return np.ascontiguousarray(p.T)
