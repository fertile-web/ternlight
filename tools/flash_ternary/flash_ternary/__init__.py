"""Portable FlashTernary-compatible 2-bit pack + fused ternary GEMM.

Backends: numpy | torch | rust* | triton (CUDA pack-maintained).
Train/QAT: see ``qat`` (STE shadow + optional pack matmul).
"""

from .quant import pack_ternary, unpack_ternary, generate_random_ternary
from .matmul import ternary_matmul, available_backends
from .qat import QATBitLinear, packed_ternary_linear, swap_to_qat_bitlinear, set_lambda_

__all__ = [
    "pack_ternary",
    "unpack_ternary",
    "generate_random_ternary",
    "ternary_matmul",
    "available_backends",
    "QATBitLinear",
    "packed_ternary_linear",
    "swap_to_qat_bitlinear",
    "set_lambda_",
]
