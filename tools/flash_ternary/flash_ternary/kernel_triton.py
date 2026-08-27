"""
Fused ternary GEMM on CUDA via Triton (pack maintained in VRAM).

Encoding compatible with abusalih-dev/FlashTernary (MIT):
  +1→01, 0→00, -1→10; 4 weights / uint8 along K.
  C = A @ dequant(B_packed) with B_packed shape (K//4, N).

Weights stay uint8 on device; dequant happens in SRAM registers only.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_ternary_matmul_kernel(
    A_ptr,
    B_packed_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    packed_k_idx = offs_k // 4
    sub_idx = offs_k % 4
    shifts = (sub_idx * 2).to(tl.int32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k

        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=a_mask,
            other=0.0,
        )

        packed_row = (k_start // 4) + packed_k_idx
        b_mask = (packed_row[:, None] < (K // 4)) & (offs_n[None, :] < N)
        packed = tl.load(
            B_packed_ptr
            + packed_row[:, None] * stride_bk
            + offs_n[None, :] * stride_bn,
            mask=b_mask,
            other=0,
        )

        packed_i32 = packed.to(tl.int32)
        enc = (packed_i32 >> shifts[:, None]) & 0x03
        w = (enc == 1).to(tl.float16) - (enc == 2).to(tl.float16)

        acc += tl.dot(a, w)

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.float16),
        mask=c_mask,
    )


def fused_ternary_matmul(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    K: int | None = None,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """C = A @ dequant(B_packed). A fp16 cuda (M,K), B_packed uint8 (K//4,N)."""
    assert A.is_cuda and B_packed.is_cuda
    assert A.dtype == torch.float16
    assert B_packed.dtype == torch.uint8
    assert A.is_contiguous() and B_packed.is_contiguous()

    M, K_a = A.shape
    Kp, N = B_packed.shape
    K = K if K is not None else K_a
    if K_a != K or Kp != K // 4:
        raise ValueError(f"shape mismatch A{tuple(A.shape)} B{tuple(B_packed.shape)} K={K}")
    if block_k % 4 != 0:
        raise ValueError("BLOCK_K must be divisible by 4")

    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fused_ternary_matmul_kernel[grid](
        A,
        B_packed,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B_packed.stride(0),
        B_packed.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return C
