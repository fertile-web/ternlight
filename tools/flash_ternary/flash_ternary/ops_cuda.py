"""CUDA pack + TernaryLinear with Triton fused matmul (weights stay packed)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pack_ternary_torch(W: torch.Tensor) -> torch.Tensor:
    """Pack (K, N) ternary → (K//4, N) uint8. Same encoding as FlashTernary."""
    if W.ndim != 2:
        raise ValueError(f"expected 2D (K,N), got {tuple(W.shape)}")
    K, N = W.shape
    if K % 4 != 0:
        raise ValueError(f"K={K} must be divisible by 4")
    W_i = W.to(torch.int8)
    enc = torch.zeros(K, N, dtype=torch.uint8, device=W.device)
    enc = torch.where(W_i == 1, torch.tensor(1, dtype=torch.uint8, device=W.device), enc)
    enc = torch.where(W_i == -1, torch.tensor(2, dtype=torch.uint8, device=W.device), enc)
    enc = enc.reshape(K // 4, 4, N)
    packed = (
        enc[:, 0, :]
        | (enc[:, 1, :] << 2)
        | (enc[:, 2, :] << 4)
        | (enc[:, 3, :] << 6)
    )
    return packed.contiguous()


def unpack_ternary_torch(W_packed: torch.Tensor, K: int) -> torch.Tensor:
    """Unpack for reference checks → (K, N) float16."""
    Kp, N = W_packed.shape
    assert Kp == K // 4
    x = W_packed.to(torch.int32)
    enc = torch.stack(
        [(x >> 0) & 3, (x >> 2) & 3, (x >> 4) & 3, (x >> 6) & 3],
        dim=1,
    )
    W = (enc == 1).to(torch.float16) - (enc == 2).to(torch.float16)
    return W.reshape(K, N)


def ternarize_absmean(W: torch.Tensor) -> torch.Tensor:
    """AbsMean threshold → {-1,0,+1} (same recipe as FlashTernary.from_float)."""
    Wf = W.float()
    scale = Wf.abs().mean().clamp_min(1e-8)
    Ws = Wf / scale
    out = torch.zeros_like(Ws)
    out = torch.where(Ws > 0.5, torch.ones_like(out), out)
    out = torch.where(Ws < -0.5, -torch.ones_like(out), out)
    return out


class TernaryLinearTriton(nn.Module):
    """nn.Linear drop-in: weights stored as packed uint8 on CUDA; Triton forward.

    weight_packed layout: (in_features//4, out_features) for Y = X @ W^T.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features % 4 != 0:
            raise ValueError(f"in_features={in_features} must be divisible by 4")
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight_packed",
            torch.zeros(in_features // 4, out_features, dtype=torch.uint8),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @property
    def packed_bytes(self) -> int:
        return int(self.weight_packed.numel())

    @property
    def fp16_bytes(self) -> int:
        return self.in_features * self.out_features * 2

    def set_ternary_weights(self, W: torch.Tensor) -> None:
        """W: (out_features, in_features) values in {-1,0,+1}."""
        if W.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"expected {(self.out_features, self.in_features)}, got {tuple(W.shape)}"
            )
        # (in, out) = K, N for fused matmul
        Wt = W.t().contiguous()
        packed = pack_ternary_torch(Wt)
        self.weight_packed = packed.to(device=self.weight_packed.device, dtype=torch.uint8)

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "TernaryLinearTriton":
        layer = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        W_t = ternarize_absmean(linear.weight.data)
        layer.set_ternary_weights(W_t)
        if linear.bias is not None:
            layer.bias = nn.Parameter(linear.bias.data.to(torch.float16))
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("TernaryLinearTriton requires CUDA tensors")
        from .kernel_triton import fused_ternary_matmul

        if self.weight_packed.device != x.device:
            self.weight_packed = self.weight_packed.to(x.device)

        orig = x.shape
        x2 = x.reshape(-1, self.in_features).to(dtype=torch.float16).contiguous()
        Wp = self.weight_packed.contiguous()
        y = fused_ternary_matmul(x2, Wp, self.in_features)
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype, device=y.device)
        if len(orig) > 2:
            y = y.reshape(*orig[:-1], self.out_features)
        return y

    def forward_ref(self, x: torch.Tensor) -> torch.Tensor:
        """CPU/GPU reference: unpack then F.linear (for correctness)."""
        W = unpack_ternary_torch(self.weight_packed.cpu(), self.in_features)  # (in, out)
        W_lin = W.t().to(dtype=torch.float16, device=x.device)  # (out, in)
        return F.linear(x.to(torch.float16), W_lin, self.bias)

    def extra_repr(self) -> str:
        comp = self.fp16_bytes / max(self.packed_bytes, 1)
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"packed_bytes={self.packed_bytes}, compression={comp:.1f}x vs fp16"
        )
