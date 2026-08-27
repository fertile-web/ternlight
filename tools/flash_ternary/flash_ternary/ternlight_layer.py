"""Full ternlight BitLinear from `.bin` on CUDA (pack-maintained Triton matmul).

Matches `training/pack/unpack.py::bitlinear_forward` / engine `kernels::bitlinear_forward`:

  1. parameterless LayerNorm (eps=1e-5)
  2. per-token AbsMax → x_scale = 128 / max|x_norm|
  3. x_quant = round(x_norm * x_scale).clamp(-128, 127)
  4. y_quant = x_quant @ W_ternary.T + bias     # W pack stays uint8 on GPU
  5. y = y_quant / (w_scale * x_scale)

Only the weight matrix is packed; LN/act-quant run in PyTorch (still cheap vs expand).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bin_loader import load_bitlinear, unpack_oi
from .kernel_triton import fused_ternary_matmul

_BITLINEAR_EPS = 1e-5
_ACTIVATION_RANGE_MAX = 128.0


def bitlinear_pre_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Steps 1–3: LN + AbsMax quant. Returns (x_quant float, x_scale)."""
    in_features = x.shape[-1]
    x_norm = torch.layer_norm(x, [in_features])
    x_max = x_norm.abs().amax(dim=-1, keepdim=True).clamp(min=_BITLINEAR_EPS)
    x_scale = _ACTIVATION_RANGE_MAX / x_max
    x_quant = (x_norm * x_scale).round().clamp(-128, 127)
    return x_quant, x_scale


def bitlinear_forward_ref(
    x: torch.Tensor,
    w_quant: torch.Tensor,
    w_scale: float,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Pure PyTorch reference (same as training/pack/unpack.bitlinear_forward)."""
    x_quant, x_scale = bitlinear_pre_act(x)
    y_quant = F.linear(x_quant, w_quant.float(), bias)
    return y_quant / (w_scale * x_scale)


class BitLinearTritonFromBin(nn.Module):
    """CUDA: weights as uint8 pack (in//4, out); full BitLinear forward via Triton matmul."""

    def __init__(
        self,
        weight_packed_io: torch.Tensor,
        w_scale: float,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
        name: str = "bitlinear",
    ):
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.w_scale = float(w_scale)
        self.register_buffer("weight_packed", weight_packed_io.contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.float().contiguous())
        else:
            self.bias = None

    @classmethod
    def from_bin(
        cls,
        path: str | Path,
        *,
        layer: int = 0,
        name: str = "fc1",
        device: str = "cuda",
    ) -> "BitLinearTritonFromBin":
        bl = load_bitlinear(path, layer=layer, name=name, device=device)
        return cls(
            bl.weight_packed_io,
            bl.w_scale,
            bl.bias,
            bl.in_features,
            bl.out_features,
            name=bl.name,
        )

    @property
    def packed_bytes(self) -> int:
        return int(self.weight_packed.numel())

    @property
    def fp16_weight_bytes(self) -> int:
        return self.in_features * self.out_features * 2

    def w_quant_matrix(self) -> torch.Tensor:
        """(out, in) ternary float32 — for reference only."""
        packed_oi = self.weight_packed.t().contiguous().cpu()
        return unpack_oi(packed_oi, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("BitLinearTritonFromBin requires CUDA")
        if self.weight_packed.device != x.device:
            self.weight_packed = self.weight_packed.to(x.device)

        orig = x.shape
        x_flat = x.reshape(-1, self.in_features).float()
        x_quant, x_scale = bitlinear_pre_act(x_flat)

        # Triton: y_raw = x_quant @ W_in_out  (W pack maintained)
        x_h = x_quant.to(dtype=torch.float16).contiguous()
        Wp = self.weight_packed.contiguous()
        y_raw = fused_ternary_matmul(x_h, Wp, self.in_features).float()

        if self.bias is not None:
            y_raw = y_raw + self.bias.to(device=y_raw.device, dtype=y_raw.dtype)

        y = y_raw / (self.w_scale * x_scale)
        if len(orig) > 2:
            y = y.reshape(*orig[:-1], self.out_features)
        return y

    def forward_ref(self, x: torch.Tensor) -> torch.Tensor:
        """Match unpack.bitlinear_forward (expand W once)."""
        W = self.w_quant_matrix().to(device=x.device)
        bias = None if self.bias is None else self.bias.to(device=x.device)
        return bitlinear_forward_ref(x.float(), W, self.w_scale, bias)

    def extra_repr(self) -> str:
        return (
            f"{self.name} in={self.in_features} out={self.out_features} "
            f"scale={self.w_scale:.4g} packed={self.packed_bytes}B "
            f"({self.fp16_weight_bytes / max(self.packed_bytes, 1):.1f}x vs fp16) "
            f"bias={self.bias is not None} full_bitlinear=LN+act_quant+triton"
        )
