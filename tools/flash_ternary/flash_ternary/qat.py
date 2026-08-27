"""QAT-oriented BitLinear with optional pack-maintained matmul.

Design (see ``docs/notes/qat-packed-backprop.md``):

* **Parameter** = fp32 shadow weight (same as library BitLinear QAT).
* **Forward** ternarizes via STE (AbsMedian, range (-1,1)); activations get
  parameterless LN + AbsMax int8 + STE round (engine / unpack contract).
* **Matmul** modes:
  - ``dense`` — ``F.linear`` on ternary values (reference).
  - ``pack``  — pack to Flash layout ``(in//4, out)`` then matmul (values must
    match dense; exercises the 2-bit codec on the train path).
* **Backward** always uses dense ``W_t`` for ``dX`` / ``dW`` (v1). Pack does
  not become the learnable state.

This module is the train-side counterpart to pack-maintained inference
(``cpu_embed`` / ``cuda_embed``). It is **not** yet wired into
``training/distill`` by default.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops_cuda import pack_ternary_torch, unpack_ternary_torch

_BITLINEAR_EPS = 1e-5
_ACTIVATION_RANGE_MAX = 128.0
_WEIGHT_RANGE = (-1.0, 1.0)

MatmulMode = Literal["dense", "pack"]


def ste_round_clamp(
    x: torch.Tensor,
    lo: float = -1.0,
    hi: float = 1.0,
    lambda_: float = 1.0,
) -> torch.Tensor:
    """Straight-through round+clamp (library BitLinear ``round_clamp``)."""
    q = x.round().clamp(lo, hi)
    return lambda_ * (q - x).detach() + x


def abs_median_scale(w: torch.Tensor, eps: float = _BITLINEAR_EPS) -> torch.Tensor:
    """``s_w = 1 / AbsMedian(|w|)`` so scaled weights sit near ternary grid."""
    # BitLinear: max(|range|) / measure = 1 / AbsMedian for range (-1,1)
    med = w.detach().abs().median().clamp_min(eps)
    return 1.0 / med


def ternarize_weight_ste(
    w: torch.Tensor,
    *,
    lambda_: float = 1.0,
    eps: float = _BITLINEAR_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AbsMedian ternarize with STE. Returns ``(w_ternary, w_scale)``.

    Matches schneiderkamplab/bitlinear defaults used by
    ``training/distill/ternary_qat.py`` (measure AbsMedian, range (-1,1)).
    ``w_scale`` is detached (same as library ``scale(..., input.detach())``).
    """
    w_scale = abs_median_scale(w, eps=eps)
    # library: strategy(weight * w_scale, range, lambda_)
    w_q = ste_round_clamp(w * w_scale, _WEIGHT_RANGE[0], _WEIGHT_RANGE[1], lambda_)
    return w_q, w_scale


def bitlinear_pre_act_ste(
    x: torch.Tensor,
    *,
    lambda_: float = 1.0,
    eps: float = _BITLINEAR_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parameterless LN + AbsMax int8 quant with STE on the round.

    Returns ``(x_quant, x_scale)`` with shapes matching ``ternlight_layer``.
    """
    in_features = x.shape[-1]
    x_norm = F.layer_norm(x, (in_features,))
    x_max = x_norm.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    x_scale = _ACTIVATION_RANGE_MAX / x_max
    # STE on round+clamp so act quant does not zero grad
    x_q = ste_round_clamp(x_norm * x_scale, -128.0, 127.0, lambda_)
    return x_q, x_scale


class _PackedTernaryLinearFn(torch.autograd.Function):
    """``y = x @ W_t.T`` with optional pack-maintained forward values.

    ``W_t`` is already STE-quantized ternary (``out, in``). Backward:
    ``dx = dy @ W_t``, ``dW = dy.T @ x`` (STE already applied outside if needed).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_t: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        # Save for backward: use detached ternary values for dX (values only).
        # Grad w.r.t. w_t still flows via explicit dW = dy.T @ x.
        ctx.mode = mode
        ctx.save_for_backward(x, w_t)
        if mode == "dense":
            return F.linear(x, w_t, bias=None)
        if mode != "pack":
            raise ValueError(f"unknown matmul mode {mode!r}")
        # Flash layout: (K, N) = (in, out) = W_t.T ; pack (in//4, out)
        # Pack is non-diff; values of W_t define the matmul.
        w_kn = w_t.detach().transpose(0, 1).contiguous()  # (in, out)
        if w_kn.shape[0] % 4 != 0:
            # fall back dense if K not divisible by 4
            return F.linear(x, w_t, bias=None)
        packed = pack_ternary_torch(w_kn)  # (in//4, out)
        # unpack is float16 in ops_cuda; promote to x dtype for train numerics
        w_hat = unpack_ternary_torch(packed, w_kn.shape[0]).to(
            device=x.device, dtype=x.dtype
        )
        # w_hat is (in, out) → y = x @ W
        return x @ w_hat

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w_t = ctx.saved_tensors
        # Under bf16 autocast, dy/x may be BFloat16 while shadow W stays fp32.
        # Match GEMM dtypes; return dW in weight (fp32) dtype for the optimizer.
        w = w_t.to(dtype=dy.dtype)
        x_g = x.to(dtype=dy.dtype)
        # dx: (M,K) = dy (M,N) @ W_t (N,K)
        dx = dy @ w
        # dW: (N,K) = dy.T (N,M) @ x (M,K)
        dw = (dy.transpose(0, 1) @ x_g).to(dtype=w_t.dtype)
        return dx, dw, None


def packed_ternary_linear(
    x: torch.Tensor,
    w_t: torch.Tensor,
    *,
    mode: MatmulMode = "dense",
) -> torch.Tensor:
    """``y = x @ w_t.T`` with autograd; ``mode`` selects pack vs dense forward."""
    return _PackedTernaryLinearFn.apply(x, w_t, mode)


class QATBitLinear(nn.Module):
    """Trainable BitLinear: shadow weights + STE ternary + optional pack matmul.

    Drop-in shape-compatible with ``nn.Linear`` / library ``BitLinear`` for the
    layers distill swaps (projection excluded remains fp32 Linear).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        matmul: MatmulMode = "dense",
        lambda_: float = 1.0,
        eps: float = _BITLINEAR_EPS,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.matmul: MatmulMode = matmul
        self.lambda_ = float(lambda_)
        self.eps = float(eps)
        # Shadow weights (what the optimizer owns)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_lambda(self, value: float) -> None:
        self.lambda_ = float(value)

    def ternary_codes(self) -> torch.Tensor:
        """Detached ternary snap of current shadow (for zero-fraction monitors)."""
        with torch.no_grad():
            w_t, _ = ternarize_weight_ste(self.weight, lambda_=1.0, eps=self.eps)
        return w_t.round().clamp(-1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        x_flat = x.reshape(-1, self.in_features)

        x_q, x_scale = bitlinear_pre_act_ste(
            x_flat, lambda_=self.lambda_, eps=self.eps
        )
        w_t, w_scale = ternarize_weight_ste(
            self.weight, lambda_=self.lambda_, eps=self.eps
        )

        y_q = packed_ternary_linear(x_q, w_t, mode=self.matmul)
        if self.bias is not None:
            y_q = y_q + self.bias
        y = y_q / (w_scale * x_scale)

        if len(orig) > 2:
            y = y.reshape(*orig[:-1], self.out_features)
        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        matmul: MatmulMode = "dense",
        lambda_: float = 0.0,
    ) -> "QATBitLinear":
        """Clone an ``nn.Linear`` (or BitLinear) into a QATBitLinear shadow."""
        m = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            matmul=matmul,
            lambda_=lambda_,
        )
        with torch.no_grad():
            m.weight.copy_(linear.weight.detach().float())
            if m.bias is not None and linear.bias is not None:
                m.bias.copy_(linear.bias.detach().float())
        return m

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}, matmul={self.matmul!r}, "
            f"lambda_={self.lambda_}"
        )


def swap_to_qat_bitlinear(
    model: nn.Module,
    *,
    matmul: MatmulMode = "dense",
    lambda_: float = 0.0,
    match_name: str = r"^(?!.*projection)",
) -> int:
    """Replace ``nn.Linear`` modules with ``QATBitLinear`` (projection excluded).

    Same default exclusion as ``training/distill/ternary_qat.swap``. Returns
    number of modules replaced. Does **not** touch library BitLinear instances
    already present — use only on fp32 students before QAT.
    """
    import re

    n = 0

    def _rec(parent: nn.Module, prefix: str = "") -> None:
        nonlocal n
        for name, child in list(parent.named_children()):
            qual = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not isinstance(child, QATBitLinear):
                if re.search(match_name, qual) is not None:
                    setattr(
                        parent,
                        name,
                        QATBitLinear.from_linear(
                            child, matmul=matmul, lambda_=lambda_
                        ),
                    )
                    n += 1
                    continue
            _rec(child, qual)

    _rec(model)
    return n


def set_lambda_(model: nn.Module, value: float) -> None:
    """Set QAT lambda on every ``QATBitLinear`` in ``model``."""
    for m in model.modules():
        if isinstance(m, QATBitLinear):
            m.set_lambda(value)
