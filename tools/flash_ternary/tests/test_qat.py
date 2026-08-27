"""QAT pack path: STE forward/backward + parity vs dense / library BitLinear."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flash_ternary.qat import (  # noqa: E402
    QATBitLinear,
    packed_ternary_linear,
    ste_round_clamp,
    ternarize_weight_ste,
)

# torch already imported above


def test_ste_round_clamp_forward_and_grad():
    x = torch.tensor([-1.7, -0.4, 0.0, 0.6, 1.2], requires_grad=True)
    y = ste_round_clamp(x, -1.0, 1.0, lambda_=1.0)
    # forward snaps
    assert torch.allclose(y.detach(), torch.tensor([-1.0, 0.0, 0.0, 1.0, 1.0]))
    y.sum().backward()
    # STE: grad ≈ 1
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_packed_ternary_linear_matches_dense_and_grads():
    torch.manual_seed(0)
    M, K, N = 5, 64, 32
    x = torch.randn(M, K, requires_grad=True)
    # exact ternary
    w_t = torch.randint(-1, 2, (N, K), dtype=torch.float32)
    w_t.requires_grad_(True)

    y_dense = packed_ternary_linear(x, w_t, mode="dense")
    y_pack = packed_ternary_linear(x, w_t, mode="pack")
    assert torch.allclose(y_dense, y_pack, atol=1e-5, rtol=1e-5)

    loss = y_dense.pow(2).mean()
    loss.backward()
    assert x.grad is not None and w_t.grad is not None
    # recompute expected grads
    dy = (2.0 / (M * N)) * y_dense.detach()
    dx_ref = dy @ w_t.detach()
    dw_ref = dy.T @ x.detach()
    assert torch.allclose(x.grad, dx_ref, atol=1e-5, rtol=1e-4)
    assert torch.allclose(w_t.grad, dw_ref, atol=1e-5, rtol=1e-4)


def test_qat_bitlinear_pack_matches_dense_forward():
    torch.manual_seed(1)
    m_d = QATBitLinear(64, 48, bias=True, matmul="dense", lambda_=1.0)
    m_p = QATBitLinear(64, 48, bias=True, matmul="pack", lambda_=1.0)
    with torch.no_grad():
        m_p.weight.copy_(m_d.weight)
        m_p.bias.copy_(m_d.bias)
    x = torch.randn(4, 64)
    y_d = m_d(x)
    y_p = m_p(x)
    err = (y_d - y_p).abs().max().item()
    assert err < 1e-4, err


def test_qat_bitlinear_backward_updates_shadow():
    torch.manual_seed(2)
    m = QATBitLinear(32, 16, bias=True, matmul="pack", lambda_=1.0)
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    x = torch.randn(8, 32)
    target = torch.randn(8, 16)
    losses = []
    for _ in range(8):
        opt.zero_grad()
        y = m(x)
        loss = (y - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    assert m.weight.grad is not None


def test_lambda_zero_is_passthrough_weight_path():
    """lambda_=0: strategy returns scaled weight (library warmup), not ternary.

    BitLinear: ``w_quant = strategy(w * s_w, …, λ)`` with λ=0 → ``w * s_w``,
    then ``y /= s_w`` cancels — not ``w_quant == w``.
    """
    torch.manual_seed(3)
    w = torch.randn(8, 16)
    w_q, s = ternarize_weight_ste(w, lambda_=0.0)
    assert torch.allclose(w_q, w * s)
    # not snapped
    assert (w_q.abs() > 1.5).any()


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("bitlinear") is None,
    reason="bitlinear not installed",
)
def test_parity_vs_library_bitlinear():
    """Same shadow weights → close forward to library BitLinear at lambda=1."""
    try:
        from bitlinear import BitLinear
    except ImportError:
        pytest.skip("bitlinear not importable")

    torch.manual_seed(4)
    in_f, out_f = 64, 32
    ref = BitLinear(
        in_f,
        out_f,
        bias=True,
        weight_range=1.58,
        weight_measure="AbsMedian",
        activation_range=8,
        activation_measure="AbsMax",
        lambda_=1.0,
    )
    ours = QATBitLinear(in_f, out_f, bias=True, matmul="dense", lambda_=1.0)
    with torch.no_grad():
        ours.weight.copy_(ref.weight)
        ours.bias.copy_(ref.bias)

    x = torch.randn(6, in_f)
    y_ref = ref(x)
    y_ours = ours(x)
    # STE + same scales: should match tightly in eval-style path
    cos = torch.nn.functional.cosine_similarity(
        y_ref.flatten(), y_ours.flatten(), dim=0
    ).item()
    max_abs = (y_ref - y_ours).abs().max().item()
    assert cos > 0.999, (cos, max_abs)
    assert max_abs < 1e-3 * (1 + y_ref.abs().max().item()), max_abs


def test_from_linear_and_swap_smoke():
    from flash_ternary.qat import set_lambda_, swap_to_qat_bitlinear

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 64)
            self.fc2 = nn.Linear(64, 16)
            self.projection = nn.Linear(16, 8)  # excluded by default pattern

        def forward(self, x):
            return self.projection(self.fc2(torch.relu(self.fc1(x))))

    m = Tiny()
    n = swap_to_qat_bitlinear(m, matmul="pack", lambda_=0.0)
    assert n == 2
    assert isinstance(m.fc1, QATBitLinear)
    assert isinstance(m.projection, nn.Linear)
    set_lambda_(m, 1.0)
    assert m.fc1.lambda_ == 1.0
    y = m(torch.randn(3, 32))
    assert y.shape == (3, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bf16 autocast")
def test_qat_bitlinear_bf16_autocast_backward():
    """H47: production train uses amp bf16; custom backward must match dtypes."""
    m = QATBitLinear(64, 32, bias=True, matmul="pack", lambda_=1.0).cuda()
    x = torch.randn(4, 64, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = m(x)
        loss = y.float().pow(2).mean()
    loss.backward()
    assert m.weight.grad is not None
    assert m.weight.grad.dtype == torch.float32
    assert torch.isfinite(m.weight.grad).all()
