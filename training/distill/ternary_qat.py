"""Ternary quantization-aware training — BitLinear swap, lambda schedule, health.

Wraps the `schneiderkamplab/bitlinear` library with a clean API for the trainer.
The library does the math (LayerNorm + activation int8 quant + ternary matmul +
rescale + bias); this file owns the orchestration: which layers to swap, when
to flip lambda from 0 → 1, and how to monitor for collapse.

Also hosts `ternarize_embedding_()`, which is post-training (not QAT proper),
but lives here so it shares the threshold formula with the BitLinear weight
quantization and can't drift. Used by `evaluate.py` and `pack/pack.py`.

Why the projection layer is excluded:
    The output projection (256 → 384) is the bridge between the student's
    internal space and the teacher's embedding space. It's small (~98K
    params, <1% of model) and quantization-sensitive — the postmortem
    (docs/training/postmortem-bitlinear-asymmetry.md) confirmed keeping it
    in fp32 is the locked decision.

Backend switch (`swap(..., backend=...)`):
    "bitlinear" (default) — schneiderkamplab/bitlinear, as above.
    "flash_qat" — tools/flash_ternary's QATBitLinear (shadow fp32 + STE,
        same math contract, with an optional 2-bit-pack matmul on the train
        path via `matmul="pack"`). Opt-in only (`TrainConfig.qat_backend`);
        the default training path never imports tools/flash_ternary. See
        tools/flash_ternary/README.md.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from bitlinear import BitLinear, replace_modules, set_lambda_

# Modules whose `name` matches this regex are EXCLUDED from the swap.
# Negative lookahead: "anything that does NOT contain 'projection'."
# Confirms 12 BitLinear instances after swap (6 per layer × 2 layers).
_SWAP_PATTERN = r"^(?!.*projection)"


def _flash_qat_module():
    """Lazy-import tools/flash_ternary.qat. Kept out of the top-level imports
    so the default (backend="bitlinear") training path has zero dependency
    on tools/flash_ternary — it's only touched when a config opts in.
    """
    try:
        import flash_ternary.qat as flash_qat
    except ImportError:
        tools_dir = Path(__file__).resolve().parents[2] / "tools" / "flash_ternary"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        import flash_ternary.qat as flash_qat
    return flash_qat


def _is_flash_qat_model(model: nn.Module) -> bool:
    # Name check (not isinstance) so this never forces the lazy import above.
    return any(type(m).__name__ == "QATBitLinear" for m in model.modules())


def swap(model: nn.Module, *, backend: str = "bitlinear", matmul: str = "dense") -> int:
    """Replace every nn.Linear in the model with a ternary-QAT layer, except
    the output projection. Sets lambda=0 (warmup mode) so the first epochs
    train in pass-through mode before ternary kicks in.

    The fp32 weights are preserved — the swapped layer's weight Parameter
    inherits the values from the original nn.Linear. These become the
    "shadow weights" that the optimizer updates during QAT.

    `backend="bitlinear"` (default): schneiderkamplab/bitlinear's BitLinear.
    `backend="flash_qat"`: flash_ternary.qat.QATBitLinear; `matmul` selects
        `"dense"` (reference F.linear) or `"pack"` (2-bit codec + matmul,
        exercises the same wire format the engine ships).

    Returns the number of layers swapped (sanity check: expect 12 for micro).
    """
    if backend == "bitlinear":
        replace_modules(model, match_name=_SWAP_PATTERN)
        set_lambda_(model, 0.0)
        return sum(1 for m in model.modules() if isinstance(m, BitLinear))
    if backend == "flash_qat":
        flash_qat = _flash_qat_module()
        return flash_qat.swap_to_qat_bitlinear(
            model, matmul=matmul, lambda_=0.0, match_name=_SWAP_PATTERN
        )
    raise ValueError(f"unknown qat backend {backend!r} (expected 'bitlinear' or 'flash_qat')")


def set_lambda(model: nn.Module, value: float) -> None:
    """Set the QAT lambda on every ternary-QAT layer in the model (either
    backend — auto-detected from what `swap()` put there).

    lambda=0 → pass-through (no weight quantization). Used during the QAT
              warmup epochs.
    lambda=1 → full ternary forward. Weights snap to {-1, 0, +1} on every
              forward pass; backward via Straight-Through Estimator updates
              the underlying fp32 shadow weights.
    """
    if _is_flash_qat_model(model):
        _flash_qat_module().set_lambda_(model, value)
    else:
        set_lambda_(model, value)


def zero_fractions(model: nn.Module) -> dict[str, float]:
    """Per-BitLinear-layer fraction of weights that snap to zero under the
    AbsMean threshold.

    Returns dict mapping module name → zero fraction in [0, 1].

    Healthy:  20–40% zeros — using all three states {-1, 0, +1}
    Watch:    40–60% — model leaning sparse, may still recover
    Collapse: >60% with flat loss — layer is dying, stop and debug

    POC scaled run finished at ~44% average — borderline but recoverable.
    """
    module_type = _flash_qat_module().QATBitLinear if _is_flash_qat_model(model) else BitLinear
    fracs: dict[str, float] = {}
    for name, module in model.named_modules():
        if isinstance(module, module_type):
            w = module.weight.detach()
            scale = w.abs().mean()
            fracs[name] = (w.abs() < 0.5 * scale).float().mean().item()
    return fracs


def int8_quantize_embedding_(model: nn.Module, mode: str = "per_row") -> dict[str, float]:
    """Post-training: int8-quantize the embedding table in place (per-row scale).

    1. Per-row scale s_i = max(|W_i|) / 127
    2. q_i = round(W_i / s_i).clamp(-128, 127)
    3. Replace W_i with (q_i * s_i) — the exact int8-representable value

    The weight tensor stays fp32 dtype for downstream PyTorch compatibility;
    only the *values* become int8-representable. Forward-pass numerics now
    match what the engine's int8 loader will produce from a packed .bin.

    Per-row scaling preserves dynamic range per token (different tokens have
    wildly different magnitudes); per-tensor scaling collapses that range.
    Skips the padding row (index 0), matching ternarize_embedding_().

    Returns {scale_min, scale_max, scale_mean} for sanity logging.

    Used by evaluate.py (Phase 4 ablation) and pack/pack.py (Phase 5 emit).
    """
    if mode != "per_row":
        raise NotImplementedError(f"int8 quantization mode {mode!r} not supported (use 'per_row')")

    with torch.no_grad():
        w = model.embedding.weight
        emb = w[1:]   # skip padding row
        scales = (emb.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        q = (emb / scales.unsqueeze(1)).round().clamp(-128, 127)
        w[1:] = q * scales.unsqueeze(1)
    return {
        "scale_min":  scales.min().item(),
        "scale_max":  scales.max().item(),
        "scale_mean": scales.mean().item(),
    }


def int4_quantize_embedding_(model: nn.Module) -> dict[str, float]:
    """Post-training: int4-quantize the embedding table in place (per-row scale).

    Same shape as int8_quantize_embedding_() but range [-7, 7]. Mirrors
    pack/encoders.encode_embedding_int4's quant math — moved here from
    pack.py so eval and pack share one implementation (the single-source
    contract from tern-training-pipeline.md Phase 5).

    Returns {scale_min, scale_max, scale_mean} for sanity logging.
    """
    with torch.no_grad():
        w = model.embedding.weight
        emb = w[1:]   # skip padding row
        scales = (emb.abs().amax(dim=1) / 7.0).clamp(min=1e-8)
        q = (emb / scales.unsqueeze(1)).round().clamp(-7, 7)
        w[1:] = q * scales.unsqueeze(1)
    return {
        "scale_min":  scales.min().item(),
        "scale_max":  scales.max().item(),
        "scale_mean": scales.mean().item(),
    }


def ternarize_embedding_(model: nn.Module) -> dict[str, float]:
    """Post-training: snap the embedding table to ternary {-1, 0, +1} in place.

    `nn.Embedding` is NOT touched during QAT — the BitLinear swap only covers
    nn.Linear instances. But the shipped `.bin` will have a ternary embedding
    (82% of params would otherwise blow the size budget), so eval/pack-time
    must apply the same projection to honestly measure what we'll ship.

    Uses the same AbsMean threshold formula as BitLinear's weight quantization
    (intentional — shared math, shared file). Skips the padding row (index 0)
    which must stay zero.

    Returns {scale, zero_fraction} for sanity logging.

    Used by evaluate.py (before Tasks 1-3) and pack/pack.py (before writing .bin).
    """
    with torch.no_grad():
        w = model.embedding.weight
        emb = w[1:]   # skip padding row — it should stay zero
        scale = emb.abs().mean().item()
        threshold = 0.5 * scale
        ternary = torch.sign(emb) * (emb.abs() > threshold).float()
        w[1:] = ternary
        zero_frac = (emb == 0).float().mean().item()
    return {"scale": scale, "zero_fraction": zero_frac}
