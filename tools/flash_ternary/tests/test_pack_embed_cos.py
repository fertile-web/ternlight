"""Pack embed cos gate — skips cleanly when `.bin` is absent (CI without artifacts)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_REPO_ROOT = Path(os.environ.get("TERNLIGHT_REPO") or Path(__file__).resolve().parents[3])

BIN = Path(
    os.environ.get(
        "TERNLIGHT_EN_BIN",
        str(_REPO_ROOT / "training" / "pack" / "out" / "en-minilm-int4.bin"),
    )
)
TOK = Path(
    os.environ.get(
        "TERNLIGHT_TOKENIZER",
        str(_REPO_ROOT / "engine" / "assets"),
    )
)
CKPT = Path(
    os.environ.get(
        "TERNLIGHT_CKPT",
        str(
            _REPO_ROOT
            / "training"
            / "distill"
            / "runs"
            / "en-qat-L3w512-rope-minilm-msmarcomix-3e1bb6d"
            / "checkpoint_ep12.pt"
        ),
    )
)
COS_MIN = float(os.environ.get("FLASH_COS_MIN", "0.999"))


def _have_bin_tok() -> bool:
    # TOK may be a tokenizer.json file itself, or a dir that contains one.
    tok_ok = TOK.is_file() or (TOK / "tokenizer.json").is_file()
    return BIN.is_file() and tok_ok


@pytest.mark.skipif(not _have_bin_tok(), reason="en-minilm bin or tokenizer.json missing")
def test_pack_cpu_blas_vs_unpacked_cos():
    from flash_ternary.cos_gate import run_gate

    results = run_gate(
        bin_path=BIN,
        tok_path=TOK,
        queries=[
            "BitLinear ternary 2-bit packed weights RoPE FlashTernary",
            "engine int4 pack format SubLN",
        ],
        cos_min=COS_MIN,
        backends=["blas"],
        ckpt=None,
        include_cuda=False,
    )
    by = {r.name: r for r in results}
    r = by["pack_cpu/blas"]
    assert r.ok, f"cos={r.cos} max_abs={r.max_abs} {r.detail}"
    assert r.cos >= COS_MIN


@pytest.mark.skipif(not _have_bin_tok(), reason="en-minilm bin or tokenizer.json missing")
def test_pack_cpu_rust_vs_unpacked_cos():
    from flash_ternary import available_backends
    from flash_ternary.cos_gate import run_gate

    if "rust" not in available_backends():
        pytest.skip("rust cdylib not built")
    results = run_gate(
        bin_path=BIN,
        tok_path=TOK,
        queries=["BitLinear ternary 2-bit packed weights RoPE FlashTernary"],
        cos_min=COS_MIN,
        backends=["rust"],
        ckpt=None,
        include_cuda=False,
    )
    r = results[0]
    assert r.ok, f"{r.name} cos={r.cos} {r.detail}"


@pytest.mark.skipif(not _have_bin_tok(), reason="en-minilm bin or tokenizer.json missing")
def test_pack_cuda_vs_unpacked_cos():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from flash_ternary.cos_gate import run_gate

    results = run_gate(
        bin_path=BIN,
        tok_path=TOK,
        queries=["BitLinear ternary 2-bit packed weights RoPE FlashTernary"],
        cos_min=COS_MIN,
        backends=[],  # skip cpu arms for speed
        ckpt=None,
        include_cuda=True,
    )
    by = {r.name: r for r in results}
    assert "pack_cuda" in by
    r = by["pack_cuda"]
    if r.detail.startswith("skipped"):
        pytest.skip(r.detail)
    assert r.ok, f"cos={r.cos} max_abs={r.max_abs} {r.detail}"


@pytest.mark.skipif(
    not (_have_bin_tok() and CKPT.is_file()),
    reason="bin/tok/ckpt missing",
)
def test_pack_blas_vs_pytorch_ckpt_cos():
    """Optional promote gate: pack_cpu/blas vs load_for_eval BitLinear."""
    from flash_ternary.cos_gate import run_gate

    results = run_gate(
        bin_path=BIN,
        tok_path=TOK,
        queries=["BitLinear ternary 2-bit packed weights RoPE FlashTernary"],
        cos_min=COS_MIN,
        backends=["blas"],
        ckpt=CKPT,
        include_cuda=False,
    )
    by = {r.name: r for r in results}
    assert "pack_cpu/blas_vs_pytorch" in by
    r = by["pack_cpu/blas_vs_pytorch"]
    assert r.ok, f"cos={r.cos} max_abs={r.max_abs} {r.detail}"
