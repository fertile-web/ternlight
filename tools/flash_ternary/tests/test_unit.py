"""Always-on unit tests (no .bin / GPU required)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flash_ternary import (  # noqa: E402
    available_backends,
    generate_random_ternary,
    pack_ternary,
    ternary_matmul,
    unpack_ternary,
)


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    W = generate_random_ternary(256, 128, rng)
    Bp = pack_ternary(W)
    Wu = unpack_ternary(Bp, 256)
    assert np.array_equal(W.astype(np.int8), Wu.astype(np.int8))
    assert Bp.shape == (64, 128)
    assert W.nbytes / Bp.nbytes == 16.0  # f32 → 2-bit


@pytest.mark.parametrize("backend", ["numpy"])
def test_matmul_numpy_vs_dense(backend):
    rng = np.random.default_rng(1)
    K, N, M = 256, 128, 8
    W = generate_random_ternary(K, N, rng)
    Bp = pack_ternary(W)
    A = rng.standard_normal((M, K), dtype=np.float32)
    ref = A @ W
    C = ternary_matmul(A, Bp, backend=backend, K=K)
    err = float(np.max(np.abs(C - ref)))
    assert err < 1e-4 * (1 + float(np.max(np.abs(ref)))), err


def test_rust_matmul_if_built():
    if "rust" not in available_backends():
        pytest.skip("rust cdylib not built")
    rng = np.random.default_rng(2)
    K, N, M = 256, 128, 8
    W = generate_random_ternary(K, N, rng)
    Bp = pack_ternary(W)
    A = rng.standard_normal((M, K), dtype=np.float32)
    ref = A @ W
    # Auto SIMD treats A as BitLinear x_quant (i8). General f32 uses scalar.
    C = ternary_matmul(A, Bp, backend="rust-scalar", K=K)
    err = float(np.max(np.abs(C - ref)))
    assert err < 1e-4 * (1 + float(np.max(np.abs(ref)))), err
    A_i8 = np.clip(np.rint(A), -128, 127).astype(np.float32)
    ref_i8 = A_i8 @ W
    C_s = ternary_matmul(A_i8, Bp, backend="rust-simd", K=K)
    err_s = float(np.max(np.abs(C_s - ref_i8)))
    assert err_s < 1e-3 * (1 + float(np.max(np.abs(ref_i8)))), err_s


def test_wire_layout_transpose_roundtrip():
    from flash_ternary.wire_layout import flash_io_to_wire_oi, wire_oi_to_flash_io

    rng = np.random.default_rng(0)
    oi = rng.integers(0, 256, size=(7, 16), dtype=np.uint8)
    io = wire_oi_to_flash_io(oi)
    assert io.shape == (16, 7)
    assert np.array_equal(flash_io_to_wire_oi(io), oi)


def test_fast_spm_parity_with_hf():
    import os

    repo_root = Path(os.environ.get("TERNLIGHT_REPO") or Path(__file__).resolve().parents[3])
    tok_dir = repo_root / "engine" / "assets"
    if not (tok_dir / "tokenizer.json").is_file():
        pytest.skip("engine/assets/tokenizer.json not present")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    from flash_ternary.spm_tok import FastSpmTokenizer

    hf = AutoTokenizer.from_pretrained(str(tok_dir))
    if hf.pad_token is None:
        # engine/assets ships tokenizer.json only (no tokenizer_config.json);
        # [PAD] is id 0 in the vocab (added_tokens_decoder) but AutoTokenizer
        # needs it named explicitly to allow padding=True below.
        hf.pad_token = "[PAD]"
    ft = FastSpmTokenizer.from_pretrained(tok_dir)
    qs = [
        "BitLinear ternary pack",
        "short",
        "engine int4 " * 30,
    ]
    a = hf(qs, padding=True, truncation=True, max_length=128, return_tensors="np")
    b_ids, b_mask = ft.encode_batch_np(qs, max_len=128)
    assert np.array_equal(a["input_ids"], b_ids)
    assert np.array_equal(a["attention_mask"], b_mask)
