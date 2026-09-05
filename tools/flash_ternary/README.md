# flash_ternary (portable)

Portable reimplementation of [abusalih-dev/FlashTernary](https://github.com/abusalih-dev/FlashTernary)
**2-bit pack + fused ternary GEMM**, without WebGPU / Triton.

Upstream is MIT (CUDA/Triton only). This tree targets **CPU** backends that fit
ternlight’s “CPU / mobile common path” constraint, with a backend switch:

| `--backend` | Runtime | Notes |
|---|---|---|
| `numpy` | pure NumPy | default; tile-unpack + BLAS |
| `torch` | PyTorch CPU | if installed |
| `rust` | Rust cdylib (auto SIMD) | `cargo build --release --manifest-path rust/Cargo.toml` |
| `rust-scalar` | Rust scalar kernel | same cdylib |
| `rust-simd` | force SIMD path | same cdylib |
| `triton` | **CUDA, pack maintained** | weights stay `uint8` in VRAM; fused dequant GEMM — working on this fork |

**Encoding** (same as FlashTernary):

```
+1 → 01,  0 → 00,  -1 → 10
4 weights / uint8 along K; C = A @ dequant(B_packed)
B_packed shape: (K//4, N)
```

WebGPU is intentionally out of scope.

## QAT / train path (started)

Pack-maintained **inference** is production (`PackCpuEmbed` / `PackCudaEmbed`).
**Training** stays on fp32 shadow weights + STE (same contract as library
BitLinear); pack is only an optional matmul representation of ternary *values*.

| piece | path |
|---|---|
| module | `flash_ternary.qat.QATBitLinear` |
| matmul modes | `dense` (ref) / `pack` (2-bit codec + matmul) |
| tests | `pytest tests/test_qat.py` |

```python
from flash_ternary import QATBitLinear, swap_to_qat_bitlinear, set_lambda_

# replace nn.Linear (projection excluded by default)
n = swap_to_qat_bitlinear(student, matmul="pack", lambda_=0.0)
set_lambda_(student, 1.0)  # after warmup
```

**Smoke (green 2026-08-29, CPU+CUDA, this fork):** micro student, 10 AdamW
steps — `QATBitLinear` dense/pack match library BitLinear (cos=1.0, loss rel
≪ 15%), against fertileweb/soycaporal's actual `StudentEncoder` (no RoPE, no
sub-LN — that architecture is a first-fork-only variant; see below).

```bash
python smoke_qat_train.py --steps 30 --device cpu   # or cuda
```

Distill flag (opt-in; default remains library BitLinear) — wired into
`training/distill/{config.py,ternary_qat.py,trainer.py}`:

```yaml
enable_qat: true
qat_backend: flash_qat   # bitlinear (default) | flash_qat
qat_matmul: pack         # dense | pack
```

`ternary_qat.swap()`/`set_lambda()`/`zero_fractions()` dispatch on this at
runtime; the default (`bitlinear`) training path has zero import-time
dependency on `tools/flash_ternary` — it's lazy-imported only when
`qat_backend: flash_qat` is set. See `configs/smoke-qat-flash.yaml` for a
runnable example, and `training/pack/pack.py` — packing doesn't care which
backend trained the shadow weights, since both re-derive the wire-format
ternary codes from the same fp32 shadow tensor at pack time.

**This fork has no RoPE / sub-LN.** `training/distill/model.py::StudentEncoder`
has no positional encoding by design (see its module docstring — validated in
the POC, not a gap) and no sub-LN. `pos_encoding` / `use_subln` kwargs, the
`engine/src/ternary_codec.rs` codec crate, and the `bitlinear_pack` engine
feature referenced elsewhere in ternlight docs are first-fork-only. This
tree's wire format is v1 (`training/pack/format.py`); `flash_ternary`'s
`bin_loader.py` / `cpu_embed.py` / `cuda_embed.py` all read v1 correctly
(`pos_encoding`/`subln_*` decode as absent, not present-but-zero) —
confirmed by `test_pack_embed_cos.py` passing end-to-end against a real
packed `.bin` from this fork's `StudentEncoder`.

## Quick start

```bash
cd tools/flash_ternary   # from the repo root

# unit tests (pack + QAT STE)
pytest tests/test_unit.py tests/test_qat.py -q

# pack / matmul correctness (numpy)
python3 -m flash_ternary.cli check --backend numpy

# pack embed cos gate vs UnpackedModel (needs .bin + tokenizer; default
# tokenizer is engine/assets — this fork's real WordPiece tokenizer.json)
# default cos≥0.999; optional --with-ckpt for pytorch BitLinear arm
PYTHONPATH=. python3 -m flash_ternary.cli cos-gate --with-ckpt
# or: pytest tests/test_pack_embed_cos.py -q

# optional torch
python3 -m flash_ternary.cli check --backend torch

# rust cdylib (codec + GEMM only; no engine/ dependency)
cargo build --release --manifest-path rust/Cargo.toml
python3 -m flash_ternary.cli check --backend rust

# microbench (single)
python3 -m flash_ternary.cli bench --backend rust-simd --M 32 --K 1024 --N 1024

# speed + accuracy table (numpy / rust-scalar / rust-simd / …)
python3 -m flash_ternary.bench

# CUDA pack-maintained (Triton) — correctness + vs dense fp16
python3 -m flash_ternary.bench_triton
```

### Triton CUDA (pack maintained)

```python
from flash_ternary.ops_cuda import TernaryLinearTriton
layer = TernaryLinearTriton.from_float(nn.Linear(1024, 1024)).cuda()
y = layer(x_half_cuda)  # weight_packed stays uint8 on GPU
```

### ternlight `.bin` → one BitLinear on Triton

```bash
# list sections (paths below are relative to the repo root; both
# CLIs also default to $TERNLIGHT_REPO/training/pack/out/en-minilm-int4.bin)
PYTHONPATH=. python3 -m flash_ternary.poc_ternlight_layer --list \
  --bin training/pack/out/en-minilm-int4.bin

# load L0.fc1 — full BitLinear (LN + act quant + Triton pack matmul)
PYTHONPATH=. python3 -m flash_ternary.poc_ternlight_layer \
  --bin training/pack/out/en-minilm-int4.bin \
  --layer 0 --name fc1 --batch 32
```

Wire note: ternlight packs `(out, in//4)`; Triton uses `(in//4, out)` — loader
transposes at load (same 2-bit codes).

Full forward matches `unpack.bitlinear_forward` / engine:

```
LN → AbsMax quant → Triton(x_quant, W_pack) + bias → / (w_scale * x_scale)
```

### Full CUDA pack embed (all BitLinears)

```bash
PYTHONPATH=. python3 -m flash_ternary.poc_cuda_embed \
  --bin training/pack/out/en-minilm-int4.bin \
  --tokenizer engine/assets
```

`PackCudaEmbed.from_bin` loads the whole graph; every BitLinear keeps
`uint8` weights on GPU (Triton). Embedding table is dequantized once at load.

### Fast tokenizer (no transformers)

`flash_ternary.spm_tok.FastSpmTokenizer` loads any HF `tokenizer.json` via
the Rust `tokenizers` package — despite the module name, it isn't
SentencePiece-specific; on this fork it loads `engine/assets/tokenizer.json`
(WordPiece, `bert-base-uncased`-derived, the same asset the Rust engine
ships). Pack embed paths use it by default — cold load ~5× faster than
`transformers.AutoTokenizer`, and encode stays off the transformers stack.
Parity with `transformers.AutoTokenizer` on this fork's tokenizer is a
regression test: `tests/test_unit.py::test_fast_spm_parity_with_hf`.


## Library (Python)

```python
from flash_ternary import pack_ternary, unpack_ternary, ternary_matmul

W = ...  # (K, N) values in {-1,0,+1}
Bp = pack_ternary(W)                 # (K//4, N) uint8
C  = ternary_matmul(A, Bp, backend="numpy")  # or torch / rust
```

## Relation to ternlight engine

**Same 2-bit codec, same codes** as `engine/src/kernels.rs` /
`engine/src/model.rs` (this fork inlines the codec there — no separate
codec crate). `training/pack/format.py` is the wire-format single source of
truth (v1); `training/pack/unpack.py::bitlinear_forward` is the reference
math both the Rust engine and this tool's Python backends must match.

| Engine feature (`engine/Cargo.toml`) | Weights in RAM | Kernel |
|---|---|---|
| `emb_fp32` / `emb_int8` / `emb_ternary` / `emb_int4` | pre-unpack → `i8` | `kernels::bitlinear_forward` |

There is no `bitlinear_pack` engine feature and no WASM A/B pack path on this
fork. The WASM ship (`packages/base`, `packages/mini`) always pre-unpacks.
Native CPU pack lives in `crates/tern-ternary` + `tools/flash_ternary/rust/`
(cdylib only — not linked into `engine/`). `scripts/build-wasm-ab.sh` is
not in this tree.

Wire layout is ternlight `(out, in/4)`; Flash GEMM wants `(in/4, out)` —
`wire_layout.py` (Python) and `tern_ternary::wire_oi_to_flash_io` (Rust).

Full matrix: [`ENGINE_UNIFY.md`](./ENGINE_UNIFY.md).

## SIMD (Rust)

Self-contained cdylib (codec + GEMM only, no `engine/` link):

```bash
cargo build --release --manifest-path rust/Cargo.toml
python3 -m flash_ternary.cli check --backend rust
```


Runtime-detected on x86_64:

1. **avx512vnni** (Zen 4 / Ice Lake+, query path `M < 64`)
   - **unpack**: 32 B/step, `_mm256_shuffle_epi8` 16-entry table (same codes as NEON `tbl`)
   - **matmul**: `dpbusd` over each 4-K group (i8 A × i8 W → i32, one convert at the end)
   - pack_cpu A is already BitLinear `x_quant` in `[-128, 127]`
2. **avx2+fma**
   - same pshufb unpack; **M < 64** 16-wide `fmadd` saxpy; **M ≥ 64** panel + rayon
3. **sse2** — 16 B/step unpack + 4-wide mul+add
4. **scalar** — portable tile-unpack reference

Unpack is tested against scalar (`unpack_four_rows_matches_scalar`).
Parallel path depends on `rayon`.

```bash
./rust/target/release/flash_ternary_cpu bench 32 1024 1024 20 --simd
./rust/target/release/flash_ternary_cpu bench 32 1024 1024 20 --scalar
```

## Benchmark notes (this host, AVX2+FMA)

Typical pattern (see `python -m flash_ternary.bench`):

- **Small** (e.g. 8×256×256): `rust-simd` ≫ `numpy` (BLAS launch overhead).
- **Medium** (32×1024×1024): `numpy` ≈ or slightly ahead of `rust-simd` (OpenBLAS).
- **Large** (8×4096×4096): `rust-simd` ~**3×** faster than `numpy`, ~**2.5×** vs `rust-scalar`.
- **Accuracy**: max |error| vs f64 dense is ~1e-4 absolute / ~1e-6 relative — f32 reorder noise only; all backends OK.
