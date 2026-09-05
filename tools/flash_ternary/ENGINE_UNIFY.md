# Engine ↔ FlashTernary unification (fertileweb/soycaporal fork)

> This file describes **this** tree (`fertile-web/ternlight`, tracking
> `soycaporal/ternlight`). Native pack GEMM is `crates/tern-ternary` +
> `tools/flash_ternary/rust/` (not linked into WASM `engine/`). There is no
> `bitlinear_pack` engine feature, no RoPE/sub-LN student, and no v2–v4 wire.

Two BitLinear runtimes share one **2-bit ternary codec** and wire layout:

| Path | Module | Weight in RAM | Hot matmul |
|---|---|---|---|
| **Engine** (WASM ship) | `engine/src/{kernels,model}.rs` | pre-unpack → `i8` | `kernels::bitlinear_forward` (+ wasm SIMD) |
| **FlashTernary CPU** | `tools/flash_ternary` | wire / Flash layout | `numpy` / `torch` / `triton`(CUDA) / native Rust cdylib |
| **RAG pack_cpu** | `flash_ternary.cpu_embed.PackCpuEmbed` | same `.bin` | numpy/torch backend (`blas` is the CPU default when no rust cdylib) |
| **QAT train** | `training/distill/ternary_qat.py` | shadow fp32 + STE | `bitlinear` lib (default) or `flash_ternary.qat.QATBitLinear` (`qat_backend: flash_qat`) |

## Shared encoding (single source of truth)

Codes (2 bits / weight, 4 / byte along `in`):

| value | code |
|---|---|
| +1 | `01` |
| 0 | `00` |
| −1 | `10` |

Keep in sync:

- `engine/src/kernels.rs` / `engine/src/model.rs` (WASM engine still inlines decode)
- `crates/tern-ternary/src/codec.rs`
- `tools/flash_ternary/flash_ternary/quant.py`
- `training/pack/pack.py` / `training/pack/unpack.py` / `training/pack/format.py`

**Wire** (ternlight `.bin`, v1 — `training/pack/format.py`): packed shape
`(out_features, in_features/4)`. No `pos_encoding` / `use_subln` header
fields — `bin_loader.py`'s parser (written for the sibling fork's v2/v3
header) reads the v1 header's zeroed reserved bytes as `pos=0, subln=0`,
which happens to be correct here since this fork's `StudentEncoder` really
has neither.

**Flash GEMM** layout: `(in/4, out)` — `flash_ternary/wire_layout.py` (Python)
and `tern_ternary::wire_oi_to_flash_io` (Rust cdylib).

## Build

```bash
# engine feature gates (engine/Cargo.toml) — exactly one, mutually exclusive
wasm-pack build --target web --features emb_int4     # ship default
wasm-pack build --target web --features emb_ternary   # alt

# flash_ternary numpy/torch/triton need no compile
cd tools/flash_ternary && pytest tests/ -q

# native CPU pack GEMM (not linked into WASM engine)
cargo build --release --manifest-path tools/flash_ternary/rust/Cargo.toml
```

There is no `bitlinear_pack` engine feature and no WASM A/B pack path.

## When to use which

| Scenario | Prefer |
|---|---|
| Ship to browser (WASM) | `engine`, `emb_int4` feature (default) |
| Server-side / bulk CPU embed via `tools/flash_ternary` | `numpy` or `torch` backend (`PackCpuEmbed`) |
| Server-side / bulk GPU embed | `triton` backend (`PackCudaEmbed`) — needs CUDA |
| QAT training, default | `training/distill` + `bitlinear` lib (`qat_backend: bitlinear`, default) |
| QAT training, exercise the wire codec on the train path | `qat_backend: flash_qat`, `qat_matmul: pack` (see `configs/smoke-qat-flash.yaml`) |

## Numerics gate

`tests/test_pack_embed_cos.py` checks `PackCpuEmbed` (numpy/BLAS) and
`PackCudaEmbed` (Triton) vs `training/pack/unpack.py::UnpackedModel` at
cos≥0.999, against a real packed `.bin` produced by `training/pack/pack.py`
from a `training/distill` checkpoint (QAT, either backend) — this is the
actual engine-equivalence proof on this fork, not a synthetic-tensor test.
Needs `TERNLIGHT_EN_BIN` / `TERNLIGHT_TOKENIZER` (defaults:
`training/pack/out/en-minilm-int4.bin`, `engine/assets`) and optionally
`TERNLIGHT_CKPT` for the extra pytorch-BitLinear arm.

## Native CPU (Rust)

Self-contained cdylib (2-bit codec + GEMM only). Not linked into `engine/`.

```bash
cargo build --release --manifest-path tools/flash_ternary/rust/Cargo.toml
python3 -m flash_ternary.cli check --backend rust
```
