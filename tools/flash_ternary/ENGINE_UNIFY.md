# Engine ↔ FlashTernary unification

Two BitLinear runtimes share one **2-bit ternary codec** and wire layout.

| Path | Crate / module | Weight in RAM | Hot matmul |
|---|---|---|---|
| **Engine default** (WASM ship) | `engine` Phase A3 | pre-unpack → `i8` | `i8×i8→i32` (+ wasm SIMD) |
| **Engine pack** | `engine` + `bitlinear_pack` | wire `u8` `(out, in/4)` | decode-once (16-entry tbl) + SIMD inner |
| **FlashTernary CPU** | `tools/flash_ternary` | wire / Flash layout | rust: AVX-512 VNNI `dpbusd` (M&lt;64) / AVX2 FMA / NEON `tbl` |
| **RAG pack_cpu** | `PackCpuEmbed` | same `.bin` | rust pack (default) or BLAS unpack |
| **QAT train (v1)** | `flash_ternary.qat.QATBitLinear` | shadow fp32 + STE ternary | dense or pack matmul (codes not Parameters) |

## Shared encoding (single source of truth)

Codes (2 bits / weight, 4 / byte along `in`):

| value | code |
|---|---|
| +1 | `01` |
| 0 | `00` |
| −1 | `10` |

Keep in sync:

- `engine/src/ternary_codec.rs`
- `tools/flash_ternary/flash_ternary/quant.py`
- `tools/flash_ternary/rust/src/lib.rs`
- `training/pack/pack.py`

**Wire** (ternlight `.bin`): packed shape `(out_features, in_features/4)`.  
**Flash GEMM** layout: `(in/4, out)` — transpose with `ternary_codec::wire_oi_to_flash_io`.

## Build

```bash
# default: unpack i8 (ship / WASM)
cargo test -p tern-engine --features emb_int4

# pack-maintained weights (4× less BitLinear RAM)
cargo test -p tern-engine --features emb_int4,bitlinear_pack

# FlashTernary CPU cdylib (native AVX)
cargo build --release --manifest-path tools/flash_ternary/rust/Cargo.toml
```

Kernel parity (no `.bin` needed):

```bash
cargo test -p tern-engine --features emb_int4 packed_matches_unpacked_bitlinear
cargo test -p tern-engine --features emb_int4 pack_unpack_roundtrip
```

## When to use which

| Scenario | Prefer |
|---|---|
| Small WASM embed model, W fits L2 | engine **unpack** (default) |
| Large W / tight RAM / mobile | engine **`bitlinear_pack`** |
| Native RAG query (prod; may co-run FlashRank) | `PackCpuEmbed` **rust** (default; no OMP fight with ONNX) |
| Hybrid-only microbench (tiny M, no CE) | `RAG_FLASH_BACKEND=blas` (f32 GEMM after one unpack) |

## Numerics gate

Pack embed cos gate (Python) already checks `PackCpuEmbed` / `PackCudaEmbed` vs
`UnpackedModel` at cos≥0.999. Engine pack vs unpack is covered by the Rust
unit test above (exact on synthetic ternary W).

## Shared crate: `crates/tern-ternary`

| feature | meaning |
|---|---|
| (none) | **codec only** — used by `engine` (WASM-safe) |
| `matmul` | Flash fused GEMM |
| `parallel` | rayon panels (implies `matmul`) |
| `c_abi` | `flash_ternary_*` ctypes exports |

```
engine  ──default-features=false──►  tern-ternary (codec)
flash_ternary_cpu ──matmul+parallel+c_abi──► tern-ternary
```

## WASM real machine (HTML + Node)

Yes — **browser is the real machine** (`--target web`). Node can load the same artifacts headlessly.

```bash
# build both variants (wiki-d256-3l-int4 by default)
bash scripts/build-wasm-ab.sh

# headless A/B
node scripts/wasm-ab-bench.mjs

# browser UI
python3 -m http.server 8765 --directory dist/web-ab
# open http://127.0.0.1:8765/
```

### Measured (2026-08-17)

`bitlinear_forward_packed` now decodes each weight row **once per forward**
(16-entry in-register table: wasm `i8x16_swizzle`, aarch64 `vqtbl1q_s8`)
and reuses the SIMD inner. Default feature stays **off** — WASM ship / upstream
still prefer unpack latency; enable `bitlinear_pack` for the 4× resident RAM cut.

en-minilm d512/L3/ffn2048 (BitLinear RAM 9.44 MB → 2.36 MB; cos 1.0 / native bit-identical):

| build | native aarch64 | wasm 8 tok | wasm 128 tok |
|---|---|---|---|
| **unpack** (default) | 9.75 ms/doc | 8.87 ms | 143.6 ms |
| **pack** (`bitlinear_pack`) | **8.65** (1.13×) | 10.25 (0.86×) | 149.5 (0.96×) |

Older wiki-d256-3l wasm row (~19 ms pack vs ~3 ms unpack) is superseded.
WASM **file** size is still nearly identical: `model.bin` is already 2-bit
on the wire; the 4× saving is **resident weight buffers after init**.

`pack_cpu` / `tern-ternary` on aarch64: NEON unpack is now `vshrq_n_u8` +
`vqtbl1q_s8` (4.75× on the primitive). End-to-end it shows at small M
(M=1: 0.32→0.16 ms) and is noise once unpack is amortized (M≥32).

x86 (2026-08-17, Ryzen 5 7600): Auto uses **AVX-512 VNNI `dpbusd`** for
`M < 64` (query). Same 4-K group as the unpack, i8 A (BitLinear `x_quant`)
× i8 W → i32, one convert at the end. AVX2+FMA saxpy remains the fallback
and the M≥64 rayon path. pshufb unpack matches the NEON table.

| shape | scalar | avx2+fma (before) | avx512vnni | vs previous simd |
|---|---:|---:|---:|---:|
| attn M=1 | 0.088 ms | 0.019 ms | **0.012 ms** | 1.63× |
| fc1 M=1 | 0.30 ms | 0.085 ms | **0.045 ms** | 1.60× |
| fc2 M=1 | 0.28 ms | 0.069 ms | **0.039 ms** | 1.73× |

`rust check` max_abs_err = 0 on integer A. QAT `flash_qat` pack mode is
unchanged (still pack→unpack→dense).