//! Shared ternary building blocks for ternlight.
//!
//! | Module | Used by |
//! |---|---|
//! | [`codec`] | `engine` (wire pack), `tools/flash_ternary` (Python + rust) |
//! | [`matmul`] | FlashTernary CPU GEMM (optional `parallel` / `c_abi`) |
//!
//! Encoding (single truth): **+1→01, 0→00, −1→10**, 4 weights/byte along K/`in`.
//! Wire layout: `(out, in/4)`; Flash GEMM layout: `(in/4, out)`.

#![allow(clippy::needless_range_loop)]

#[cfg(feature = "matmul")]
pub mod cache_detect;
pub mod codec;
#[cfg(feature = "matmul")]
pub mod matmul;
#[cfg(feature = "matmul")]
pub mod popcount;

pub use codec::{
    decode_code_f32, decode_code_i8, flash_io_to_wire_oi, pack_wire_oi, unpack_wire_oi,
    wire_oi_to_flash_io,
};
#[cfg(feature = "matmul")]
pub use matmul::{
    active_simd_level, matmul_f32, matmul_f32_ex, pack, MatmulKernel,
};
#[cfg(feature = "matmul")]
pub use popcount::{
    matmul_ternary_ternary_popcount, matmul_ternary_ternary_popcount_fast, pack_activations,
};
