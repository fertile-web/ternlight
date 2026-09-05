//! FlashTernary CPU cdylib — thin re-export of workspace crate `tern-ternary`.
//!
//! Builds a `cdylib` with C ABI (`flash_ternary_*`) for Python `ctypes`.
//! Math lives in `crates/tern-ternary`.

pub use tern_ternary::*;
// C ABI symbols are compiled into this crate via tern-ternary `c_abi` feature.
