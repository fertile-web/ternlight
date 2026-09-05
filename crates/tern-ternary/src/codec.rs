//! Shared 2-bit ternary codec (FlashTernary / ternlight wire format).
//!
//! **KEEP IN SYNC** with:
//! - `tools/flash_ternary/flash_ternary/quant.py` (Python)
//! - `tools/flash_ternary/rust/src/lib.rs` (`decode_nibble_*`, `pack`)
//! - `training/pack/pack.py` BitLinear encoder
//!
//! Encoding (2 bits per weight, 4 weights per `u8` along `in_features`):
//!   +1 → `01` (1)
//!    0 → `00` (0)
//!   -1 → `10` (2)
//!   `11` reserved → treat as 0
//!
//! Wire layout for one BitLinear matrix: **row-major `(out_features, in_features/4)`**
//! (ternlight `.bin`). FlashTernary GEMM wants `(in/4, out)` — transpose at the
//! boundary when feeding the fused kernel (`tools/flash_ternary`).

/// Decode one 2-bit code to ternary `i8`.
#[inline]
pub const fn decode_code_i8(code: u8) -> i8 {
    match code & 0b11 {
        0b01 => 1,
        0b10 => -1,
        _ => 0, // 00 and reserved 11
    }
}

/// Decode one 2-bit code to ternary `f32`.
#[inline]
pub const fn decode_code_f32(code: u8) -> f32 {
    decode_code_i8(code) as f32
}

/// Unpack ternlight wire pack → row-major `i8` of shape `[out * in]`.
///
/// `packed` length must be `out_features * (in_features / 4)`.
/// Codes walk **along `in_features`** first (4 per byte), then next out-row.
pub fn unpack_wire_oi(packed: &[u8], in_features: usize, out_features: usize) -> Vec<i8> {
    debug_assert_eq!(in_features % 4, 0);
    debug_assert_eq!(packed.len(), out_features * (in_features / 4));
    let n = in_features * out_features;
    let mut out = Vec::with_capacity(n);
    for &byte in packed {
        for sub in 0..4 {
            let code = (byte >> (sub * 2)) & 0b11;
            out.push(decode_code_i8(code));
        }
    }
    debug_assert_eq!(out.len(), n);
    out
}

/// Pack row-major `i8` `[out * in]` → wire `(out, in/4)` uint8.
pub fn pack_wire_oi(w: &[i8], in_features: usize, out_features: usize) -> Vec<u8> {
    debug_assert_eq!(in_features % 4, 0);
    debug_assert_eq!(w.len(), out_features * in_features);
    let kp = in_features / 4;
    let mut packed = vec![0u8; out_features * kp];
    for o in 0..out_features {
        for p in 0..kp {
            let mut byte = 0u8;
            for s in 0..4 {
                let v = w[o * in_features + p * 4 + s];
                let enc = if v > 0 {
                    1u8
                } else if v < 0 {
                    2u8
                } else {
                    0u8
                };
                byte |= enc << (2 * s);
            }
            packed[o * kp + p] = byte;
        }
    }
    packed
}

/// Transpose wire `(out, in/4)` → FlashTernary `(in/4, out)` for fused GEMM.
pub fn wire_oi_to_flash_io(packed_oi: &[u8], in_features: usize, out_features: usize) -> Vec<u8> {
    let kp = in_features / 4;
    debug_assert_eq!(packed_oi.len(), out_features * kp);
    let mut io = vec![0u8; kp * out_features];
    for o in 0..out_features {
        for p in 0..kp {
            io[p * out_features + o] = packed_oi[o * kp + p];
        }
    }
    io
}

/// Transpose Flash `(in/4, out)` → wire `(out, in/4)`.
pub fn flash_io_to_wire_oi(packed_io: &[u8], in_features: usize, out_features: usize) -> Vec<u8> {
    let kp = in_features / 4;
    debug_assert_eq!(packed_io.len(), kp * out_features);
    let mut oi = vec![0u8; out_features * kp];
    for o in 0..out_features {
        for p in 0..kp {
            oi[o * kp + p] = packed_io[p * out_features + o];
        }
    }
    oi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let in_f = 8usize;
        let out_f = 3usize;
        let w: Vec<i8> = vec![
            1, 0, -1, 1, 0, 0, -1, 1, // row0
            0, 1, 1, -1, 0, -1, 0, 1, // row1
            -1, -1, 0, 0, 1, 1, 0, -1, // row2
        ];
        let p = pack_wire_oi(&w, in_f, out_f);
        assert_eq!(p.len(), out_f * (in_f / 4));
        let u = unpack_wire_oi(&p, in_f, out_f);
        assert_eq!(u, w);
    }

    #[test]
    fn flash_transpose_preserves_values() {
        let in_f = 8usize;
        let out_f = 2usize;
        let w: Vec<i8> = (0..out_f * in_f)
            .map(|i| match i % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            })
            .collect();
        let oi = pack_wire_oi(&w, in_f, out_f);
        let io = wire_oi_to_flash_io(&oi, in_f, out_f);
        // dequant flash layout column o should match wire row o
        let kp = in_f / 4;
        for o in 0..out_f {
            let mut row = Vec::new();
            for p in 0..kp {
                let byte = io[p * out_f + o];
                for s in 0..4 {
                    row.push(decode_code_i8((byte >> (2 * s)) & 3));
                }
            }
            assert_eq!(&row[..], &w[o * in_f..(o + 1) * in_f]);
        }
    }
}
