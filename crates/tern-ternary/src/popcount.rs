//! Multiplier-free ternary×ternary GEMM via AND + popcount.
//!
//! Derived from `coalgebra-lab/math-like/ternary-spectral` Phase 8 (H8-2),
//! exhaustively verified there over all 65,536 packed-byte pairs
//! (`phase8_2bit_codec_bittricks_check.py`). This is a CPU-only proof of
//! concept for `ternlight/HYPOTHESES.md` H96 — the inner loop has no
//! multiply instruction at all, only AND + `count_ones` (POPCNT) +
//! accumulate.
//!
//! Requires **both** operands packed in ternlight's usual 2-bit codec (see
//! [`crate::codec`]): `+1->01, 0->00, -1->10, 0b11 reserved->0`. Production
//! BitLinear only has this for weights (activations are int8), so this
//! module has no wired-up caller yet — it exists to measure whether the
//! idea is worth wiring up at all (see `bench_popcount` example).

const POS_MASK: u8 = 0b0101_0101; // bit0 of every lane

/// One packed byte (4 trits) dotted against another, both codec-packed.
/// `neg` plane is shifted right by 1 before masking so lane `i`'s neg bit
/// lands at the same bit position (`2*i`) as lane `i`'s pos bit — without
/// this shift, `pos & neg` cross terms never share a set bit position and
/// silently evaluate to 0 for every lane (self-caught bug, see
/// `RESEARCH_LOG.md` Phase 8 / H8-2 write-up).
#[inline]
fn plane_dot_u8(a: u8, w: u8) -> i32 {
    let a_pos = a & POS_MASK;
    let a_neg = (a >> 1) & POS_MASK;
    let w_pos = w & POS_MASK;
    let w_neg = (w >> 1) & POS_MASK;
    let same = (a_pos & w_pos).count_ones() + (a_neg & w_neg).count_ones();
    let diff = (a_pos & w_neg).count_ones() + (a_neg & w_pos).count_ones();
    same as i32 - diff as i32
}

/// Pack an `(M,K)` f32 ternary array into row-major `(M, K/4)` codec bytes —
/// same threshold convention as [`crate::matmul::pack`], but packed
/// per-row along K (activations), not per-column along K (weights).
pub fn pack_activations(a: &[f32], m: usize, k: usize, out: &mut [u8]) -> bool {
    if k % 4 != 0 || a.len() < m * k || out.len() < m * (k / 4) {
        return false;
    }
    let kp = k / 4;
    for row in 0..m {
        for p in 0..kp {
            let mut byte: u8 = 0;
            for s in 0..4 {
                let v = a[row * k + p * 4 + s];
                let enc: u8 = if v > 0.5 { 1 } else if v < -0.5 { 2 } else { 0 };
                byte |= enc << (2 * s);
            }
            out[row * kp + p] = byte;
        }
    }
    true
}

/// `C[M,N] = A_packed[M,K/4] . W_packed[K/4,N]`, both operands ternary
/// codec-packed, via AND + popcount only (no multiply instruction).
/// `w_packed` layout matches [`crate::matmul::pack`]'s output: `(K/4, N)`.
///
/// **Portable baseline** — compiled for the crate's default target, which
/// does *not* enable the `popcnt` CPU feature. `count_ones()` therefore
/// lowers to a ~10-instruction software bit-counting sequence, not the
/// single-cycle `POPCNT` instruction, even on hardware that has it. Use
/// [`matmul_ternary_ternary_popcount_fast`] for a fair benchmark — this
/// function exists to show *why* the naive version was slow (see
/// `bench_popcount`'s "portable" column).
pub fn matmul_ternary_ternary_popcount(
    a_packed: &[u8], w_packed: &[u8], c: &mut [f32], m: usize, n: usize, k: usize,
) -> bool {
    if k % 4 != 0 {
        return false;
    }
    let kp = k / 4;
    if a_packed.len() < m * kp || w_packed.len() < kp * n || c.len() < m * n {
        return false;
    }
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0i32;
            for p in 0..kp {
                acc += plane_dot_u8(a_packed[mi * kp + p], w_packed[p * n + ni]);
            }
            c[mi * n + ni] = acc as f32;
        }
    }
    true
}

#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "popcnt"))]
unsafe fn plane_dot_u8_popcnt(a: u8, w: u8) -> i32 {
    plane_dot_u8(a, w)
}

/// Same computation as [`matmul_ternary_ternary_popcount`], but with the
/// hot loop compiled under `#[target_feature(enable = "popcnt")]` so
/// `count_ones()` lowers to the real `POPCNT` instruction — matching how
/// every other specialized kernel in [`crate::matmul`] is written (runtime
/// `is_x86_feature_detected!` gate + `#[target_feature]` fn), instead of
/// relying on the crate's default (baseline, no `popcnt`) compile target.
/// Falls back to the portable version if the running CPU lacks `popcnt`
/// (true on some older x86_64 and on non-x86_64 targets).
// Same threshold `crate::matmul` uses before switching a kernel to the
// rayon panel path (`PAR_M_MIN`, private there) — mirrored here so the
// "fast" kernel only pays thread-pool dispatch cost where the SIMD kernel
// it's compared against does too (small-M rows in `bench_popcount` stay
// single-threaded on both sides).
const PAR_M_MIN: usize = 64;

pub fn matmul_ternary_ternary_popcount_fast(
    a_packed: &[u8], w_packed: &[u8], c: &mut [f32], m: usize, n: usize, k: usize,
) -> bool {
    if k % 4 != 0 {
        return false;
    }
    let kp = k / 4;
    if a_packed.len() < m * kp || w_packed.len() < kp * n || c.len() < m * n {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("popcnt") {
        #[cfg(feature = "parallel")]
        if m >= PAR_M_MIN {
            // Grok R26 addendum (a2a-board #18, 2026-08-29): the parallel
            // kernel below walks `c` in n-wide chunks with no `m` bound —
            // if a caller ever passes `c.len() > m*n` (the public contract
            // only requires `>=`), it would process phantom rows past `m`
            // and index `a_packed` out of bounds. Slicing to exactly `m*n`
            // here closes that off regardless of what the caller over-sized
            // `c` to. Existing tests/bench always pass `c.len() == m*n`, so
            // this never fired before.
            unsafe { matmul_popcnt_native_par(a_packed, w_packed, &mut c[..m * n], n, kp) };
            return true;
        }
        unsafe { matmul_popcnt_native(a_packed, w_packed, c, m, n, kp) };
        return true;
    }
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0i32;
            for p in 0..kp {
                acc += plane_dot_u8(a_packed[mi * kp + p], w_packed[p * n + ni]);
            }
            c[mi * n + ni] = acc as f32;
        }
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn matmul_popcnt_native(
    a_packed: &[u8], w_packed: &[u8], c: &mut [f32], m: usize, n: usize, kp: usize,
) {
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0i32;
            for p in 0..kp {
                acc += plane_dot_u8_popcnt(a_packed[mi * kp + p], w_packed[p * n + ni]);
            }
            c[mi * n + ni] = acc as f32;
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "parallel"))]
#[target_feature(enable = "popcnt")]
unsafe fn matmul_popcnt_native_par(
    a_packed: &[u8], w_packed: &[u8], c: &mut [f32], n: usize, kp: usize,
) {
    use rayon::prelude::*;
    c.par_chunks_mut(n).enumerate().for_each(|(mi, c_row)| {
        for (ni, out) in c_row.iter_mut().enumerate() {
            let mut acc = 0i32;
            for p in 0..kp {
                acc += unsafe { plane_dot_u8_popcnt(a_packed[mi * kp + p], w_packed[p * n + ni]) };
            }
            *out = acc as f32;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_naive_reference() {
        let m = 5;
        let n = 7;
        let k = 16;
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 1) % 3) as f32 - 1.0).collect();
        let w: Vec<f32> = (0..k * n).map(|i| ((i * 11 + 2) % 3) as f32 - 1.0).collect();

        let mut a_packed = vec![0u8; m * (k / 4)];
        assert!(pack_activations(&a, m, k, &mut a_packed));
        let mut w_packed = vec![0u8; (k / 4) * n];
        assert!(crate::matmul::pack(&w, k, n, &mut w_packed));

        let mut c = vec![0f32; m * n];
        assert!(matmul_ternary_ternary_popcount(&a_packed, &w_packed, &mut c, m, n, k));

        for mi in 0..m {
            for ni in 0..n {
                let mut expected = 0f32;
                for ki in 0..k {
                    expected += a[mi * k + ki] * w[ki * n + ni];
                }
                assert_eq!(c[mi * n + ni], expected, "mismatch at ({mi},{ni})");
            }
        }
    }

    #[test]
    fn fast_matches_portable() {
        let m = 66; // exercises the parallel (>=PAR_M_MIN-scale) path too
        let n = 5;
        let k = 24;
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 5 + 2) % 3) as f32 - 1.0).collect();
        let w: Vec<f32> = (0..k * n).map(|i| ((i * 13 + 1) % 3) as f32 - 1.0).collect();

        let mut a_packed = vec![0u8; m * (k / 4)];
        assert!(pack_activations(&a, m, k, &mut a_packed));
        let mut w_packed = vec![0u8; (k / 4) * n];
        assert!(crate::matmul::pack(&w, k, n, &mut w_packed));

        let mut c_portable = vec![0f32; m * n];
        assert!(matmul_ternary_ternary_popcount(&a_packed, &w_packed, &mut c_portable, m, n, k));
        let mut c_fast = vec![0f32; m * n];
        assert!(matmul_ternary_ternary_popcount_fast(&a_packed, &w_packed, &mut c_fast, m, n, k));

        assert_eq!(c_portable, c_fast);
    }

    #[test]
    fn fast_par_oversized_output_buffer_no_oob() {
        // Regression for the a2a-board #18 / R26-addendum finding:
        // matmul_popcnt_native_par used to walk `c` in n-wide chunks with no
        // `m` bound, so a caller-oversized `c` (public contract only
        // requires `c.len() >= m*n`) would index a_packed out of bounds on
        // the phantom rows past `m`. This exercises exactly that shape —
        // must not panic, and the extra tail of `c` must stay untouched.
        let m = 70; // >= PAR_M_MIN so this hits the parallel path
        let n = 3;
        let k = 8;
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 3 + 1) % 3) as f32 - 1.0).collect();
        let w: Vec<f32> = (0..k * n).map(|i| ((i * 5 + 2) % 3) as f32 - 1.0).collect();

        let mut a_packed = vec![0u8; m * (k / 4)];
        assert!(pack_activations(&a, m, k, &mut a_packed));
        let mut w_packed = vec![0u8; (k / 4) * n];
        assert!(crate::matmul::pack(&w, k, n, &mut w_packed));

        let extra = 5;
        let sentinel = -999.0f32;
        let mut c = vec![sentinel; m * n + extra]; // oversized on purpose
        assert!(matmul_ternary_ternary_popcount_fast(&a_packed, &w_packed, &mut c, m, n, k));

        let mut c_ref = vec![0f32; m * n];
        assert!(matmul_ternary_ternary_popcount(&a_packed, &w_packed, &mut c_ref, m, n, k));
        assert_eq!(&c[..m * n], &c_ref[..]);
        assert!(c[m * n..].iter().all(|&v| v == sentinel), "wrote past m*n");
    }

    #[test]
    fn rejects_bad_dims() {
        let mut c = vec![0f32; 4];
        assert!(!matmul_ternary_ternary_popcount(&[0u8; 1], &[0u8; 1], &mut c, 2, 2, 3));
    }
}
