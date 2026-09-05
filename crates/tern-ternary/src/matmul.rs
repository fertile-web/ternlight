//! FlashTernary-compatible 2-bit pack + fused ternary matmul (CPU).
//!
//! Encoding: +1 → 01, 0 → 00, -1 → 10; 4 weights per `u8` along K.
//! Matmul: `C[M,N] = A[M,K] @ dequant(B_packed[K/4,N])`.
//!
//! Kernels:
//! - **scalar** — portable reference (general f32 A)
//! - **avx512vnni** — pshufb unpack + `dpbusd` over each 4-K group (i8 A)
//! - **avx2+fma** — pshufb unpack + f32 saxpy (fallback when VNNI is absent)
//! - **neon** (aarch64) — tbl unpack + f32 FMA panel
//!
//! Panel (GEBP-style): for each K-packed row, unpack W once, then process
//! M in panels of MR so each W vector loads once per N-block.

/// Pack (K,N) f32 ternary into (K/4,N) u8. Returns false on bad dims.
pub fn pack(w: &[f32], k: usize, n: usize, out: &mut [u8]) -> bool {
    if k % 4 != 0 || w.len() < k * n || out.len() < (k / 4) * n {
        return false;
    }
    let kp = k / 4;
    for row_p in 0..kp {
        for col in 0..n {
            let mut byte: u8 = 0;
            for s in 0..4 {
                let v = w[(row_p * 4 + s) * n + col];
                let enc: u8 = if v > 0.5 {
                    1
                } else if v < -0.5 {
                    2
                } else {
                    0
                };
                byte |= enc << (2 * s);
            }
            out[row_p * n + col] = byte;
        }
    }
    true
}

#[inline]
fn decode_nibble_f32(enc: u8) -> f32 {
    (enc == 1) as i8 as f32 - (enc == 2) as i8 as f32
}

#[inline]
fn decode_nibble_i8(enc: u8) -> i8 {
    (enc == 1) as i8 - (enc == 2) as i8
}

/// BitLinear `x_quant` → i8. Production A is already `rint`+clamp; `.round()`
/// keeps the SIMD path honest if a caller left a 0.5 ulp.
fn quantize_a_to_i8(a: &[f32], out: &mut [i8]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i].round().clamp(-128.0, 127.0) as i8;
    }
}

fn store_i32_as_f32(acc: &[i32], c: &mut [f32]) {
    debug_assert_eq!(acc.len(), c.len());
    let n = acc.len();
    let mut i = 0usize;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 detected; we only touch `i..i+8` inside the loop.
            unsafe {
                use std::arch::x86_64::*;
                while i + 8 <= n {
                    let v = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
                    _mm256_storeu_ps(c.as_mut_ptr().add(i), _mm256_cvtepi32_ps(v));
                    i += 8;
                }
            }
        }
    }
    while i < n {
        c[i] = acc[i] as f32;
        i += 1;
    }
}

/// Kernel selection for matmul.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum MatmulKernel {
    /// Runtime pick (AVX2+FMA / NEON if available, else scalar).
    Auto = 0,
    Scalar = 1,
    /// Force SIMD path; falls back to scalar if unavailable.
    Simd = 2,
}

/// Which kernel would `Auto`/`Simd` use on this CPU.
pub fn active_simd_level() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl") {
            return "avx512vnni";
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return "avx2+fma";
        }
        if is_x86_feature_detected!("sse2") {
            return "sse2";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is baseline on aarch64 (AAPCS64) — no runtime detection needed.
        return "neon";
    }
    #[cfg(not(target_arch = "aarch64"))]
    "none"
}

pub fn matmul_f32(a: &[f32], b_packed: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) -> bool {
    matmul_f32_ex(a, b_packed, c, m, n, k, MatmulKernel::Auto)
}

pub fn matmul_f32_ex(
    a: &[f32],
    b_packed: &[u8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    kernel: MatmulKernel,
) -> bool {
    if k % 4 != 0 {
        return false;
    }
    let kp = k / 4;
    if a.len() < m * k || b_packed.len() < kp * n || c.len() < m * n {
        return false;
    }

    match kernel {
        MatmulKernel::Scalar => {
            matmul_scalar(a, b_packed, c, m, n, k);
            true
        }
        MatmulKernel::Auto | MatmulKernel::Simd => {
            #[cfg(target_arch = "x86_64")]
            {
                // VNNI wins on the query path (M < PAR_M_MIN). Large-M keeps
                // the AVX2+FMA + rayon panel — VNNI is sequential today.
                if m < PAR_M_MIN
                    && is_x86_feature_detected!("avx512vnni")
                    && is_x86_feature_detected!("avx512vl")
                {
                    unsafe {
                        matmul_avx512_vnni(a, b_packed, c, m, n, k);
                    }
                    return true;
                }
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // SAFETY: feature detected.
                    unsafe {
                        matmul_avx2_fma(a, b_packed, c, m, n, k);
                    }
                    return true;
                }
                if is_x86_feature_detected!("sse2") {
                    unsafe {
                        matmul_sse2(a, b_packed, c, m, n, k);
                    }
                    return true;
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                unsafe {
                    matmul_neon(a, b_packed, c, m, n, k);
                }
                return true;
            }
            #[allow(unreachable_code)]
            if matches!(kernel, MatmulKernel::Simd) {
                // requested SIMD but none available
                matmul_scalar(a, b_packed, c, m, n, k);
                return true;
            }
            matmul_scalar(a, b_packed, c, m, n, k);
            true
        }
    }
}

// ── Scalar (reference) ──────────────────────────────────────────────────────

fn matmul_scalar(a: &[f32], b_packed: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    c.fill(0.0);
    const TILE_K: usize = 64;
    let mut w_tile = vec![0f32; TILE_K * n];

    let mut k0 = 0;
    while k0 < k {
        let k1 = (k0 + TILE_K).min(k);
        let tk = k1 - k0;
        for kk in k0..k1 {
            let local = kk - k0;
            let byte_row = kk / 4;
            let shift = (kk % 4) * 2;
            let prow = &b_packed[byte_row * n..(byte_row + 1) * n];
            let dest = &mut w_tile[local * n..(local + 1) * n];
            for j in 0..n {
                dest[j] = decode_nibble_f32((prow[j] >> shift) & 0x03);
            }
        }
        for i in 0..m {
            let a_row = &a[i * k + k0..i * k + k1];
            let c_row = &mut c[i * n..(i + 1) * n];
            for t in 0..tk {
                let av = a_row[t];
                if av == 0.0 {
                    continue;
                }
                let wrow = &w_tile[t * n..(t + 1) * n];
                for j in 0..n {
                    c_row[j] += av * wrow[j];
                }
            }
        }
        k0 = k1;
    }
}

// ── x86_64 SSE2 ─────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn matmul_sse2(a: &[f32], b_packed: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    c.fill(0.0);
    let mut w_i8 = vec![0i8; 4 * n];
    let kp = k / 4;
    for p in 0..kp {
        let prow = &b_packed[p * n..(p + 1) * n];
        unpack_four_rows_sse2(prow, &mut w_i8);
        for i in 0..m {
            let c_row = &mut c[i * n..(i + 1) * n];
            for s in 0..4 {
                let av = a[i * k + p * 4 + s];
                if av == 0.0 {
                    continue;
                }
                saxpy_i8_sse2(c_row, av, &w_i8[s * n..(s + 1) * n]);
            }
        }
    }
}

/// Decode 8+8 i16 encodings → 16 i8 ternary values (SSE2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn decode_enc_i16_sse2(lo: std::arch::x86_64::__m128i, hi: std::arch::x86_64::__m128i) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    let one = _mm_set1_epi16(1);
    let two = _mm_set1_epi16(2);
    let r_lo = _mm_sub_epi16(_mm_cmpeq_epi16(lo, two), _mm_cmpeq_epi16(lo, one));
    let r_hi = _mm_sub_epi16(_mm_cmpeq_epi16(hi, two), _mm_cmpeq_epi16(hi, one));
    _mm_packs_epi16(r_lo, r_hi)
}

/// Unpack one packed row into 4 i8 weight rows (shifts 0,2,4,6) — SSE2, 16 B/step.
///
/// `srli_epi16` immediates must be const, so the four shifts are unrolled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn unpack_four_rows_sse2(prow: &[u8], dest4: &mut [i8]) {
    use std::arch::x86_64::*;
    let n = prow.len();
    debug_assert_eq!(dest4.len(), 4 * n);
    let mask03 = _mm_set1_epi16(0x03);
    let mut j = 0usize;
    while j + 16 <= n {
        let bytes = _mm_loadu_si128(prow.as_ptr().add(j) as *const __m128i);
        // zero-extend 8+8 u8 → i16 so byte-wise shift is lane-safe
        let lo8 = _mm_unpacklo_epi8(bytes, _mm_setzero_si128());
        let hi8 = _mm_unpackhi_epi8(bytes, _mm_setzero_si128());
        // shift 0
        {
            let lo = _mm_and_si128(lo8, mask03);
            let hi = _mm_and_si128(hi8, mask03);
            let packed = decode_enc_i16_sse2(lo, hi);
            _mm_storeu_si128(dest4.as_mut_ptr().add(0 * n + j) as *mut __m128i, packed);
        }
        // shift 2
        {
            let lo = _mm_and_si128(_mm_srli_epi16::<2>(lo8), mask03);
            let hi = _mm_and_si128(_mm_srli_epi16::<2>(hi8), mask03);
            let packed = decode_enc_i16_sse2(lo, hi);
            _mm_storeu_si128(dest4.as_mut_ptr().add(1 * n + j) as *mut __m128i, packed);
        }
        // shift 4
        {
            let lo = _mm_and_si128(_mm_srli_epi16::<4>(lo8), mask03);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(hi8), mask03);
            let packed = decode_enc_i16_sse2(lo, hi);
            _mm_storeu_si128(dest4.as_mut_ptr().add(2 * n + j) as *mut __m128i, packed);
        }
        // shift 6
        {
            let lo = _mm_and_si128(_mm_srli_epi16::<6>(lo8), mask03);
            let hi = _mm_and_si128(_mm_srli_epi16::<6>(hi8), mask03);
            let packed = decode_enc_i16_sse2(lo, hi);
            _mm_storeu_si128(dest4.as_mut_ptr().add(3 * n + j) as *mut __m128i, packed);
        }
        j += 16;
    }
    while j < n {
        let b = *prow.get_unchecked(j);
        for s in 0..4 {
            *dest4.get_unchecked_mut(s * n + j) = decode_nibble_i8((b >> (s * 2)) & 0x03);
        }
        j += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn saxpy_i8_sse2(c: &mut [f32], av: f32, w: &[i8]) {
    use std::arch::x86_64::*;
    let n = c.len();
    debug_assert_eq!(w.len(), n);
    let va = _mm_set1_ps(av);
    let mut j = 0usize;
    while j + 4 <= n {
        let mut tmp = [0i8; 4];
        tmp.copy_from_slice(&w[j..j + 4]);
        let wi = _mm_set_ps(tmp[3] as f32, tmp[2] as f32, tmp[1] as f32, tmp[0] as f32);
        let vc = _mm_loadu_ps(c.as_ptr().add(j));
        _mm_storeu_ps(c.as_mut_ptr().add(j), _mm_add_ps(vc, _mm_mul_ps(va, wi)));
        j += 4;
    }
    while j < n {
        c[j] += av * (w[j] as f32);
        j += 1;
    }
}

// ── x86_64 AVX2 + FMA (panel + optional rayon) ──────────────────────────────

/// Rows per panel for W-reuse microkernel.
const MR: usize = 4;
/// Use panel+rayon only when enough M rows to amortize thread pool + W sharing.
/// Below this: single-row AVX saxpy (best for small-batch inference).
/// Empirically M=32 still loses to sequential on this host; M≥64 wins.
const PAR_M_MIN: usize = 64;

/// AVX-512 VNNI: one `dpbusd` per 16 outputs folds the 4 unpacked K-rows
/// (`acc[j] += a0 w0[j] + … + a3 w3[j]`) without widening each weight to f32.
/// `dpbusd` is u8×i8, so activations are biased +128 and we subtract
/// `128 · Σ_s w_s[j]` (the four weights already sit in registers).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512vl,avx512bw,avx512f")]
unsafe fn matmul_avx512_vnni(
    a: &[f32],
    b_packed: &[u8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let mut a_i8 = vec![0i8; m * k];
    quantize_a_to_i8(a, &mut a_i8);
    let mut acc = vec![0i32; m * n];
    let mut w_i8 = vec![0i8; 4 * n];
    let kp = k / 4;
    for p in 0..kp {
        unpack_four_rows_avx2(&b_packed[p * n..(p + 1) * n], &mut w_i8);
        for i in 0..m {
            let mut avs = [0i8; 4];
            let mut any = false;
            for s in 0..4 {
                let av = *a_i8.get_unchecked(i * k + p * 4 + s);
                avs[s] = av;
                any |= av != 0;
            }
            if !any {
                continue;
            }
            let row = std::slice::from_raw_parts_mut(acc.as_mut_ptr().add(i * n), n);
            acc_k4_vnni(row, avs, &w_i8);
        }
    }
    store_i32_as_f32(&acc, c);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512vl,avx512bw,avx512f")]
unsafe fn acc_k4_vnni(acc: &mut [i32], avs: [i8; 4], w: &[i8]) {
    use std::arch::x86_64::*;
    let n = acc.len();
    debug_assert_eq!(w.len(), 4 * n);
    let au = [
        (avs[0] as i32 + 128) as u8,
        (avs[1] as i32 + 128) as u8,
        (avs[2] as i32 + 128) as u8,
        (avs[3] as i32 + 128) as u8,
    ];
    let src1 = _mm512_set1_epi32(i32::from_le_bytes(au));
    let mut j = 0usize;
    while j + 16 <= n {
        let w0 = _mm_loadu_si128(w.as_ptr().add(j) as *const __m128i);
        let w1 = _mm_loadu_si128(w.as_ptr().add(n + j) as *const __m128i);
        let w2 = _mm_loadu_si128(w.as_ptr().add(2 * n + j) as *const __m128i);
        let w3 = _mm_loadu_si128(w.as_ptr().add(3 * n + j) as *const __m128i);
        let a01 = _mm_unpacklo_epi8(w0, w1);
        let a23 = _mm_unpackhi_epi8(w0, w1);
        let b01 = _mm_unpacklo_epi8(w2, w3);
        let b23 = _mm_unpackhi_epi8(w2, w3);
        let c0 = _mm_unpacklo_epi16(a01, b01);
        let c1 = _mm_unpackhi_epi16(a01, b01);
        let c2 = _mm_unpacklo_epi16(a23, b23);
        let c3 = _mm_unpackhi_epi16(a23, b23);
        let src2 = _mm512_inserti32x4::<3>(
            _mm512_inserti32x4::<2>(
                _mm512_inserti32x4::<1>(_mm512_castsi128_si512(c0), c1),
                c2,
            ),
            c3,
        );
        let sumw = _mm512_add_epi32(
            _mm512_add_epi32(_mm512_cvtepi8_epi32(w0), _mm512_cvtepi8_epi32(w1)),
            _mm512_add_epi32(_mm512_cvtepi8_epi32(w2), _mm512_cvtepi8_epi32(w3)),
        );
        let prev = _mm512_loadu_si512(acc.as_ptr().add(j) as *const __m512i);
        let dotted = _mm512_dpbusd_epi32(prev, src1, src2);
        let corr = _mm512_sub_epi32(dotted, _mm512_slli_epi32::<7>(sumw));
        _mm512_storeu_si512(acc.as_mut_ptr().add(j) as *mut __m512i, corr);
        j += 16;
    }
    while j < n {
        let mut ssum = acc[j];
        for s in 0..4 {
            ssum += (avs[s] as i32) * (w[s * n + j] as i32);
        }
        acc[j] = ssum;
        j += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_avx2_fma(a: &[f32], b_packed: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    c.fill(0.0);
    let mut w_i8 = vec![0i8; 4 * n];
    let kp = k / 4;

    #[cfg(feature = "parallel")]
    if m >= PAR_M_MIN {
        matmul_avx2_fma_par(a, b_packed, c, m, n, k, &mut w_i8, kp);
        return;
    }
    for p in 0..kp {
        let prow = &b_packed[p * n..(p + 1) * n];
        unpack_four_rows_avx2(prow, &mut w_i8);
        for i in 0..m {
            let c_row = std::slice::from_raw_parts_mut(c.as_mut_ptr().add(i * n), n);
            for s in 0..4 {
                let av = *a.get_unchecked(i * k + p * 4 + s);
                if av == 0.0 {
                    continue;
                }
                saxpy_i8_avx2_fma(c_row, av, &w_i8[s * n..(s + 1) * n]);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "parallel"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_avx2_fma_par(
    a: &[f32],
    b_packed: &[u8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    w_i8: &mut [i8],
    kp: usize,
) {
    use rayon::prelude::*;

    for p in 0..kp {
        let prow = &b_packed[p * n..(p + 1) * n];
        unpack_four_rows_avx2(prow, w_i8);

        let n_panels = (m + MR - 1) / MR;
        let c_addr = c.as_mut_ptr() as usize;
        let a_addr = a.as_ptr() as usize;
        let w_addr = w_i8.as_ptr() as usize;
        (0..n_panels).into_par_iter().for_each(|pi| {
            let i0 = pi * MR;
            let mr = (m - i0).min(MR);
            unsafe {
                panel_kpack_avx2(
                    c_addr as *mut f32,
                    a_addr as *const f32,
                    w_addr as *const i8,
                    i0,
                    mr,
                    n,
                    k,
                    p,
                );
            }
        });
    }
}

#[cfg(all(target_arch = "x86_64", feature = "parallel"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn panel_kpack_avx2(
    c: *mut f32,
    a: *const f32,
    w: *const i8,
    i0: usize,
    mr: usize,
    n: usize,
    k: usize,
    p: usize,
) {
    for s in 0..4 {
        let mut avs = [0f32; MR];
        let mut any = false;
        for r in 0..mr {
            let av = *a.add((i0 + r) * k + p * 4 + s);
            avs[r] = av;
            any |= av != 0.0;
        }
        if !any {
            continue;
        }
        let wrow = std::slice::from_raw_parts(w.add(s * n), n);
        panel_saxpy_avx2(c, i0, mr, n, &avs, wrow);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn panel_saxpy_avx2(
    c: *mut f32,
    i0: usize,
    mr: usize,
    n: usize,
    avs: &[f32; MR],
    w: &[i8],
) {
    use std::arch::x86_64::*;
    debug_assert!(mr <= MR);
    debug_assert_eq!(w.len(), n);

    let mut j = 0usize;
    while j + 8 <= n {
        let vw = load_i8x8_as_f32x8(w.as_ptr().add(j));
        for r in 0..mr {
            let av = avs[r];
            if av == 0.0 {
                continue;
            }
            let va = _mm256_set1_ps(av);
            let cp = c.add((i0 + r) * n + j);
            let vc = _mm256_loadu_ps(cp);
            _mm256_storeu_ps(cp, _mm256_fmadd_ps(va, vw, vc));
        }
        j += 8;
    }
    while j < n {
        let wj = *w.get_unchecked(j) as f32;
        if wj != 0.0 {
            for r in 0..mr {
                let av = avs[r];
                if av != 0.0 {
                    *c.add((i0 + r) * n + j) += av * wj;
                }
            }
        }
        j += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_i8x8_as_f32x8(w: *const i8) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let w64 = std::ptr::read_unaligned(w as *const u64);
    let v8 = _mm_cvtsi64_si128(w64 as i64);
    let v16 = _mm_cvtepi8_epi16(v8);
    let f_lo = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(v16));
    let f_hi = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(v16, 8)));
    _mm256_insertf128_ps(_mm256_castps128_ps256(f_lo), f_hi, 1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn saxpy_i8_avx2_fma(c: &mut [f32], av: f32, w: &[i8]) {
    use std::arch::x86_64::*;
    let n = c.len();
    let va = _mm256_set1_ps(av);
    let mut j = 0usize;
    while j + 16 <= n {
        let vw0 = load_i8x8_as_f32x8(w.as_ptr().add(j));
        let vw1 = load_i8x8_as_f32x8(w.as_ptr().add(j + 8));
        let vc0 = _mm256_loadu_ps(c.as_ptr().add(j));
        let vc1 = _mm256_loadu_ps(c.as_ptr().add(j + 8));
        _mm256_storeu_ps(c.as_mut_ptr().add(j), _mm256_fmadd_ps(va, vw0, vc0));
        _mm256_storeu_ps(c.as_mut_ptr().add(j + 8), _mm256_fmadd_ps(va, vw1, vc1));
        j += 16;
    }
    while j + 8 <= n {
        let vw = load_i8x8_as_f32x8(w.as_ptr().add(j));
        let vc = _mm256_loadu_ps(c.as_ptr().add(j));
        _mm256_storeu_ps(c.as_mut_ptr().add(j), _mm256_fmadd_ps(va, vw, vc));
        j += 8;
    }
    while j < n {
        c[j] += av * (w[j] as f32);
        j += 1;
    }
}

/// SIMD unpack: 32 bytes / step, emit all 4 sub-rows into `dest4 = [r0|r1|r2|r3]`.
///
/// 2-bit codes index a 16-entry in-register table via `_mm256_shuffle_epi8`
/// (the AVX2 spelling of NEON `tbl` / wasm `i8x16_swizzle`). Byte-wise shifts
/// are `and` the field then `srli_epi16` — the pre-mask zeros bits that would
/// otherwise leak across the 16-bit lane (AVX2 has no `srli_epi8`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unpack_four_rows_avx2(prow: &[u8], dest4: &mut [i8]) {
    use std::arch::x86_64::*;
    let n = prow.len();
    debug_assert_eq!(dest4.len(), 4 * n);

    // Duplicated into both 128-bit lanes: pshufb is per-lane.
    let tbl = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ));
    let mask03 = _mm256_set1_epi8(0x03);

    let mut j = 0usize;
    while j + 32 <= n {
        let bytes = _mm256_loadu_si256(prow.as_ptr().add(j) as *const __m256i);

        let i0 = _mm256_and_si256(bytes, mask03);
        _mm256_storeu_si256(dest4.as_mut_ptr().add(j) as *mut __m256i, _mm256_shuffle_epi8(tbl, i0));

        let i1 = _mm256_and_si256(
            _mm256_srli_epi16::<2>(_mm256_and_si256(bytes, _mm256_set1_epi8(0x0C))),
            mask03,
        );
        _mm256_storeu_si256(
            dest4.as_mut_ptr().add(n + j) as *mut __m256i,
            _mm256_shuffle_epi8(tbl, i1),
        );

        let i2 = _mm256_and_si256(
            _mm256_srli_epi16::<4>(_mm256_and_si256(bytes, _mm256_set1_epi8(0x30))),
            mask03,
        );
        _mm256_storeu_si256(
            dest4.as_mut_ptr().add(2 * n + j) as *mut __m256i,
            _mm256_shuffle_epi8(tbl, i2),
        );

        let i3 = _mm256_and_si256(
            _mm256_srli_epi16::<6>(_mm256_and_si256(bytes, _mm256_set1_epi8(0xC0u8 as i8))),
            mask03,
        );
        _mm256_storeu_si256(
            dest4.as_mut_ptr().add(3 * n + j) as *mut __m256i,
            _mm256_shuffle_epi8(tbl, i3),
        );
        j += 32;
    }
    // 16-byte tail — reuse SSE2 four-row unpack on the suffix
    if j < n {
        let mut tmp = vec![0i8; 4 * (n - j)];
        unpack_four_rows_sse2(&prow[j..], &mut tmp);
        let rem = n - j;
        for s in 0..4 {
            std::ptr::copy_nonoverlapping(
                tmp.as_ptr().add(s * rem),
                dest4.as_mut_ptr().add(s * n + j),
                rem,
            );
        }
    }
}

// ── aarch64 NEON (panel + optional rayon) ───────────────────────────────────
//
// Mirrors the AVX2+FMA structure above at 128-bit width: unpack 16 packed
// bytes/step into 4 i8 rows, then either a single-row saxpy (small M) or a
// rayon panel microkernel (large M). NEON is baseline on aarch64, so unlike
// x86_64 there is no is_aarch64_feature_detected! gate — see active_simd_level.

/// N-block width (aarch64 NEON only): sized so a K-group unpack buffer + MR
/// C-panel stay inside L1d via `cache_detect`. **Hobby / optional** at shipped
/// d512 shapes (null on M1; unused on x86). Override `FLASH_TERNARY_NC`.
/// See `docs/notes/exp-sram-cache-block/REPORT.md`.
#[cfg(target_arch = "aarch64")]
fn tuned_nc(n: usize) -> usize {
    if let Ok(v) = std::env::var("FLASH_TERNARY_NC") {
        if let Ok(nc) = v.trim().parse::<usize>() {
            if nc > 0 {
                return nc.clamp(8, n);
            }
        }
    }
    let l1 = crate::cache_detect::detect().l1d;
    let bytes_per_elem = 4 + MR * 4; // unpack-buffer row + MR C-panel rows
    let budget = l1 / 2; // leave half of L1d for A-rows, avs, code, etc.
    let nc = (budget / bytes_per_elem).max(8);
    let nc = (nc / 8) * 8; // round to the NEON 8-wide step
    nc.clamp(8, n)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matmul_neon(a: &[f32], b_packed: &[u8], c: &mut [f32], m: usize, n: usize, k: usize) {
    c.fill(0.0);
    let kp = k / 4;
    let nc = tuned_nc(n);

    #[cfg(feature = "parallel")]
    if m >= PAR_M_MIN {
        matmul_neon_par(a, b_packed, c, m, n, k, nc);
        return;
    }
    let mut w_i8 = vec![0i8; 4 * nc];
    let n_panels = (m + MR - 1) / MR;
    let mut n0 = 0usize;
    while n0 < n {
        let nb = (n - n0).min(nc);
        for p in 0..kp {
            let prow = &b_packed[p * n + n0..p * n + n0 + nb];
            unpack_four_rows_neon(prow, &mut w_i8[..4 * nb]);
            for pi in 0..n_panels {
                let i0 = pi * MR;
                let mr = (m - i0).min(MR);
                panel_kpack_neon(
                    c.as_mut_ptr(),
                    a.as_ptr(),
                    w_i8.as_ptr(),
                    i0,
                    mr,
                    n,
                    n0,
                    nb,
                    k,
                    p,
                );
            }
        }
        n0 += nb;
    }
}

#[cfg(all(target_arch = "aarch64", feature = "parallel"))]
#[target_feature(enable = "neon")]
unsafe fn matmul_neon_par(
    a: &[f32],
    b_packed: &[u8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    nc: usize,
) {
    use rayon::prelude::*;

    let kp = k / 4;
    let n_panels = (m + MR - 1) / MR;
    let mut n0 = 0usize;
    while n0 < n {
        let nb = (n - n0).min(nc);
        let mut w_all = vec![0i8; kp * 4 * nb];
        for p in 0..kp {
            let prow = &b_packed[p * n + n0..p * n + n0 + nb];
            unpack_four_rows_neon(prow, &mut w_all[p * 4 * nb..(p + 1) * 4 * nb]);
        }
        let c_addr = c.as_mut_ptr() as usize;
        let a_addr = a.as_ptr() as usize;
        let w_addr = w_all.as_ptr() as usize;
        (0..n_panels).into_par_iter().for_each(|pi| {
            let i0 = pi * MR;
            let mr = (m - i0).min(MR);
            unsafe {
                for p in 0..kp {
                    let w = (w_addr as *const i8).add(p * 4 * nb);
                    panel_kpack_neon(
                        c_addr as *mut f32,
                        a_addr as *const f32,
                        w,
                        i0,
                        mr,
                        n,
                        n0,
                        nb,
                        k,
                        p,
                    );
                }
            }
        });
        n0 += nb;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn panel_kpack_neon(
    c: *mut f32,
    a: *const f32,
    w: *const i8,
    i0: usize,
    mr: usize,
    n: usize,
    n0: usize,
    nb: usize,
    k: usize,
    p: usize,
) {
    for s in 0..4 {
        let mut avs = [0f32; MR];
        let mut any = false;
        for r in 0..mr {
            let av = *a.add((i0 + r) * k + p * 4 + s);
            avs[r] = av;
            any |= av != 0.0;
        }
        if !any {
            continue;
        }
        let wrow = std::slice::from_raw_parts(w.add(s * nb), nb);
        panel_saxpy_neon(c, i0, mr, n, n0, &avs, wrow);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn panel_saxpy_neon(
    c: *mut f32,
    i0: usize,
    mr: usize,
    n: usize,
    n0: usize,
    avs: &[f32; MR],
    w: &[i8],
) {
    use std::arch::aarch64::*;
    debug_assert!(mr <= MR);
    let nb = w.len();
    let mut j = 0usize;
    while j + 8 <= nb {
        let (vw0, vw1) = load_i8x8_as_f32x4x2(w.as_ptr().add(j));
        for r in 0..mr {
            let av = avs[r];
            if av == 0.0 {
                continue;
            }
            let va = vdupq_n_f32(av);
            let cp0 = c.add((i0 + r) * n + n0 + j);
            let cp1 = c.add((i0 + r) * n + n0 + j + 4);
            vst1q_f32(cp0, vfmaq_f32(vld1q_f32(cp0), va, vw0));
            vst1q_f32(cp1, vfmaq_f32(vld1q_f32(cp1), va, vw1));
        }
        j += 8;
    }
    while j < nb {
        let wj = *w.get_unchecked(j) as f32;
        if wj != 0.0 {
            for r in 0..mr {
                let av = avs[r];
                if av != 0.0 {
                    *c.add((i0 + r) * n + n0 + j) += av * wj;
                }
            }
        }
        j += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_i8x8_as_f32x4x2(
    w: *const i8,
) -> (std::arch::aarch64::float32x4_t, std::arch::aarch64::float32x4_t) {
    use std::arch::aarch64::*;
    let v8 = vld1_s8(w);
    let v16 = vmovl_s8(v8);
    (
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16))),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16))),
    )
}

/// Unpack one packed row into 4 i8 weight rows (shifts 0,2,4,6) — NEON, 16 B/step.
///
/// The 2-bit code indexes a **16-entry in-register table** via `vqtbl1q_s8`
/// (`tbl`), replacing the `vceqq`×2 + `vsubq` decode. Two NEON-specific wins
/// over the AVX2 shape this was ported from:
///
/// - `vshrq_n_u8` shifts 8-bit lanes, so neighbouring bytes cannot contaminate
///   each other and the `vmovl_u8` widen to u16 (needed on x86, which has no
///   `_mm256_srli_epi8`) is unnecessary.
/// - `tbl` collapses the compare/subtract decode into one instruction.
///
/// ~10 ops per 16 bytes vs ~40 before: **4.75×** on the unpack primitive
/// (56.1 ns vs 266.2 ns for 1 KiB → 4 KiB on M1). End to end it only shows
/// where unpack is not amortized over M — 1.8× at M=1, noise at M ≥ 32.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unpack_four_rows_neon(prow: &[u8], dest4: &mut [i8]) {
    use std::arch::aarch64::*;
    let n = prow.len();
    debug_assert_eq!(dest4.len(), 4 * n);

    // Only indices 0..=3 are reachable (2-bit codes); the rest pad the table.
    const TBL: [i8; 16] = [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let tbl = vld1q_s8(TBL.as_ptr());
    let mask03 = vdupq_n_u8(0x03);

    let mut j = 0usize;
    while j + 16 <= n {
        let bytes = vld1q_u8(prow.as_ptr().add(j));

        let i0 = vandq_u8(bytes, mask03);
        vst1q_s8(dest4.as_mut_ptr().add(j), vqtbl1q_s8(tbl, i0));

        let i1 = vandq_u8(vshrq_n_u8::<2>(bytes), mask03);
        vst1q_s8(dest4.as_mut_ptr().add(n + j), vqtbl1q_s8(tbl, i1));

        let i2 = vandq_u8(vshrq_n_u8::<4>(bytes), mask03);
        vst1q_s8(dest4.as_mut_ptr().add(2 * n + j), vqtbl1q_s8(tbl, i2));

        let i3 = vshrq_n_u8::<6>(bytes); // top field needs no mask
        vst1q_s8(dest4.as_mut_ptr().add(3 * n + j), vqtbl1q_s8(tbl, i3));

        j += 16;
    }
    while j < n {
        let b = *prow.get_unchecked(j);
        for s in 0..4 {
            *dest4.get_unchecked_mut(s * n + j) = decode_nibble_i8((b >> (s * 2)) & 0x03);
        }
        j += 1;
    }
}

// ── C ABI (Python ctypes) — enable with feature `c_abi` ─────────────────────

#[cfg(feature = "c_abi")]
#[no_mangle]
pub unsafe extern "C" fn flash_ternary_pack(
    w: *const f32,
    out: *mut u8,
    k: usize,
    n: usize,
) -> i32 {
    if w.is_null() || out.is_null() {
        return -1;
    }
    let w_sl = std::slice::from_raw_parts(w, k * n);
    let out_sl = std::slice::from_raw_parts_mut(out, (k / 4) * n);
    if pack(w_sl, k, n, out_sl) {
        0
    } else {
        -2
    }
}

/// mode: 0=auto, 1=scalar, 2=simd
#[cfg(feature = "c_abi")]
#[no_mangle]
pub unsafe extern "C" fn flash_ternary_matmul_f32_ex(
    a: *const f32,
    b_packed: *const u8,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    mode: u32,
) -> i32 {
    if a.is_null() || b_packed.is_null() || c.is_null() {
        return -1;
    }
    let kernel = match mode {
        1 => MatmulKernel::Scalar,
        2 => MatmulKernel::Simd,
        _ => MatmulKernel::Auto,
    };
    let a_sl = std::slice::from_raw_parts(a, m * k);
    let b_sl = std::slice::from_raw_parts(b_packed, (k / 4) * n);
    let c_sl = std::slice::from_raw_parts_mut(c, m * n);
    if matmul_f32_ex(a_sl, b_sl, c_sl, m, n, k, kernel) {
        0
    } else {
        -2
    }
}

#[cfg(feature = "c_abi")]
#[no_mangle]
pub unsafe extern "C" fn flash_ternary_matmul_f32(
    a: *const f32,
    b_packed: *const u8,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) -> i32 {
    flash_ternary_matmul_f32_ex(a, b_packed, c, m, n, k, 0)
}

/// Returns pointer to static C string: "avx512vnni" | "avx2+fma" | "sse2" | "neon" | "none"
#[cfg(feature = "c_abi")]
#[no_mangle]
pub extern "C" fn flash_ternary_simd_level() -> *const std::os::raw::c_char {
    static VNNI: &[u8] = b"avx512vnni\0";
    static AVX: &[u8] = b"avx2+fma\0";
    static SSE: &[u8] = b"sse2\0";
    static NEON: &[u8] = b"neon\0";
    static NONE: &[u8] = b"none\0";
    match active_simd_level() {
        "avx512vnni" => VNNI.as_ptr() as *const _,
        "avx2+fma" => AVX.as_ptr() as *const _,
        "sse2" => SSE.as_ptr() as *const _,
        "neon" => NEON.as_ptr() as *const _,
        _ => NONE.as_ptr() as *const _,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip_style() {
        let w = [1.0f32, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let mut packed = [0u8; 2];
        assert!(pack(&w, 4, 2, &mut packed));
        let e0 = packed[0];
        assert_eq!(e0 & 0x03, 1);
        assert_eq!((e0 >> 2) & 0x03, 0);
        assert_eq!((e0 >> 4) & 0x03, 2);
        assert_eq!((e0 >> 6) & 0x03, 1);
    }

    fn dense_ref(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += a[i * k + kk] * w[kk * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    #[test]
    fn matmul_scalar_matches_dense() {
        let k = 64usize;
        let n = 33usize; // not multiple of 8 — tail coverage
        let m = 3usize;
        let w: Vec<f32> = (0..k * n)
            .map(|i| ((i % 3) as i32 - 1) as f32)
            .collect();
        let mut packed = vec![0u8; (k / 4) * n];
        assert!(pack(&w, k, n, &mut packed));
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as i32 - 8) as f32).collect();
        let mut c = vec![0f32; m * n];
        assert!(matmul_f32_ex(
            &a,
            &packed,
            &mut c,
            m,
            n,
            k,
            MatmulKernel::Scalar
        ));
        let ref_c = dense_ref(&a, &w, m, n, k);
        for i in 0..m * n {
            assert!(
                (c[i] - ref_c[i]).abs() < 1e-4,
                "scalar c[{i}]={} ref={}",
                c[i],
                ref_c[i]
            );
        }
    }

    #[test]
    fn matmul_auto_matches_dense() {
        let k = 128usize;
        let n = 64usize;
        let m = 4usize;
        let w: Vec<f32> = (0..k * n)
            .map(|i| ((i % 3) as i32 - 1) as f32)
            .collect();
        let mut packed = vec![0u8; (k / 4) * n];
        assert!(pack(&w, k, n, &mut packed));
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as i32 - 8) as f32).collect();
        let mut c_auto = vec![0f32; m * n];
        let mut c_sc = vec![0f32; m * n];
        assert!(matmul_f32_ex(
            &a,
            &packed,
            &mut c_auto,
            m,
            n,
            k,
            MatmulKernel::Auto
        ));
        assert!(matmul_f32_ex(
            &a,
            &packed,
            &mut c_sc,
            m,
            n,
            k,
            MatmulKernel::Scalar
        ));
        let ref_c = dense_ref(&a, &w, m, n, k);
        for i in 0..m * n {
            assert!((c_auto[i] - ref_c[i]).abs() < 1e-4, "auto vs dense");
            assert!((c_sc[i] - ref_c[i]).abs() < 1e-4, "scalar vs dense");
            assert!(
                (c_auto[i] - c_sc[i]).abs() < 1e-4,
                "auto vs scalar at {i}: {} vs {}",
                c_auto[i],
                c_sc[i]
            );
        }
        println!("simd_level={}", active_simd_level());
    }

    #[test]
    fn matmul_auto_integer_a_is_exact() {
        // Production pack_cpu A is i8-valued. i32 acc must match dense f32
        // bit-exactly (ternary × i8 fits the f32 mantissa).
        let k = 128usize;
        let n = 64usize;
        let m = 3usize;
        let w: Vec<f32> = (0..k * n)
            .map(|i| ((i % 3) as i32 - 1) as f32)
            .collect();
        let mut packed = vec![0u8; (k / 4) * n];
        assert!(pack(&w, k, n, &mut packed));
        let extremes = [-128.0f32, -64.0, -1.0, 0.0, 1.0, 64.0, 127.0];
        let a: Vec<f32> = (0..m * k).map(|i| extremes[i % extremes.len()]).collect();
        let mut c = vec![0f32; m * n];
        assert!(matmul_f32_ex(&a, &packed, &mut c, m, n, k, MatmulKernel::Auto));
        let ref_c = dense_ref(&a, &w, m, n, k);
        let mut max_abs = 0f32;
        for i in 0..m * n {
            max_abs = max_abs.max((c[i] - ref_c[i]).abs());
        }
        assert_eq!(max_abs, 0.0, "integer-A Auto vs dense max_abs={max_abs}");
    }

    /// Forces a tiny N-block width so a realistic N (not a multiple of it,
    /// and spanning multiple M-panels) exercises the n0/nb loop boundary in
    /// matmul_neon/matmul_neon_par, not just the single-block case the
    /// default (~3000-wide on this host) would hit for small test matrices.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn matmul_neon_nblock_matches_dense() {
        // SAFETY: no other test reads/writes FLASH_TERNARY_NC.
        unsafe {
            std::env::set_var("FLASH_TERNARY_NC", "16");
        }
        let k = 64usize;
        let n = 37usize; // not a multiple of nc=16 — exercises the tail block
        let m = 10usize; // > MR=4 — exercises multiple M-panels per N-block
        let w: Vec<f32> = (0..k * n).map(|i| ((i % 3) as i32 - 1) as f32).collect();
        let mut packed = vec![0u8; (k / 4) * n];
        assert!(pack(&w, k, n, &mut packed));
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as i32 - 8) as f32).collect();
        let mut c_auto = vec![0f32; m * n];
        assert!(matmul_f32_ex(&a, &packed, &mut c_auto, m, n, k, MatmulKernel::Auto));
        let ref_c = dense_ref(&a, &w, m, n, k);
        // SAFETY: no other test reads/writes FLASH_TERNARY_NC.
        unsafe {
            std::env::remove_var("FLASH_TERNARY_NC");
        }
        for i in 0..m * n {
            assert!(
                (c_auto[i] - ref_c[i]).abs() < 1e-4,
                "n-blocked neon vs dense at {i}: {} vs {}",
                c_auto[i],
                ref_c[i]
            );
        }
    }

    /// Same as matmul_neon_nblock_matches_dense but M >= PAR_M_MIN (64), so
    /// this exercises the rayon matmul_neon_par path's n0/nb loop instead of
    /// the sequential one.
    #[cfg(all(target_arch = "aarch64", feature = "parallel"))]
    #[test]
    fn matmul_neon_par_nblock_matches_dense() {
        // SAFETY: no other test reads/writes FLASH_TERNARY_NC.
        unsafe {
            std::env::set_var("FLASH_TERNARY_NC", "16");
        }
        let k = 64usize;
        let n = 37usize;
        let m = 80usize; // >= PAR_M_MIN
        let w: Vec<f32> = (0..k * n).map(|i| ((i % 3) as i32 - 1) as f32).collect();
        let mut packed = vec![0u8; (k / 4) * n];
        assert!(pack(&w, k, n, &mut packed));
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as i32 - 8) as f32).collect();
        let mut c_auto = vec![0f32; m * n];
        assert!(matmul_f32_ex(&a, &packed, &mut c_auto, m, n, k, MatmulKernel::Auto));
        let ref_c = dense_ref(&a, &w, m, n, k);
        // SAFETY: no other test reads/writes FLASH_TERNARY_NC.
        unsafe {
            std::env::remove_var("FLASH_TERNARY_NC");
        }
        for i in 0..m * n {
            assert!(
                (c_auto[i] - ref_c[i]).abs() < 1e-4,
                "n-blocked neon_par vs dense at {i}: {} vs {}",
                c_auto[i],
                ref_c[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn unpack_four_rows_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // N covers 32-wide + 16-wide + scalar tail
        let n = 32 + 16 + 7;
        let mut prow = vec![0u8; n];
        for j in 0..n {
            // pack all three encodings in each byte
            prow[j] = 0b01_00_10_01; // w0=+1, w1=-1, w2=0, w3=+1
            if j % 5 == 0 {
                prow[j] = 0b10_10_00_01;
            }
            if j % 7 == 0 {
                prow[j] = 0xFF; // invalid enc=3 treated as 0 by decode
            }
        }
        let mut dest_simd = vec![0i8; 4 * n];
        let mut dest_sc = vec![0i8; 4 * n];
        unsafe {
            unpack_four_rows_avx2(&prow, &mut dest_simd);
        }
        for j in 0..n {
            for s in 0..4 {
                dest_sc[s * n + j] = decode_nibble_i8((prow[j] >> (s * 2)) & 0x03);
            }
        }
        assert_eq!(dest_simd, dest_sc, "AVX2 unpack != scalar");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn unpack_four_rows_neon_matches_scalar() {
        // N covers 16-wide + 16-wide + scalar tail
        let n = 16 + 16 + 7;
        let mut prow = vec![0u8; n];
        for j in 0..n {
            prow[j] = 0b01_00_10_01; // w0=+1, w1=-1, w2=0, w3=+1
            if j % 5 == 0 {
                prow[j] = 0b10_10_00_01;
            }
            if j % 7 == 0 {
                prow[j] = 0xFF; // invalid enc=3 treated as 0 by decode
            }
        }
        let mut dest_simd = vec![0i8; 4 * n];
        let mut dest_sc = vec![0i8; 4 * n];
        unsafe {
            unpack_four_rows_neon(&prow, &mut dest_simd);
        }
        for j in 0..n {
            for s in 0..4 {
                dest_sc[s * n + j] = decode_nibble_i8((prow[j] >> (s * 2)) & 0x03);
            }
        }
        assert_eq!(dest_simd, dest_sc, "NEON unpack != scalar");
    }
}
