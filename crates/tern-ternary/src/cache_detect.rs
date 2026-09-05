//! Runtime cache-size self-detection ("meta programming": ask the OS what
//! its own L1d/L2 sizes are instead of hardcoding per-arch constants).
//!
//! **Status (2026-08-12): hobby / optional infrastructure.** Used only to size
//! NEON N-blocks (`tuned_nc` on aarch64). Measured null at shipped d512 shapes
//! (working set fits L2); x86 AVX2 path does not consume these sizes. Keep for
//! large-backbone experiments — not a production RAG/QAT speed lever.
//! See `docs/notes/exp-sram-cache-block/REPORT.md`.
//!
//! macOS: `sysctl hw.perflevel0.{l1dcachesize,l2cachesize}` (P-core; falls
//! back to the unqualified `hw.l1dcachesize`/`hw.l2cachesize` on Intel Macs
//! that have no perflevel split).
//! Linux: `/sys/devices/system/cpu/cpu0/cache/index*/{level,type,size}` —
//! the same sysfs tree `getconf LEVEL1_DCACHE_SIZE` reads from.
//!
//! Detected once per process (OnceLock) and cached. Override for testing /
//! unusual hosts with `FLASH_TERNARY_L1_BYTES` / `FLASH_TERNARY_L2_BYTES`.
//! Falls back to the smaller of the two hosts we've actually measured
//! (32KB L1 / 1MB L2 — a Linux x86_64 box) if detection fails entirely, so
//! an unrecognized target still gets a conservative-but-sane block size.

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug)]
pub struct CacheSizes {
    pub l1d: usize,
    pub l2: usize,
}

const FALLBACK: CacheSizes = CacheSizes {
    l1d: 32 * 1024,
    l2: 1024 * 1024,
};

pub fn detect() -> CacheSizes {
    static CACHE: OnceLock<CacheSizes> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let mut sizes = detect_uncached().unwrap_or(FALLBACK);
        if let Some(v) = env_bytes("FLASH_TERNARY_L1_BYTES") {
            sizes.l1d = v;
        }
        if let Some(v) = env_bytes("FLASH_TERNARY_L2_BYTES") {
            sizes.l2 = v;
        }
        sizes
    })
}

fn env_bytes(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

fn detect_uncached() -> Option<CacheSizes> {
    #[cfg(target_os = "macos")]
    {
        return detect_macos();
    }
    #[cfg(target_os = "linux")]
    {
        return detect_linux();
    }
    #[allow(unreachable_code)]
    None
}

#[cfg(target_os = "macos")]
fn detect_macos() -> Option<CacheSizes> {
    let l1d = sysctl_u64("hw.perflevel0.l1dcachesize").or_else(|| sysctl_u64("hw.l1dcachesize"))?;
    let l2 = sysctl_u64("hw.perflevel0.l2cachesize").or_else(|| sysctl_u64("hw.l2cachesize"))?;
    Some(CacheSizes {
        l1d: l1d as usize,
        l2: l2 as usize,
    })
}

#[cfg(target_os = "macos")]
fn sysctl_u64(name: &str) -> Option<u64> {
    let out = std::process::Command::new("sysctl")
        .arg("-n")
        .arg(name)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

#[cfg(target_os = "linux")]
fn detect_linux() -> Option<CacheSizes> {
    let mut l1d = None;
    let mut l2 = None;
    for idx in 0..8 {
        let base = format!("/sys/devices/system/cpu/cpu0/cache/index{idx}");
        let Ok(level_s) = std::fs::read_to_string(format!("{base}/level")) else {
            continue;
        };
        let Ok(level) = level_s.trim().parse::<u32>() else {
            continue;
        };
        let typ = std::fs::read_to_string(format!("{base}/type")).unwrap_or_default();
        if typ.trim() != "Data" && typ.trim() != "Unified" {
            continue;
        }
        let Ok(size_s) = std::fs::read_to_string(format!("{base}/size")) else {
            continue;
        };
        let Some(bytes) = parse_linux_size(size_s.trim()) else {
            continue;
        };
        match level {
            1 => l1d = Some(bytes),
            2 => l2 = Some(bytes),
            _ => {}
        }
    }
    Some(CacheSizes {
        l1d: l1d?,
        l2: l2?,
    })
}

#[cfg(target_os = "linux")]
fn parse_linux_size(s: &str) -> Option<usize> {
    if let Some(num) = s.strip_suffix('K') {
        num.parse::<usize>().ok().map(|k| k * 1024)
    } else if let Some(num) = s.strip_suffix('M') {
        num.parse::<usize>().ok().map(|m| m * 1024 * 1024)
    } else {
        s.parse::<usize>().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_plausible_sizes() {
        let sizes = detect();
        // Sanity bounds — real L1d is 8KB-256KB, L2 is 128KB-64MB on every
        // host we target. Catches a parse bug returning 0 or overflow.
        assert!(sizes.l1d >= 8 * 1024 && sizes.l1d <= 256 * 1024, "{sizes:?}");
        assert!(sizes.l2 >= 128 * 1024 && sizes.l2 <= 64 * 1024 * 1024, "{sizes:?}");
    }
}
