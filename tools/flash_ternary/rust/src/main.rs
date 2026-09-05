//! Tiny CLI: flash_ternary_cpu check|bench [M K N iters] [--scalar|--simd]
use flash_ternary_cpu::{active_simd_level, matmul_f32_ex, pack, MatmulKernel};
use std::env;
use std::time::Instant;

fn parse_kernel(args: &[String]) -> MatmulKernel {
    if args.iter().any(|a| a == "--scalar") {
        MatmulKernel::Scalar
    } else if args.iter().any(|a| a == "--simd") {
        MatmulKernel::Simd
    } else {
        MatmulKernel::Auto
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("check");
    let kernel = parse_kernel(&args);
    let nums: Vec<usize> = args
        .iter()
        .filter_map(|s| s.parse::<usize>().ok())
        .collect();

    match cmd {
        "check" => {
            let k = 256usize;
            let n = 128usize;
            let m = 8usize;
            let w: Vec<f32> = (0..k * n)
                .map(|i| ((i % 3) as i32 - 1) as f32)
                .collect();
            let mut packed = vec![0u8; (k / 4) * n];
            assert!(pack(&w, k, n, &mut packed));
            let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as i32 - 8) as f32).collect();
            let mut c = vec![0f32; m * n];
            assert!(matmul_f32_ex(&a, &packed, &mut c, m, n, k, kernel));
            let mut max_err = 0f32;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0f32;
                    for kk in 0..k {
                        acc += a[i * k + kk] * w[kk * n + j];
                    }
                    max_err = max_err.max((c[i * n + j] - acc).abs());
                }
            }
            println!(
                "OK rust check kernel={kernel:?} simd_hw={} max_abs_err={max_err:.3e} M={m} K={k} N={n}",
                active_simd_level()
            );
        }
        "bench" => {
            let m = nums.first().copied().unwrap_or(32);
            let k = nums.get(1).copied().unwrap_or(1024);
            let n = nums.get(2).copied().unwrap_or(1024);
            let iters = nums.get(3).copied().unwrap_or(20);
            let w: Vec<f32> = (0..k * n)
                .map(|i| ((i % 3) as i32 - 1) as f32)
                .collect();
            let mut packed = vec![0u8; (k / 4) * n];
            assert!(pack(&w, k, n, &mut packed));
            let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as i32 - 8) as f32).collect();
            let mut c = vec![0f32; m * n];
            for _ in 0..3 {
                assert!(matmul_f32_ex(&a, &packed, &mut c, m, n, k, kernel));
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                assert!(matmul_f32_ex(&a, &packed, &mut c, m, n, k, kernel));
            }
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            let gops = (2.0 * m as f64 * n as f64 * k as f64) / dt / 1e9;
            println!(
                "backend=rust kernel={kernel:?} hw={} M={m} K={k} N={n} latency={:.2} ms ~{gops:.1} GOPS",
                active_simd_level(),
                dt * 1e3
            );
        }
        other => {
            eprintln!("unknown cmd {other}; use check|bench [M K N iters] [--scalar|--simd]");
            std::process::exit(2);
        }
    }
}
