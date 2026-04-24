//! Compile the `.cu` kernels to PTX at build time so the Rust host code can
//! `include_str!` the resulting text blob and hand it to CUDA at runtime via
//! `Module::load_from_ptx` (cudarc). The PTX stays virtual (`compute_80`),
//! and the CUDA JIT lowers it to whatever SM version the device actually
//! supports — keeps a single artifact working on RTX 4090 (sm_89),
//! RTX 5090 (sm_120), and other modern GPUs.

use std::env;
use std::path::PathBuf;
use std::process::Command;

const KERNELS: &[(&str, &str)] = &[
    ("src/kernel.cu", "kernel.ptx"),
    ("src/kernel_myers.cu", "kernel_myers.ptx"),
];

fn nvcc_path() -> PathBuf {
    let root = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .or_else(|_| env::var("CUDA_ROOT"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    PathBuf::from(root).join("bin").join("nvcc")
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR unset"));
    let nvcc = nvcc_path();
    if !nvcc.exists() {
        panic!(
            "nvcc not found at {:?}. Set CUDA_PATH / CUDA_HOME to the CUDA \
             toolkit root (e.g. the micromamba env prefix that holds bin/nvcc).",
            nvcc
        );
    }

    for (src, out_name) in KERNELS {
        let out = out_dir.join(out_name);
        let status = Command::new(&nvcc)
            .arg("-ptx")
            .arg("-arch=compute_80")
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-o")
            .arg(&out)
            .arg(src)
            .status()
            .unwrap_or_else(|e| panic!("failed to invoke {:?}: {e}", nvcc));
        if !status.success() {
            panic!("nvcc failed for {src} (exit {:?})", status.code());
        }
        println!("cargo:rerun-if-changed={src}");
    }

    // Expose PTX paths to src/ via env!().
    let kernel_ptx = out_dir.join("kernel.ptx");
    let kernel_myers_ptx = out_dir.join("kernel_myers.ptx");
    println!("cargo:rustc-env=PTX_KERNEL={}", kernel_ptx.display());
    println!(
        "cargo:rustc-env=PTX_KERNEL_MYERS={}",
        kernel_myers_ptx.display()
    );

    // Re-run if CUDA root env changes so we don't cache stale nvcc paths.
    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
}
