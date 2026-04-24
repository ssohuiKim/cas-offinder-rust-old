//! Runtime device configuration for the CUDA port.
//!
//! The legacy OpenCL implementation distinguished CPU / GPU / Accelerator
//! platforms at the OpenCL level. CUDA only exposes NVIDIA GPUs, so the
//! available choices collapse to "use the CPU fallback" vs "use the CUDA
//! GPUs discovered on this host".

use cudarc::driver::CudaContext;
use std::sync::Arc;
use std::thread;

/// Selection passed by the CLI. `Cpu` runs the pure-Rust CPU pipeline;
/// `Gpu`/`Accel`/`All` all route to CUDA devices (CUDA has no concept of
/// non-GPU accelerators so those aliases just map to GPU here for backward
/// compatibility with the original CLI argument set).
pub enum OclDeviceType {
    CPU,
    GPU,
    ACCEL,
    ALL,
}

/// Holds one primary context per CUDA-visible GPU. When `contexts` is empty
/// the caller falls back to the CPU implementation (either the user asked
/// for `C` or no CUDA devices were found).
pub struct OclRunConfig {
    contexts: Vec<Arc<CudaContext>>,
}

impl OclRunConfig {
    pub fn new(ty: OclDeviceType) -> Result<Self, Box<dyn std::error::Error>> {
        let contexts = match ty {
            OclDeviceType::CPU => Vec::new(),
            OclDeviceType::GPU | OclDeviceType::ACCEL | OclDeviceType::ALL => {
                let n = CudaContext::device_count().unwrap_or(0).max(0) as usize;
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    if let Ok(ctx) = CudaContext::new(i) {
                        v.push(ctx);
                    }
                }
                v
            }
        };
        Ok(Self { contexts })
    }

    pub fn is_empty(&self) -> bool {
        self.contexts.is_empty()
    }

    pub fn contexts(&self) -> &[Arc<CudaContext>] {
        &self.contexts
    }

    pub fn get_device_strs(&self) -> Vec<String> {
        if self.is_empty() {
            let n_threads: usize = thread::available_parallelism().unwrap().into();
            vec![format!(
                "Rust CPU implementation using {} threads",
                n_threads
            )]
        } else {
            self.contexts
                .iter()
                .enumerate()
                .map(|(i, ctx)| {
                    ctx.name()
                        .unwrap_or_else(|_| format!("CUDA device {}", i))
                })
                .collect()
        }
    }
}
