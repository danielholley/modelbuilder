//! Hardware profiles used by cost models.
//!
//! Throughput figures are vendor spec-sheet peaks for dense BF16/FP16 matmul.
//! Real training runs reach a fraction of that (model FLOPs utilization, MFU),
//! which the cost model applies as an explicit range.

use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cuda,
    Metal,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HardwareProfile {
    pub id: &'static str,
    pub description: &'static str,
    pub backend: Backend,
    pub gpus: u32,
    /// Memory per device, GiB (unified memory for Apple Silicon).
    pub mem_per_gpu_gib: f64,
    /// Share of device memory a training job can actually use.
    pub usable_fraction: f64,
    /// Spec-sheet dense BF16/FP16 TFLOPS per device.
    pub peak_tflops: f64,
}

const PROFILES: &[HardwareProfile] = &[
    HardwareProfile {
        id: "1x24GB",
        description: "one 24 GB consumer GPU (RTX 4090 class)",
        backend: Backend::Cuda,
        gpus: 1,
        mem_per_gpu_gib: 24.0,
        usable_fraction: 0.9,
        peak_tflops: 165.0,
    },
    HardwareProfile {
        id: "1x32GB",
        description: "one 32 GB consumer GPU (RTX 5090 class)",
        backend: Backend::Cuda,
        gpus: 1,
        mem_per_gpu_gib: 32.0,
        usable_fraction: 0.9,
        peak_tflops: 209.0,
    },
    HardwareProfile {
        id: "1x80GB",
        description: "one 80 GB datacenter GPU (H100 SXM class)",
        backend: Backend::Cuda,
        gpus: 1,
        mem_per_gpu_gib: 80.0,
        usable_fraction: 0.9,
        peak_tflops: 989.0,
    },
    HardwareProfile {
        id: "8xH100",
        description: "one node of 8 × H100 SXM 80 GB",
        backend: Backend::Cuda,
        gpus: 8,
        mem_per_gpu_gib: 80.0,
        usable_fraction: 0.9,
        peak_tflops: 989.0,
    },
    HardwareProfile {
        id: "m3-max-128gb",
        description: "Apple M3 Max, 128 GB unified memory (40-core GPU)",
        backend: Backend::Metal,
        gpus: 1,
        mem_per_gpu_gib: 128.0,
        // macOS caps GPU-wired memory well below the total by default.
        usable_fraction: 0.75,
        peak_tflops: 16.0,
    },
];

pub fn profiles() -> &'static [HardwareProfile] {
    PROFILES
}

pub fn profile(id: &str) -> Option<&'static HardwareProfile> {
    PROFILES.iter().find(|p| p.id.eq_ignore_ascii_case(id))
}

impl HardwareProfile {
    pub fn usable_gib_per_gpu(&self) -> f64 {
        self.mem_per_gpu_gib * self.usable_fraction
    }
}
