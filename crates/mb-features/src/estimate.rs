//! Estimate types and the training cost model.
//!
//! Every figure is a range with the assumptions that produced it. Plugins
//! describe *what* has to be trained ([`Stage`]); [`cost`] turns that into
//! compute and memory for a [`HardwareProfile`].

use serde::Serialize;

use crate::hardware::{Backend, HardwareProfile};

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Range {
    pub low: f64,
    pub high: f64,
}

impl Range {
    pub const fn new(low: f64, high: f64) -> Self {
        Self { low, high }
    }

    pub const fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    pub fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Self::new(f(self.low), f(self.high))
    }
}

impl std::ops::Add for Range {
    type Output = Self;

    fn add(self, o: Self) -> Self {
        Self::new(self.low + o.low, self.high + o.high)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug, Serialize)]
pub struct QualityRisk {
    pub level: RiskLevel,
    pub expected: String,
    pub recovery: String,
}

/// A measurable runtime effect of the feature, e.g. KV bytes per token.
#[derive(Clone, Debug, Serialize)]
pub struct Effect {
    pub metric: String,
    pub before: f64,
    pub after: f64,
    pub unit: String,
}

/// One training stage, described in terms the cost model can price.
#[derive(Clone, Debug, Serialize)]
pub struct Stage {
    pub name: String,
    pub what: String,
    /// Parameters updated by the optimizer.
    pub trainable_params: u64,
    /// Parameters that activation gradients flow back through (at least
    /// `trainable_params`; everything above the lowest trainable layer).
    pub backprop_params: u64,
    /// Training tokens.
    pub tokens: Range,
    pub seq_len: u64,
    pub loss: String,
    pub data: String,
    /// Whether a teacher forward pass (the unmodified model) runs per token.
    pub teacher_forward: bool,
}

/// Model-level inputs to the cost model.
#[derive(Clone, Debug, Serialize)]
pub struct CostInputs {
    /// Parameters in the forward pass (trunk, embeddings, head).
    pub forward_params: u64,
    pub hidden: u64,
    pub layers: u64,
    pub vocab: u64,
    /// Storage bits per weight of the source checkpoint (e.g. 2.125 for PQ2_0).
    pub source_bits: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ComputeEstimate {
    pub profile: &'static str,
    pub train_flops: Range,
    pub gpu_hours: Range,
    pub wall_hours: Range,
    /// Peak memory per device with the frozen trunk held in BF16.
    pub peak_gib_per_gpu: f64,
    /// Same, if the frozen trunk stays packed at the source bit width.
    pub peak_gib_per_gpu_packed_trunk: f64,
    pub fits: Fit,
    pub notes: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Fit {
    /// No training needed.
    NotApplicable,
    Yes,
    /// Only if the frozen trunk stays packed low-bit during training.
    WithPackedTrunk,
    No,
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
/// Micro-batch size in tokens assumed for activation memory.
pub const MICRO_BATCH_TOKENS: u64 = 4096;

fn mfu(backend: Backend) -> Range {
    match backend {
        Backend::Cuda => Range::new(0.25, 0.45),
        Backend::Metal => Range::new(0.15, 0.35),
    }
}

/// Assumptions shared by every compute estimate; shown once per plan.
pub fn cost_model_assumptions() -> Vec<String> {
    vec![
        "Training FLOPs per token = 2·forward params + 2·params backpropagated through + 2·trainable params (+ 2·forward params for a teacher pass).".into(),
        "Utilization (MFU) of spec-sheet peak: 25–45% on CUDA, 15–35% on Metal.".into(),
        "Trainable params cost 16 B each (BF16 weight + FP32 master + two FP32 Adam moments); frozen params 2 B each in BF16.".into(),
        format!("Activations: {MICRO_BATCH_TOKENS}-token micro-batch with activation checkpointing (≈2 B × hidden × layers per token) plus chunked BF16 logits."),
        "Multi-GPU profiles shard weights and optimizer state evenly (FSDP-style).".into(),
    ]
}

/// Prices a list of stages on one hardware profile.
pub fn cost(inputs: &CostInputs, stages: &[Stage], hw: &HardwareProfile) -> ComputeEstimate {
    if stages.is_empty() {
        return ComputeEstimate {
            profile: hw.id,
            train_flops: Range::zero(),
            gpu_hours: Range::zero(),
            wall_hours: Range::zero(),
            peak_gib_per_gpu: 0.0,
            peak_gib_per_gpu_packed_trunk: 0.0,
            fits: Fit::NotApplicable,
            notes: vec!["No training stages.".into()],
        };
    }
    let n = inputs.forward_params as f64;
    let mut flops = Range::zero();
    let mut peak_bf16: f64 = 0.0;
    let mut peak_packed: f64 = 0.0;
    for s in stages {
        let per_token = 2.0 * n
            + 2.0 * s.backprop_params as f64
            + 2.0 * s.trainable_params as f64
            + if s.teacher_forward { 2.0 * n } else { 0.0 };
        flops = flops + s.tokens.map(|t| t * per_token);

        let trainable = s.trainable_params as f64;
        let frozen = (n - trainable).max(0.0);
        // A teacher is the unmodified model: only the originals of the
        // modified tensors need to be kept alongside the student.
        let teacher_extra = if s.teacher_forward {
            trainable * 2.0
        } else {
            0.0
        };
        let state = trainable * 16.0 + teacher_extra;
        let act = MICRO_BATCH_TOKENS as f64
            * (2.0 * (inputs.hidden * inputs.layers) as f64
                + 2.0 * 16.0 * inputs.hidden as f64
                // BF16 logits, cross-entropy computed in 8 chunks.
                + 2.0 * inputs.vocab as f64 / 8.0);
        let g = f64::from(hw.gpus);
        peak_bf16 = peak_bf16.max(((frozen * 2.0 + state) / g + act) / GIB);
        peak_packed =
            peak_packed.max(((frozen * inputs.source_bits / 8.0 + state) / g + act) / GIB);
    }
    let per_gpu = hw.peak_tflops * 1e12;
    let gpu_hours = Range::new(
        flops.low / (per_gpu * mfu(hw.backend).high) / 3600.0,
        flops.high / (per_gpu * mfu(hw.backend).low) / 3600.0,
    );
    let usable = hw.usable_gib_per_gpu();
    let fits = if peak_bf16 <= usable {
        Fit::Yes
    } else if peak_packed <= usable {
        Fit::WithPackedTrunk
    } else {
        Fit::No
    };
    let mut notes = Vec::new();
    match fits {
        Fit::WithPackedTrunk => notes.push(format!(
            "Needs {peak_bf16:.0} GiB/GPU with a BF16 trunk; fits ({peak_packed:.0} GiB) only if the frozen trunk stays packed at {:.2} bits, which needs low-bit training kernels the HF backend doesn't have yet.",
            inputs.source_bits
        )),
        Fit::No => notes.push(format!(
            "Needs {peak_packed:.0}–{peak_bf16:.0} GiB/GPU vs {usable:.0} GiB usable: use more GPUs, LoRA on the trainable tensors, or fewer trainable layers per stage."
        )),
        _ => {}
    }
    ComputeEstimate {
        profile: hw.id,
        train_flops: flops,
        gpu_hours,
        wall_hours: gpu_hours.map(|h| h / f64::from(hw.gpus)),
        peak_gib_per_gpu: peak_bf16,
        peak_gib_per_gpu_packed_trunk: peak_packed,
        fits,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::profile;

    fn inputs() -> CostInputs {
        CostInputs {
            forward_params: 1_000_000_000,
            hidden: 2048,
            layers: 24,
            vocab: 32000,
            source_bits: 16.0,
        }
    }

    fn stage(trainable: u64, backprop: u64, tokens: f64, teacher: bool) -> Stage {
        Stage {
            name: "s".into(),
            what: String::new(),
            trainable_params: trainable,
            backprop_params: backprop,
            tokens: Range::new(tokens, tokens),
            seq_len: 4096,
            loss: String::new(),
            data: String::new(),
            teacher_forward: teacher,
        }
    }

    #[test]
    fn full_finetune_is_six_n_per_token() {
        let n = 1_000_000_000u64;
        let c = cost(
            &inputs(),
            &[stage(n, n, 1e9, false)],
            profile("8xH100").unwrap(),
        );
        assert_eq!(c.train_flops.low, 6.0 * 1e9 * 1e9);
        // 6e18 FLOPs at 989 TFLOPS × 45% ≈ 3.7 GPU-hours.
        assert!((c.gpu_hours.low - 6e18 / (989e12 * 0.45) / 3600.0).abs() < 1e-9);
        assert!((c.wall_hours.low - c.gpu_hours.low / 8.0).abs() < 1e-12);
    }

    #[test]
    fn head_only_training_is_cheap_and_fits() {
        let c = cost(
            &inputs(),
            &[stage(10_000_000, 10_000_000, 1e8, true)],
            profile("1x24GB").unwrap(),
        );
        // 2N (student) + 2N (teacher) + tiny head terms.
        let per_token = 4.0 * 1e9 + 4.0 * 1e7;
        assert!((c.train_flops.low - per_token * 1e8).abs() < 1.0);
        assert_eq!(c.fits, Fit::Yes);
    }

    #[test]
    fn packed_trunk_fallback() {
        let big = CostInputs {
            forward_params: 27_000_000_000,
            hidden: 5120,
            layers: 64,
            vocab: 248_320,
            source_bits: 2.125,
        };
        let c = cost(
            &big,
            &[stage(1_000_000_000, 1_000_000_000, 1e8, false)],
            profile("1x24GB").unwrap(),
        );
        assert_eq!(
            c.fits,
            Fit::No,
            "{} / {}",
            c.peak_gib_per_gpu,
            c.peak_gib_per_gpu_packed_trunk
        );
        let c = cost(
            &big,
            &[stage(50_000_000, 50_000_000, 1e8, false)],
            profile("1x24GB").unwrap(),
        );
        assert_eq!(
            c.fits,
            Fit::WithPackedTrunk,
            "{} / {}",
            c.peak_gib_per_gpu,
            c.peak_gib_per_gpu_packed_trunk
        );
        let c = cost(
            &big,
            &[stage(400_000_000, 400_000_000, 1e8, false)],
            profile("8xH100").unwrap(),
        );
        assert_eq!(c.fits, Fit::Yes);
    }

    #[test]
    fn no_stages_means_no_training() {
        let c = cost(&inputs(), &[], profile("1x24GB").unwrap());
        assert_eq!(c.fits, Fit::NotApplicable);
    }
}
