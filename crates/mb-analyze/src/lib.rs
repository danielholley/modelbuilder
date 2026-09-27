//! Analyzers that turn a [`ModelIr`] into a [`Report`].
//!
//! Everything here works from the tensor index and metadata only; no tensor
//! data is read. Weight statistics and behavioral probes build on top later.

mod kv;
mod params;
mod pattern;
mod provenance;
mod quant;

use mb_ir::{ModelIr, SourceFormat};
use serde::Serialize;

pub use kv::{KvCacheEstimate, KvPrecision};
pub use params::ParamBreakdown;
pub use pattern::{layer_label, LayerPattern};
pub use provenance::Provenance;
pub use quant::{DTypeStat, QuantSummary};

#[derive(Clone, Debug, Serialize)]
pub struct SourceSummary {
    pub format: SourceFormat,
    pub path: String,
    pub files: usize,
    pub tensors: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ArchSummary {
    pub family: Option<String>,
    pub architectures: Vec<String>,
    pub num_layers: usize,
    pub hidden_size: Option<u64>,
    pub vocab_size: Option<u64>,
    pub max_positions: Option<u64>,
    pub tie_word_embeddings: Option<bool>,
    pub rope_theta: Option<f64>,
    pub rope_scaling: Option<serde_json::Value>,
    pub layer_pattern: LayerPattern,
    pub mtp_modules: u64,
    pub multimodal: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub source: SourceSummary,
    pub architecture: ArchSummary,
    pub params: ParamBreakdown,
    pub quantization: QuantSummary,
    pub kv_cache: KvCacheEstimate,
    pub provenance: Provenance,
    pub warnings: Vec<String>,
}

/// Runs every static analyzer. `context` overrides the context length used
/// for KV-cache totals (defaults to the model's max positions).
pub fn analyze(ir: &ModelIr, context: Option<u64>) -> Report {
    Report {
        source: SourceSummary {
            format: ir.raw.format,
            path: ir.raw.root.display().to_string(),
            files: ir.raw.files.len(),
            tensors: ir.raw.tensors.len(),
        },
        architecture: ArchSummary {
            family: ir.family.clone(),
            architectures: ir.architectures.clone(),
            num_layers: ir.layers.len(),
            hidden_size: ir.hidden_size,
            vocab_size: ir.vocab_size,
            max_positions: ir.max_positions,
            tie_word_embeddings: ir.tie_word_embeddings,
            rope_theta: ir.rope.theta,
            rope_scaling: ir.rope.scaling.clone(),
            layer_pattern: LayerPattern::of(&ir.layers),
            mtp_modules: ir.mtp.as_ref().map_or(0, |m| m.num_modules),
            multimodal: ir.has_multimodal,
        },
        params: ParamBreakdown::of(ir),
        quantization: QuantSummary::of(ir),
        kv_cache: KvCacheEstimate::of(ir, context),
        provenance: Provenance::of(ir),
        warnings: ir.warnings.clone(),
    }
}
