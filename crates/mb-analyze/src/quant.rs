use std::collections::BTreeMap;

use mb_ir::{Component, ConfigView, DType, ModelIr, WeightRotation};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct DTypeStat {
    pub dtype: String,
    pub tensors: usize,
    pub params: u64,
    pub bytes: u64,
    /// False if any tensor's size had to be inferred (unknown GGUF type).
    pub bytes_exact: bool,
    /// Measured storage bits per parameter for this dtype.
    pub bits_per_param: f64,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct QuantSummary {
    pub by_dtype: Vec<DTypeStat>,
    pub total_bytes: u64,
    /// Storage bits per parameter across the whole checkpoint.
    pub bits_per_param: f64,
    /// Same, for the main trunk's layer weights only (the part that's usually quantized).
    pub trunk_bits_per_param: f64,
    /// What the checkpoint declares (HF `quantization_config`, GGUF `general.file_type`).
    pub declared: Option<serde_json::Value>,
    /// Rotation folded into the stored weights, if the checkpoint declares one.
    pub rotation: Option<WeightRotation>,
    pub notes: Vec<String>,
}

fn bits(bytes: u64, params: u64) -> f64 {
    if params == 0 {
        0.0
    } else {
        bytes as f64 * 8.0 / params as f64
    }
}

impl QuantSummary {
    pub fn of(ir: &ModelIr) -> Self {
        let mut by: BTreeMap<String, DTypeStat> = BTreeMap::new();
        let (mut trunk_bytes, mut trunk_params) = (0u64, 0u64);
        let mut unknown_types = Vec::new();
        for (t, r) in ir.tensors() {
            let name = t.dtype.name();
            let s = by.entry(name.clone()).or_insert_with(|| DTypeStat {
                dtype: name.clone(),
                tensors: 0,
                params: 0,
                bytes: 0,
                bytes_exact: true,
                bits_per_param: 0.0,
            });
            s.tensors += 1;
            s.params += t.n_elements();
            s.bytes += t.n_bytes;
            s.bytes_exact &= t.bytes_exact;
            if r.component == Component::Layer {
                trunk_bytes += t.n_bytes;
                trunk_params += t.n_elements();
            }
            if matches!(t.dtype, DType::Ggml(g) if g.known_name().is_none())
                && !unknown_types.contains(&name)
            {
                unknown_types.push(name);
            }
        }
        let mut by_dtype: Vec<DTypeStat> = by.into_values().collect();
        for s in &mut by_dtype {
            s.bits_per_param = bits(s.bytes, s.params);
        }
        by_dtype.sort_by_key(|s| std::cmp::Reverse(s.bytes));

        let total_bytes: u64 = by_dtype.iter().map(|s| s.bytes).sum();
        let total_params: u64 = by_dtype.iter().map(|s| s.params).sum();

        let mut notes = Vec::new();
        for u in &unknown_types {
            notes.push(format!(
                "{u} is not an upstream ggml type (vendor-specific?). Its size was inferred from tensor offsets and may include padding."
            ));
        }
        let vendor: Vec<&str> = by_dtype
            .iter()
            .map(|s| s.dtype.as_str())
            .filter(|d| ["PQ2_0", "PTQ1_0"].contains(d))
            .collect();
        if !vendor.is_empty() {
            notes.push(format!(
                "{} are PrismML vendor types: they need the PrismML-Eng/llama.cpp fork and are rejected by stock llama.cpp.",
                vendor.join(", ")
            ));
        }
        if let Some(r) = &ir.weight_rotation {
            notes.push(format!(
                "Weights are stored in a rotated basis ({}, block {}, {} tensors). Weight statistics must undo it, and surgery must keep new or modified tensors in the same basis and update the `{}.*` metadata.",
                r.scheme,
                r.block_size.map_or("?".into(), |b| b.to_string()),
                r.rotated_tensors,
                r.metadata_prefix
            ));
        }
        let cfg = ConfigView::new(&ir.raw.metadata);
        let declared = cfg.quantization_config().cloned().or_else(|| {
            cfg.gguf_raw("general.file_type")
                .map(|v| v.to_display_json())
        });
        let packed = by_dtype
            .iter()
            .any(|s| matches!(s.dtype.as_str(), "U8" | "I8" | "I32" | "U32"))
            && declared.is_some();
        if packed {
            notes.push("Integer tensors alongside a quantization_config are probably packed; parameter counts for them are storage elements, not weights.".into());
        }

        Self {
            by_dtype,
            total_bytes,
            bits_per_param: bits(total_bytes, total_params),
            trunk_bits_per_param: bits(trunk_bytes, trunk_params),
            declared,
            rotation: ir.weight_rotation.clone(),
            notes,
        }
    }
}
