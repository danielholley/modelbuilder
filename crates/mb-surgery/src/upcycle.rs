//! Dense → MoE sparse upcycling of a GGUF (Komatsuzaki et al., 2022, arXiv
//! 2212.05055, §3): every expert starts as a copy of the layer's dense MLP and
//! a new router is initialized at random.
//!
//! Expert tensors are the dense tensors' stored bytes repeated `experts`
//! times along a new leading axis (`ffn_gate_exps [E, ff, hidden]` row-major,
//! ggml `ne = [hidden, ff, E]`). Quantized blocks are never re-encoded, so a
//! ternary or rotated source stays exactly as it was, and the rotation list
//! (`prism.hadamard.weight_names`) is renamed to the expert tensors.
//!
//! llama.cpp routes with softmax, top-k and renormalized weights for the
//! architectures supported here (Mixtral-style `llama` with `expert_count`,
//! and `qwen3moe`), so identical experts compute the dense MLP exactly: the
//! upcycled model starts with the dense model's function, whatever the router.

use std::path::Path;

use mb_formats::{gguf, LoadedModel};
use mb_ir::{DType, MetaValue, ModelIr};

use crate::gguf_edit::{check_output, file_len, gguf_kv, layer_of, set};
use crate::{SurgeryError, SurgeryReport, WrittenTensor};

/// A dense architecture llama.cpp can also run with routed experts.
#[derive(Clone, Copy, Debug)]
pub struct MoeArch {
    /// `general.architecture` of the dense model.
    pub dense: &'static str,
    /// `general.architecture` of the upcycled model (keys are renamed if it differs).
    pub moe: &'static str,
    /// The loader reads `{arch}.expert_feed_forward_length`.
    pub expert_ff_key: bool,
}

/// Dense → MoE architectures whose llama.cpp loaders expect exactly the
/// tensors written here (`ffn_gate_inp`, `ffn_{gate,up,down}_exps`) in every
/// layer, with no shared expert, and route with renormalized top-k softmax.
pub const MOE_ARCHS: &[MoeArch] = &[
    // Mixtral layout: `llama` with `llama.expert_count > 0`.
    MoeArch {
        dense: "llama",
        moe: "llama",
        expert_ff_key: false,
    },
    MoeArch {
        dense: "qwen3",
        moe: "qwen3moe",
        expert_ff_key: true,
    },
];

#[derive(Clone, Debug)]
pub struct UpcycleOptions {
    pub experts: u32,
    pub top_k: u32,
    /// Standard deviation of the router's Gaussian initialization.
    pub router_std: f32,
    pub seed: u64,
    pub overwrite: bool,
}

impl Default for UpcycleOptions {
    fn default() -> Self {
        Self {
            experts: 8,
            top_k: 2,
            router_std: 0.02,
            seed: 0,
            overwrite: false,
        }
    }
}

/// SplitMix64 plus Box-Muller: a deterministic Gaussian stream.
struct Gauss(u64);

impl Gauss {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
    }
}

const DENSE: [(&str, &str); 3] = [
    ("ffn_gate.weight", "ffn_gate_exps.weight"),
    ("ffn_up.weight", "ffn_up_exps.weight"),
    ("ffn_down.weight", "ffn_down_exps.weight"),
];

fn expert_name(name: &str) -> Option<String> {
    let (i, rest) = layer_of(name)?;
    DENSE
        .iter()
        .find(|(d, _)| *d == rest)
        .map(|(_, e)| format!("blk.{i}.{e}"))
}

/// Upcycles every layer's dense MLP into `experts` routed copies.
pub fn upcycle(
    model: &LoadedModel,
    ir: &ModelIr,
    opts: &UpcycleOptions,
    out: &Path,
) -> Result<SurgeryReport, SurgeryError> {
    let (mut kv, arch) = gguf_kv(ir)?;
    check_output(ir, out, opts.overwrite)?;
    let a = MOE_ARCHS.iter().find(|a| a.dense == arch).ok_or_else(|| {
        let known: Vec<&str> = MOE_ARCHS.iter().map(|a| a.dense).collect();
        SurgeryError::Unsupported(format!(
            "no dense → MoE mapping for `{arch}` (have: {}). Hybrid architectures such as qwen35 \
             have a MoE counterpart (qwen35moe) whose loader expects a shared expert; that \
             layout isn't verified here yet",
            known.join(", ")
        ))
    })?;
    if opts.experts < 2 || opts.top_k == 0 || opts.top_k > opts.experts {
        return Err(SurgeryError::Incompatible(format!(
            "need experts ≥ 2 and 1 ≤ top_k ≤ experts (got {} and {})",
            opts.experts, opts.top_k
        )));
    }
    if !(opts.router_std.is_finite() && opts.router_std >= 0.0) {
        return Err(SurgeryError::Incompatible("router_std must be ≥ 0".into()));
    }
    let get = |k: &str| kv.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone());
    if get(&format!("{arch}.expert_count"))
        .and_then(|v| v.as_u64())
        .is_some_and(|n| n > 0)
    {
        return Err(SurgeryError::Incompatible(
            "the model already has experts".into(),
        ));
    }
    let hidden = get(&format!("{arch}.embedding_length"))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| SurgeryError::Incompatible(format!("no {arch}.embedding_length")))?;
    let ff = get(&format!("{arch}.feed_forward_length"));
    let n_layers = ir.layers.len() as u32;
    // Every trunk layer needs all three dense tensors: these loaders make every layer MoE.
    for l in 0..n_layers {
        for (d, _) in DENSE {
            if model.tensor(&format!("blk.{l}.{d}")).is_none() {
                return Err(SurgeryError::Incompatible(format!(
                    "layer {l} has no {d}; llama.cpp's {} loader expects every layer to be MoE",
                    a.moe
                )));
            }
        }
    }

    let mut changed = Vec::new();
    if a.moe != arch {
        let (from, to) = (format!("{arch}."), format!("{}.", a.moe));
        for (k, _) in kv.iter_mut() {
            if let Some(rest) = k.strip_prefix(&from) {
                *k = format!("{to}{rest}");
            }
        }
        set(
            &mut kv,
            "general.architecture",
            MetaValue::String(a.moe.into()),
        );
        changed.push(format!(
            "general.architecture: {arch} → {} (and every {from}* key renamed)",
            a.moe
        ));
    }
    let p = a.moe;
    set(
        &mut kv,
        &format!("{p}.expert_count"),
        MetaValue::U32(opts.experts),
    );
    set(
        &mut kv,
        &format!("{p}.expert_used_count"),
        MetaValue::U32(opts.top_k),
    );
    changed.push(format!(
        "{p}.expert_count = {}, {p}.expert_used_count = {}",
        opts.experts, opts.top_k
    ));
    if a.expert_ff_key {
        let ff =
            ff.ok_or_else(|| SurgeryError::Incompatible(format!("no {arch}.feed_forward_length")))?;
        let ff = ff.as_u64().unwrap_or(0) as u32;
        set(
            &mut kv,
            &format!("{p}.expert_feed_forward_length"),
            MetaValue::U32(ff),
        );
        changed.push(format!("{p}.expert_feed_forward_length = {ff}"));
    }
    // Rotated (or otherwise listed) dense tensors are now expert tensors.
    for (k, v) in kv.iter_mut() {
        if let MetaValue::Array { values, .. } = v {
            let mut renamed = false;
            for x in values.iter_mut() {
                if let Some(e) = x.as_str().and_then(expert_name) {
                    *x = MetaValue::String(e);
                    renamed = true;
                }
            }
            if renamed {
                changed.push(format!("{k}: dense MLP names → expert names"));
            }
        }
    }
    set(
        &mut kv,
        "modelbuilder.upcycled",
        MetaValue::String(format!(
            "sparse upcycling: {} experts, top-{}, router N(0, {}²), seed {}",
            opts.experts, opts.top_k, opts.router_std, opts.seed
        )),
    );
    changed.push("modelbuilder.upcycled (added)".into());

    let mut rng = Gauss(opts.seed);
    let routers: Vec<Vec<u8>> = (0..n_layers)
        .map(|_| {
            (0..u64::from(opts.experts) * hidden)
                .flat_map(|_| (rng.normal() * opts.router_std).to_le_bytes())
                .collect()
        })
        .collect();

    let e = opts.experts as usize;
    let mut tensors = Vec::new();
    let mut written = Vec::new();
    for t in &ir.raw.tensors {
        let bytes = model.tensor_bytes(t)?;
        let trunk_layer = layer_of(&t.name).is_some_and(|(i, _)| i < n_layers);
        match expert_name(&t.name).filter(|_| trunk_layer) {
            Some(name) => {
                let mut shape = vec![u64::from(opts.experts)];
                shape.extend(&t.shape);
                written.push(WrittenTensor {
                    name: name.clone(),
                    dtype: t.dtype.to_string(),
                    shape: shape.clone(),
                    source: format!("target:{}", t.name),
                    transform: format!("{e} copies of the stored bytes"),
                });
                tensors.push(gguf::TensorParts {
                    name,
                    dtype: t.dtype,
                    shape,
                    parts: vec![bytes; e],
                });
                // The router goes right after the layer's last MLP tensor.
                if t.name.ends_with("ffn_down.weight") {
                    let (l, _) = layer_of(&t.name).expect("layer tensor");
                    let name = format!("blk.{l}.ffn_gate_inp.weight");
                    written.push(WrittenTensor {
                        name: name.clone(),
                        dtype: "F32".into(),
                        shape: vec![u64::from(opts.experts), hidden],
                        source: "new".into(),
                        transform: format!("router, N(0, {}²)", opts.router_std),
                    });
                    tensors.push(gguf::TensorParts {
                        name,
                        dtype: DType::F32,
                        shape: vec![u64::from(opts.experts), hidden],
                        parts: vec![&routers[l as usize]],
                    });
                }
            }
            None => tensors.push(gguf::TensorParts {
                name: t.name.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                parts: vec![bytes],
            }),
        }
    }
    gguf::write_parts(out, &kv, &tensors)?;
    let mut notes = vec![
        format!(
            "Upcycled {n_layers} layers to {} experts (top-{}); every other tensor copied byte for byte.",
            opts.experts, opts.top_k
        ),
        format!(
            "llama.cpp's {} graph renormalizes the top-k weights, so before training the model computes the dense model's function exactly.",
            a.moe
        ),
        "Continue pretraining to make the experts diverge (the router and the training data break the symmetry).".into(),
    ];
    if a.expert_ff_key {
        notes.push("Exporting to Hugging Face: set norm_topk_prob = true in the MoE config, or the HF model won't match.".into());
    }
    Ok(SurgeryReport {
        output: out.to_owned(),
        bytes: file_len(out)?,
        tensors: written,
        metadata: changed,
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_moments() {
        let mut g = Gauss(7);
        let xs: Vec<f64> = (0..20000).map(|_| f64::from(g.normal())).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
        assert!(mean.abs() < 0.03, "{mean}");
        assert!((var - 1.0).abs() < 0.05, "{var}");
    }

    #[test]
    fn names() {
        assert_eq!(
            expert_name("blk.3.ffn_up.weight").unwrap(),
            "blk.3.ffn_up_exps.weight"
        );
        assert!(expert_name("blk.3.attn_q.weight").is_none());
        assert!(expert_name("output.weight").is_none());
    }
}
