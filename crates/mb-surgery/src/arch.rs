//! Architecture adapters: how a llama.cpp GGUF architecture's tensors map to
//! the Hugging Face checkpoint it was converted from, and back.
//!
//! Each adapter is data, not code: a tensor-name table plus per-tensor
//! transforms that invert what llama.cpp's `convert_hf_to_gguf.py` (and the
//! PrismML fork's `conversion/`) do to that architecture:
//!
//! - `llama` (also Mistral-style checkpoints): Q/K rows are permuted for
//!   llama.cpp's RoPE layout (`LlamaModel.permute`).
//! - `qwen2` (Q/K/V biases), `qwen3` (Q/K RMSNorm): names only.
//! - `qwen35` (Qwen3.5-family hybrids): zero-centered RMSNorm weights are
//!   stored `+1`; gated-DeltaNet tensors are renamed, `A_log` is stored as
//!   `-exp(A_log)`, `conv1d` is squeezed, and V heads are reordered from
//!   grouped to tiled order when there are fewer key heads than value heads
//!   (`Qwen3NextModel.modify_tensors`, `_LinearAttentionVReorderBase`).
//!
//! Adding an architecture means adding an [`Adapter`] and, if it transforms
//! tensors in new ways, a case in [`to_hf`]; the round-trip tests cover each.

use serde_json::Value;

use crate::SurgeryError;

#[derive(Clone, Copy, Debug)]
pub struct Adapter {
    /// llama.cpp `general.architecture` values this adapter handles.
    pub gguf_archs: &'static [&'static str],
    /// `architectures[0]` and `model_type` for an exported HF config.
    pub hf_architecture: &'static str,
    pub hf_model_type: &'static str,
    /// Zero-centered RMSNorm: HF applies `x·(1+w)`, the GGUF stores `w+1`.
    pub norm_offset: bool,
    /// Q/K rows permuted for llama.cpp's RoPE (`LlamaModel.permute`).
    pub rope_permute: bool,
    /// The post-attention norm's GGUF name: `ffn_norm` (llama style) or
    /// `post_attention_norm` (qwen35 style).
    pub post_attn_norm: &'static str,
    /// Has gated-DeltaNet linear-attention layers (qwen35 family).
    pub gated_deltanet: bool,
    /// Supports an MTP ("nextn") block in llama.cpp's layout.
    pub nextn: bool,
}

pub const ADAPTERS: &[Adapter] = &[
    Adapter {
        gguf_archs: &["llama"],
        hf_architecture: "LlamaForCausalLM",
        hf_model_type: "llama",
        norm_offset: false,
        rope_permute: true,
        post_attn_norm: "ffn_norm",
        gated_deltanet: false,
        nextn: false,
    },
    Adapter {
        gguf_archs: &["qwen2"],
        hf_architecture: "Qwen2ForCausalLM",
        hf_model_type: "qwen2",
        norm_offset: false,
        rope_permute: false,
        post_attn_norm: "ffn_norm",
        gated_deltanet: false,
        nextn: false,
    },
    Adapter {
        gguf_archs: &["qwen3"],
        hf_architecture: "Qwen3ForCausalLM",
        hf_model_type: "qwen3",
        norm_offset: false,
        rope_permute: false,
        post_attn_norm: "ffn_norm",
        gated_deltanet: false,
        nextn: false,
    },
    Adapter {
        gguf_archs: &["qwen35"],
        hf_architecture: "Qwen3_5ForCausalLM",
        hf_model_type: "qwen3_5_text",
        norm_offset: true,
        rope_permute: false,
        post_attn_norm: "post_attention_norm",
        gated_deltanet: true,
        nextn: true,
    },
];

pub fn adapter(arch: &str) -> Result<&'static Adapter, SurgeryError> {
    ADAPTERS
        .iter()
        .find(|a| a.gguf_archs.contains(&arch))
        .ok_or_else(|| {
            let known: Vec<&str> = ADAPTERS
                .iter()
                .flat_map(|a| a.gguf_archs.iter().copied())
                .collect();
            SurgeryError::Unsupported(format!(
                "no architecture adapter for `{arch}` (have: {}); see mb-surgery/src/arch.rs",
                known.join(", ")
            ))
        })
}

/// Shape parameters from the HF config that the transforms need.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dims {
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub hidden: usize,
    /// Gated-DeltaNet heads: (key heads, value heads, key dim, value dim).
    pub gdn: Option<(usize, usize, usize, usize)>,
}

fn key(t: &Value, k: &str) -> Option<usize> {
    t.get(k).and_then(Value::as_u64).map(|v| v as usize)
}

impl Dims {
    /// Reads a (text) config: `text_config` if nested, else the top level.
    pub fn from_config(cfg: &Value, a: &Adapter) -> Result<Self, SurgeryError> {
        let t = cfg.get("text_config").unwrap_or(cfg);
        let need = |k: &str| {
            key(t, k).ok_or_else(|| SurgeryError::Incompatible(format!("config has no `{k}`")))
        };
        let n_head = need("num_attention_heads")?;
        let hidden = need("hidden_size")?;
        let gdn = if a.gated_deltanet {
            Some((
                need("linear_num_key_heads")?,
                need("linear_num_value_heads")?,
                need("linear_key_head_dim")?,
                need("linear_value_head_dim")?,
            ))
        } else {
            None
        };
        Ok(Self {
            n_head,
            n_kv_head: key(t, "num_key_value_heads").unwrap_or(n_head),
            head_dim: key(t, "head_dim").unwrap_or(hidden / n_head.max(1)),
            layers: need("num_hidden_layers")?,
            hidden,
            gdn,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    Copy,
    /// Zero-centered norm: GGUF `w+1` → HF `w`.
    MinusOne,
    /// GGUF `-exp(A_log)` → HF `A_log`.
    LogNeg,
}

impl Elem {
    /// GGUF value → HF value.
    pub fn to_hf(self, v: f32) -> Result<f32, String> {
        match self {
            Elem::Copy => Ok(v),
            Elem::MinusOne => Ok(v - 1.0),
            Elem::LogNeg if v < 0.0 => Ok((-v).ln()),
            Elem::LogNeg => Err(format!("expected -exp(A_log) < 0, found {v}")),
        }
    }

    /// HF value → GGUF value.
    pub fn to_gguf(self, v: f32) -> f32 {
        match self {
            Elem::Copy => v,
            Elem::MinusOne => v + 1.0,
            Elem::LogNeg => -v.exp(),
        }
    }
}

/// How one GGUF tensor becomes one HF tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Mapping {
    pub hf_name: String,
    /// Decoder layer, `None` for embedding / final norm / LM head.
    pub layer: Option<u32>,
    pub hf_shape: Vec<u64>,
    /// For each HF row, the GGUF row it comes from (`None`: identity).
    pub rows: Option<Vec<usize>>,
    /// For each HF column, the GGUF column it comes from (`None`: identity).
    pub cols: Option<Vec<usize>>,
    pub elem: Elem,
    pub note: &'static str,
}

impl Mapping {
    /// The inverse gathers, for writing an HF tensor back in GGUF order.
    pub fn inverse(perm: &[usize]) -> Vec<usize> {
        let mut inv = vec![0; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        inv
    }
}

/// For each HF (grouped-order) index along a V-head axis, the GGUF (tiled-order)
/// index it comes from. The converter's `_reorder_v_heads` maps grouped
/// `[nk, per_k, hd]` to tiled `[per_k, nk, hd]`; this is its inverse gather.
pub fn grouped_from_tiled(nk: usize, per_k: usize, hd: usize) -> Vec<usize> {
    (0..nk * per_k * hd)
        .map(|h| {
            let (head, d) = (h / hd, h % hd);
            let (k, j) = (head / per_k, head % per_k);
            (j * nk + k) * hd + d
        })
        .collect()
}

/// For each HF Q/K row, the GGUF row it comes from, undoing
/// `LlamaModel.permute` (`reshape(n, 2, hd/2).swapaxes(1, 2)`).
pub fn hf_from_llama_rope(n_head: usize, head_dim: usize) -> Vec<usize> {
    let half = head_dim / 2;
    (0..n_head * head_dim)
        .map(|r| {
            let (h, rest) = (r / head_dim, r % head_dim);
            let (which, j) = (rest / half, rest % half);
            h * head_dim + 2 * j + which
        })
        .collect()
}

/// Maps a GGUF tensor (name, row-major shape) to its HF counterpart.
/// `Ok(None)` for tensors that aren't part of the trunk (an MTP block).
/// `grouped_out`: `ssm_out`'s columns are already in grouped order
/// (PrismML's `prism.hadamard.gdn_v_grouped`).
pub fn to_hf(
    a: &Adapter,
    d: &Dims,
    name: &str,
    shape: &[u64],
    grouped_out: bool,
) -> Result<Option<Mapping>, SurgeryError> {
    let norm = if a.norm_offset {
        Elem::MinusOne
    } else {
        Elem::Copy
    };
    let global = |hf: &str, elem| Mapping {
        hf_name: hf.into(),
        layer: None,
        hf_shape: shape.to_vec(),
        rows: None,
        cols: None,
        elem,
        note: "",
    };
    match name {
        "token_embd.weight" => return Ok(Some(global("model.embed_tokens.weight", Elem::Copy))),
        "output_norm.weight" => return Ok(Some(global("model.norm.weight", norm))),
        "output.weight" => return Ok(Some(global("lm_head.weight", Elem::Copy))),
        _ => {}
    }
    let unknown =
        || SurgeryError::Unsupported(format!("{}: no HF name for tensor {name}", a.gguf_archs[0]));
    let rest = name.strip_prefix("blk.").ok_or_else(unknown)?;
    let (idx, suffix) = rest.split_once('.').ok_or_else(unknown)?;
    let layer: u32 = idx.parse().map_err(|_| unknown())?;
    if suffix.starts_with("nextn.") || layer as usize >= d.layers {
        return Ok(None); // an MTP block after the trunk
    }
    let mut m = Mapping {
        hf_name: String::new(),
        layer: Some(layer),
        hf_shape: shape.to_vec(),
        rows: None,
        cols: None,
        elem: Elem::Copy,
        note: "",
    };
    let rope = |n: usize| a.rope_permute.then(|| hf_from_llama_rope(n, d.head_dim));
    let (hf, elem): (String, Elem) = match suffix {
        "attn_norm.weight" => ("input_layernorm.weight".into(), norm),
        s if s == format!("{}.weight", a.post_attn_norm) => {
            ("post_attention_layernorm.weight".into(), norm)
        }
        "ffn_gate.weight" => ("mlp.gate_proj.weight".into(), Elem::Copy),
        "ffn_up.weight" => ("mlp.up_proj.weight".into(), Elem::Copy),
        "ffn_down.weight" => ("mlp.down_proj.weight".into(), Elem::Copy),
        "attn_q.weight" | "attn_q.bias" => {
            m.rows = rope(d.n_head);
            (format!("self_attn.q_proj.{}", &suffix[7..]), Elem::Copy)
        }
        "attn_k.weight" | "attn_k.bias" => {
            m.rows = rope(d.n_kv_head);
            (format!("self_attn.k_proj.{}", &suffix[7..]), Elem::Copy)
        }
        "attn_v.weight" | "attn_v.bias" => {
            (format!("self_attn.v_proj.{}", &suffix[7..]), Elem::Copy)
        }
        "attn_output.weight" => ("self_attn.o_proj.weight".into(), Elem::Copy),
        "attn_q_norm.weight" => ("self_attn.q_norm.weight".into(), norm),
        "attn_k_norm.weight" => ("self_attn.k_norm.weight".into(), norm),
        s if a.gated_deltanet => {
            let (nk, nv, hk, hv) = d.gdn.expect("gated_deltanet adapters read gdn dims");
            let per_k = nv.checked_div(nk).unwrap_or(0);
            let v = |hd: usize| {
                (nk > 0 && nv > 0 && nk != nv).then(|| grouped_from_tiled(nk, per_k, hd))
            };
            // [q (nk·hk), k (nk·hk), v (nv·hv)]: only the V part is reordered.
            let qkv = || {
                v(hv).map(|p| {
                    (0..2 * nk * hk)
                        .chain(p.into_iter().map(|r| r + 2 * nk * hk))
                        .collect()
                })
            };
            match s {
                "attn_qkv.weight" => {
                    m.rows = qkv();
                    ("linear_attn.in_proj_qkv.weight".into(), Elem::Copy)
                }
                "attn_gate.weight" => {
                    m.rows = v(hv);
                    ("linear_attn.in_proj_z.weight".into(), Elem::Copy)
                }
                "ssm_beta.weight" => {
                    m.rows = v(1);
                    ("linear_attn.in_proj_b.weight".into(), Elem::Copy)
                }
                "ssm_alpha.weight" => {
                    m.rows = v(1);
                    ("linear_attn.in_proj_a.weight".into(), Elem::Copy)
                }
                "ssm_a" => {
                    m.cols = v(1);
                    ("linear_attn.A_log".into(), Elem::LogNeg)
                }
                "ssm_dt.bias" => {
                    m.cols = v(1);
                    ("linear_attn.dt_bias".into(), Elem::Copy)
                }
                "ssm_conv1d.weight" => {
                    m.rows = qkv();
                    if let [c, k] = shape[..] {
                        m.hf_shape = vec![c, 1, k];
                    }
                    ("linear_attn.conv1d.weight".into(), Elem::Copy)
                }
                "ssm_norm.weight" => ("linear_attn.norm.weight".into(), Elem::Copy), // gated norm: no offset
                "ssm_out.weight" => {
                    if grouped_out {
                        m.note = "columns already grouped (gdn_v_grouped)";
                    } else {
                        m.cols = v(hv);
                    }
                    ("linear_attn.out_proj.weight".into(), Elem::Copy)
                }
                _ => return Err(unknown()),
            }
        }
        _ => return Err(unknown()),
    };
    let rows = shape.first().copied().unwrap_or(1) as usize;
    let cols = shape.last().copied().unwrap_or(1) as usize;
    if m.rows.as_ref().is_some_and(|r| r.len() != rows)
        || m.cols.as_ref().is_some_and(|c| c.len() != cols)
    {
        return Err(SurgeryError::Incompatible(format!(
            "{name} {shape:?} doesn't match the head layout in the config"
        )));
    }
    m.hf_name = format!("model.layers.{layer}.{hf}");
    m.elem = elem;
    Ok(Some(m))
}

/// The MTP ("nextn") block's tensors in llama.cpp's layout, for adapters with
/// `nextn`: `(HF suffix after "mtp.", GGUF suffix after "blk.{n}.", is_norm)`.
/// The block's decoder layer uses the trunk's own tensor names.
pub fn nextn_map(a: &Adapter) -> Vec<(String, String, bool)> {
    let mut v: Vec<(String, String, bool)> = [
        ("fc.weight", "nextn.eh_proj.weight", false),
        ("pre_fc_norm_embedding.weight", "nextn.enorm.weight", true),
        ("pre_fc_norm_hidden.weight", "nextn.hnorm.weight", true),
        ("norm.weight", "nextn.shared_head_norm.weight", true),
    ]
    .iter()
    .map(|(h, g, n)| (h.to_string(), g.to_string(), *n))
    .collect();
    let layer = [
        (
            "input_layernorm.weight",
            "attn_norm.weight".to_string(),
            true,
        ),
        (
            "post_attention_layernorm.weight",
            format!("{}.weight", a.post_attn_norm),
            true,
        ),
        ("self_attn.q_proj.weight", "attn_q.weight".into(), false),
        ("self_attn.k_proj.weight", "attn_k.weight".into(), false),
        ("self_attn.v_proj.weight", "attn_v.weight".into(), false),
        (
            "self_attn.o_proj.weight",
            "attn_output.weight".into(),
            false,
        ),
        ("self_attn.q_norm.weight", "attn_q_norm.weight".into(), true),
        ("self_attn.k_norm.weight", "attn_k_norm.weight".into(), true),
        ("mlp.gate_proj.weight", "ffn_gate.weight".into(), false),
        ("mlp.up_proj.weight", "ffn_up.weight".into(), false),
        ("mlp.down_proj.weight", "ffn_down.weight".into(), false),
    ];
    v.extend(
        layer
            .into_iter()
            .map(|(h, g, n)| (format!("layers.0.{h}"), g, n)),
    );
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The converter's `_reorder_v_heads` (grouped → tiled), as a gather.
    fn tiled_from_grouped(nk: usize, per_k: usize, hd: usize) -> Vec<usize> {
        (0..nk * per_k * hd)
            .map(|g| {
                let (head, d) = (g / hd, g % hd);
                let (j, k) = (head / nk, head % nk);
                (k * per_k + j) * hd + d
            })
            .collect()
    }

    /// `LlamaModel.permute` as a gather: GGUF row g takes HF row src[g].
    fn llama_permute(n: usize, hd: usize) -> Vec<usize> {
        // reshape(n, 2, hd/2).swapaxes(1, 2) → index (h, j, which) holds HF (h, which, j)
        let half = hd / 2;
        (0..n * hd)
            .map(|g| {
                let (h, rest) = (g / hd, g % hd);
                let (j, which) = (rest / 2, rest % 2);
                h * hd + which * half + j
            })
            .collect()
    }

    fn roundtrip(forward: &[usize], back: &[usize]) {
        let hf: Vec<usize> = (0..forward.len()).collect();
        let gguf: Vec<usize> = forward.iter().map(|&i| hf[i]).collect();
        let again: Vec<usize> = back.iter().map(|&i| gguf[i]).collect();
        assert_eq!(again, hf);
    }

    #[test]
    fn v_head_reorder_inverts_the_converter() {
        for (nk, per_k, hd) in [(16, 3, 128), (2, 2, 32), (4, 1, 8), (3, 5, 1)] {
            roundtrip(
                &tiled_from_grouped(nk, per_k, hd),
                &grouped_from_tiled(nk, per_k, hd),
            );
        }
        // The converter's own example: grouped [G0v0, G0v1, G1v0, G1v1] → tiled [G0v0, G1v0, G0v1, G1v1].
        assert_eq!(tiled_from_grouped(2, 2, 1), [0, 2, 1, 3]);
    }

    #[test]
    fn llama_rope_permute_inverts_the_converter() {
        for (n, hd) in [(32, 128), (8, 64), (2, 4)] {
            roundtrip(&llama_permute(n, hd), &hf_from_llama_rope(n, hd));
        }
        // One head of 4: HF [a0 a1 | b0 b1] (halves) → GGUF [a0 b0 a1 b1] (pairs).
        assert_eq!(llama_permute(1, 4), [0, 2, 1, 3]);
    }

    #[test]
    fn inverse_perm() {
        let p = vec![2, 0, 3, 1];
        let inv = Mapping::inverse(&p);
        for (i, &x) in p.iter().enumerate() {
            assert_eq!(inv[x], i);
        }
    }

    #[test]
    fn elementwise_transforms_round_trip() {
        for e in [Elem::Copy, Elem::MinusOne, Elem::LogNeg] {
            for v in [-2.5f32, 0.3, 1.7] {
                let back = e.to_hf(e.to_gguf(v)).unwrap();
                assert!((back - v).abs() < 1e-5, "{e:?} {v} → {back}");
            }
        }
        assert!(Elem::LogNeg.to_hf(0.5).is_err());
    }

    #[test]
    fn names_per_architecture() {
        let llama = adapter("llama").unwrap();
        let d = Dims {
            n_head: 4,
            n_kv_head: 2,
            head_dim: 8,
            layers: 2,
            hidden: 32,
            gdn: None,
        };
        let m = to_hf(llama, &d, "blk.1.attn_k.weight", &[16, 32], false)
            .unwrap()
            .unwrap();
        assert_eq!(m.hf_name, "model.layers.1.self_attn.k_proj.weight");
        assert!(m.rows.is_some(), "llama permutes K");
        let n = to_hf(llama, &d, "blk.0.ffn_norm.weight", &[32], false)
            .unwrap()
            .unwrap();
        assert_eq!(
            (n.hf_name.as_str(), n.elem),
            ("model.layers.0.post_attention_layernorm.weight", Elem::Copy)
        );
        let q3 = adapter("qwen3").unwrap();
        let k = to_hf(q3, &d, "blk.0.attn_k.weight", &[16, 32], false)
            .unwrap()
            .unwrap();
        assert!(k.rows.is_none(), "qwen3 doesn't permute");
        assert!(
            to_hf(q3, &d, "blk.0.ssm_a", &[4], false).is_err(),
            "no DeltaNet tensors in qwen3"
        );
        let q35 = adapter("qwen35").unwrap();
        let n = to_hf(q35, &d, "output_norm.weight", &[32], false)
            .unwrap()
            .unwrap();
        assert_eq!(n.elem, Elem::MinusOne);
        assert!(adapter("gpt2").is_err());
    }
}
