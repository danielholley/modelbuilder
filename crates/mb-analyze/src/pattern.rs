use mb_ir::{AttentionKind, FfnSpec, Layer, Mixer};
use serde::Serialize;

/// Short label for a layer's mixer and FFN, e.g. `gqa(24q/4kv,d256,gated)+dense`.
pub fn layer_label(layer: &Layer) -> String {
    let mixer = match &layer.mixer {
        Mixer::Attention(a) => {
            let kind = match a.kind {
                AttentionKind::Mha => "mha",
                AttentionKind::Gqa => "gqa",
                AttentionKind::Mqa => "mqa",
                AttentionKind::Mla => "mla",
            };
            let mut parts = Vec::new();
            if a.kind == AttentionKind::Mla {
                if let Some(r) = a.mla.as_ref().and_then(|m| m.kv_lora_rank) {
                    parts.push(format!("kv_rank{r}"));
                }
                if let Some(h) = a.num_heads {
                    parts.push(format!("{h}h"));
                }
            } else {
                if let (Some(q), Some(kv)) = (a.num_heads, a.num_kv_heads) {
                    parts.push(format!("{q}q/{kv}kv"));
                }
                if let Some(d) = a.head_dim {
                    parts.push(format!("d{d}"));
                }
            }
            if a.output_gate {
                parts.push("gated".into());
            }
            if let Some(w) = a.sliding_window {
                parts.push(format!("swa{w}"));
            }
            if parts.is_empty() {
                kind.to_string()
            } else {
                format!("{kind}({})", parts.join(","))
            }
        }
        Mixer::LinearAttention(l) => l.variant.clone(),
        Mixer::Unknown => "unknown".into(),
    };
    let ffn = match &layer.ffn {
        FfnSpec::Dense { .. } => "dense".to_string(),
        FfnSpec::Moe(m) => {
            let mut s = format!(
                "moe({}",
                m.num_experts.map_or("?".into(), |n| n.to_string())
            );
            if let Some(k) = m.experts_per_token {
                s += &format!(" top{k}");
            }
            if m.num_shared_experts > 0 {
                s += &format!(" +{} shared", m.num_shared_experts);
            }
            s + ")"
        }
        FfnSpec::None => "no-ffn".into(),
    };
    format!("{mixer}+{ffn}")
}

/// Run-length encoded layer layout: `repeats × [count × label, ...]`, plus a prefix
/// of leading layers that don't fit the repeating block (e.g. DeepSeek's dense first layers).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LayerPattern {
    pub prefix: Vec<(usize, String)>,
    pub repeats: usize,
    pub block: Vec<(usize, String)>,
}

fn rle(labels: &[String]) -> Vec<(usize, String)> {
    let mut out: Vec<(usize, String)> = Vec::new();
    for l in labels {
        match out.last_mut() {
            Some((n, last)) if last == l => *n += 1,
            _ => out.push((1, l.clone())),
        }
    }
    out
}

/// The smallest period that tiles `labels` exactly.
fn period(labels: &[String]) -> usize {
    let n = labels.len();
    (1..=n)
        .find(|&p| n % p == 0 && (p..n).all(|i| labels[i] == labels[i - p]))
        .unwrap_or(n)
}

impl LayerPattern {
    pub fn of(layers: &[Layer]) -> Self {
        let labels: Vec<String> = layers.iter().map(layer_label).collect();
        // Choose the split (prefix, periodic tail) that gives the shortest description.
        let (mut best, mut best_cost) = (None, usize::MAX);
        for skip in 0..labels.len().max(1) {
            let tail = &labels[skip.min(labels.len())..];
            let p = period(tail);
            let cost = rle(&labels[..skip]).len() + rle(&tail[..p]).len();
            if cost < best_cost {
                best_cost = cost;
                best = Some((skip, p));
            }
        }
        let Some((skip, p)) = best.filter(|_| !labels.is_empty()) else {
            return Self {
                prefix: vec![],
                repeats: 0,
                block: vec![],
            };
        };
        let tail = &labels[skip..];
        Self {
            prefix: rle(&labels[..skip]),
            repeats: tail.len() / p,
            block: rle(&tail[..p]),
        }
    }
}

impl std::fmt::Display for LayerPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let list = |v: &[(usize, String)]| {
            v.iter()
                .map(|(n, l)| format!("{n} × {l}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        if !self.prefix.is_empty() {
            write!(f, "[{}] then ", list(&self.prefix))?;
        }
        match self.block.as_slice() {
            [(1, label)] => write!(f, "{} × {label}", self.repeats),
            block => write!(f, "{} × [{}]", self.repeats, list(block)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn finds_period() {
        assert_eq!(period(&s(&["a", "a", "a", "b", "a", "a", "a", "b"])), 4);
        assert_eq!(period(&s(&["a", "b", "c"])), 3);
        assert_eq!(period(&s(&["a", "a"])), 1);
    }
}
