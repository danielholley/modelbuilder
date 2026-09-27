use mb_ir::{ConfigView, Metadata, ModelIr};
use serde::Serialize;
use serde_json::Value;

/// What the checkpoint says about where it came from and how it was trained.
#[derive(Clone, Debug, Default, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct Provenance {
    pub name: Option<String>,
    pub license: Option<String>,
    pub base_models: Vec<String>,
    pub tags: Vec<String>,
    pub has_chat_template: bool,
    /// Declared training dtype (`torch_dtype`/`dtype`).
    pub dtype: Option<String>,
    /// Context the model was pre-trained at, if RoPE scaling reveals it.
    pub original_context: Option<u64>,
    /// Context the checkpoint is configured for.
    pub configured_context: Option<u64>,
    /// Best guess at base vs. post-trained, with the reason.
    pub lineage_hint: String,
}

/// Minimal YAML front-matter reader for model cards: `key: value` and `- item` lists.
fn front_matter(card: &str) -> Vec<(String, Vec<String>)> {
    let mut lines = card.lines();
    if lines.next().map(str::trim) != Some("---") {
        return vec![];
    }
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    for line in lines {
        if line.trim() == "---" {
            break;
        }
        let unquote = |s: &str| s.trim().trim_matches(|c| c == '"' || c == '\'').to_string();
        if let Some(item) = line.trim_start().strip_prefix("- ") {
            if let Some((_, v)) = out.last_mut() {
                v.push(unquote(item));
            }
        } else if let Some((k, v)) = line.split_once(':') {
            if line.starts_with(char::is_whitespace) {
                continue; // nested mappings aren't needed here
            }
            let v = v.trim();
            let vals = if v.is_empty() {
                vec![]
            } else if let Some(list) = v.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
                list.split(',')
                    .map(unquote)
                    .filter(|s| !s.is_empty())
                    .collect()
            } else {
                vec![unquote(v)]
            };
            out.push((k.trim().to_string(), vals));
        }
    }
    out
}

impl Provenance {
    pub fn of(ir: &ModelIr) -> Self {
        let cfg = ConfigView::new(&ir.raw.metadata);
        let aux = &ir.raw.aux;
        let mut p = Provenance {
            configured_context: ir.max_positions,
            ..Default::default()
        };

        match &ir.raw.metadata {
            Metadata::Hf { config, .. } => {
                p.dtype = ["torch_dtype", "dtype"].iter().find_map(|k| {
                    config
                        .get(k)
                        .or_else(|| cfg.hf_value(k))?
                        .as_str()
                        .map(str::to_owned)
                });
                p.name = config
                    .get("_name_or_path")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
            }
            Metadata::Gguf { .. } => {
                let s = |k: &str| cfg.gguf_raw(k).and_then(|v| v.as_str()).map(str::to_owned);
                p.name = s("general.name");
                p.license = s("general.license");
                let n = cfg
                    .gguf_raw("general.base_model.count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                p.base_models = (0..n)
                    .filter_map(|i| {
                        s(&format!("general.base_model.{i}.repo_url"))
                            .or_else(|| s(&format!("general.base_model.{i}.name")))
                    })
                    .collect();
                if let Some(tags) = cfg.gguf_raw("general.tags").and_then(|v| v.as_array()) {
                    p.tags = tags
                        .iter()
                        .filter_map(|t| t.as_str().map(str::to_owned))
                        .collect();
                }
            }
        }

        if let Some(card) = &aux.model_card {
            for (k, v) in front_matter(card) {
                match k.as_str() {
                    "license" if p.license.is_none() => p.license = v.into_iter().next(),
                    "base_model" => p.base_models.extend(v),
                    "tags" => p.tags.extend(v),
                    "model_name" if p.name.is_none() => p.name = v.into_iter().next(),
                    _ => {}
                }
            }
        }

        p.has_chat_template = aux.chat_template.is_some()
            || aux
                .tokenizer_config
                .as_ref()
                .and_then(|t| t.get("chat_template"))
                .is_some_and(|v| !v.is_null());

        p.original_context = ir.rope.scaling.as_ref().and_then(|s| {
            s.get("original_max_position_embeddings")
                .or_else(|| s.get("original_context_length"))
                .and_then(Value::as_u64)
        });

        let name_hint = p.name.as_deref().unwrap_or_default().to_ascii_lowercase();
        p.lineage_hint = if p.has_chat_template
            && ["instruct", "chat", "-it"]
                .iter()
                .any(|s| name_hint.contains(s))
        {
            "post-trained (chat template present, name says instruct/chat)".into()
        } else if p.has_chat_template {
            "probably post-trained (chat template present); base models sometimes ship one too"
                .into()
        } else if name_hint.contains("base") {
            "base model (name says base, no chat template)".into()
        } else {
            "probably base (no chat template found)".into()
        };
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_front_matter() {
        let card = "---\nlicense: mit\nbase_model:\n- org/a\n- \"org/b\"\ntags: [x, y]\n---\n# hi";
        let fm = front_matter(card);
        assert_eq!(fm[0], ("license".into(), vec!["mit".into()]));
        assert_eq!(
            fm[1],
            ("base_model".into(), vec!["org/a".into(), "org/b".into()])
        );
        assert_eq!(fm[2], ("tags".into(), vec!["x".into(), "y".into()]));
        assert!(front_matter("# no front matter").is_empty());
    }
}
