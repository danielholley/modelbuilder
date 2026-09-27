//! Recipes and plan assembly.
//!
//! A plan evaluates requested features (or the whole catalog) against one
//! model: detection, compatibility, the plugin's estimate, and compute priced
//! on each hardware profile.

use mb_features::estimate::{cost, cost_model_assumptions, ComputeEstimate};
use mb_features::hardware::{profile, profiles, HardwareProfile};
use mb_features::{catalog, feature, Compat, Context, Detection, Estimate, FeatureError, Params};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("invalid recipe: {0}")]
    Recipe(String),
    #[error("unknown hardware profile `{0}`; known: {1}")]
    Hardware(String, String),
    #[error(transparent)]
    Feature(#[from] FeatureError),
}

/// A recipe file:
///
/// ```toml
/// [source]
/// path = "models/Ternary-Bonsai-2-27B-PQ2_0.gguf"
///
/// [hardware]
/// profiles = ["1x24GB", "8xH100"]
///
/// [[feature]]
/// id = "mtp"
/// from = "models/Qwen3.8-27B"
///
/// [[feature]]
/// id = "fp4-kv"
/// mode = "qat"
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub source: Source,
    #[serde(default)]
    pub hardware: HardwareSpec,
    #[serde(default, rename = "feature")]
    pub features: Vec<FeatureRequest>,
    #[serde(default)]
    pub export: Option<ExportSpec>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub path: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(deny_unknown_fields)]
pub struct HardwareSpec {
    /// A single profile (shorthand for `profiles = [..]`).
    pub profile: Option<String>,
    #[serde(default)]
    pub profiles: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(deny_unknown_fields)]
pub struct ExportSpec {
    #[serde(default)]
    pub formats: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct FeatureRequest {
    pub id: String,
    /// Every other key in the `[[feature]]` table.
    #[serde(flatten)]
    pub params: Params,
}

impl Recipe {
    pub fn parse(text: &str) -> Result<Self, PlanError> {
        toml::from_str(text).map_err(|e| PlanError::Recipe(e.to_string()))
    }

    pub fn hardware_ids(&self) -> Vec<String> {
        self.hardware
            .profile
            .iter()
            .chain(&self.hardware.profiles)
            .cloned()
            .collect()
    }
}

/// Resolves hardware ids; an empty list means every built-in profile.
pub fn resolve_hardware(ids: &[String]) -> Result<Vec<&'static HardwareProfile>, PlanError> {
    if ids.is_empty() {
        return Ok(profiles().iter().collect());
    }
    ids.iter()
        .map(|id| {
            profile(id).ok_or_else(|| {
                PlanError::Hardware(
                    id.clone(),
                    profiles()
                        .iter()
                        .map(|p| p.id)
                        .collect::<Vec<_>>()
                        .join(", "),
                )
            })
        })
        .collect()
}

/// Parses a CLI feature spec: `id` or `id:key=value,key=value`. Values that
/// parse as JSON (numbers, booleans) keep that type; the rest are strings.
pub fn parse_feature_spec(spec: &str) -> FeatureRequest {
    let (id, rest) = spec.split_once(':').unwrap_or((spec, ""));
    let params = rest
        .split(',')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, "true"));
            let value = serde_json::from_str(v)
                .unwrap_or_else(|_| serde_json::Value::String(v.to_string()));
            (k.trim().to_string(), value)
        })
        .collect();
    FeatureRequest {
        id: id.trim().to_string(),
        params,
    }
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct FeaturePlan {
    pub id: &'static str,
    pub title: &'static str,
    pub summary: &'static str,
    pub params: Params,
    pub detection: Detection,
    pub compat: Compat,
    /// Absent when compatibility checks block the feature.
    pub estimate: Option<Estimate>,
    pub compute: Vec<ComputeEstimate>,
    pub surgery: Vec<String>,
    pub export_notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct Plan {
    pub model: String,
    pub hardware: Vec<HardwareProfile>,
    pub features: Vec<FeaturePlan>,
    pub cost_model_assumptions: Vec<String>,
}

/// Evaluates `requests` (or the whole catalog with default parameters when
/// empty) on `hardware`.
pub fn plan(
    ctx: &Context,
    requests: &[FeatureRequest],
    hardware: &[&HardwareProfile],
) -> Result<Plan, PlanError> {
    let defaults: Vec<FeatureRequest>;
    let requests = if requests.is_empty() {
        defaults = catalog()
            .iter()
            .map(|f| FeatureRequest {
                id: f.id().into(),
                params: Params::new(),
            })
            .collect();
        &defaults
    } else {
        requests
    };
    let inputs = ctx.cost_inputs();
    let mut features = Vec::new();
    for req in requests {
        let f = feature(&req.id)?;
        let compat = f.check_compat(ctx, &req.params)?;
        let (estimate, compute) = if compat.ok() {
            let e = f.estimate(ctx, &req.params)?;
            let c = hardware
                .iter()
                .map(|hw| cost(&inputs, &e.stages, hw))
                .collect();
            (Some(e), c)
        } else {
            (None, Vec::new())
        };
        features.push(FeaturePlan {
            id: f.id(),
            title: f.title(),
            summary: f.summary(),
            params: req.params.clone(),
            detection: f.detect(ctx),
            surgery: if compat.ok() {
                f.surgery_outline(ctx, &req.params)?
            } else {
                Vec::new()
            },
            export_notes: f.export_notes(ctx, &req.params)?,
            compat,
            estimate,
            compute,
        });
    }
    Ok(Plan {
        model: ctx.ir.raw.root.display().to_string(),
        hardware: hardware.iter().map(|h| (*h).clone()).collect(),
        features,
        cost_model_assumptions: cost_model_assumptions(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_specs() {
        let r = parse_feature_spec("kv-share:group=2,note=abc");
        assert_eq!(r.id, "kv-share");
        assert_eq!(r.params["group"], serde_json::json!(2));
        assert_eq!(r.params["note"], serde_json::json!("abc"));
        assert!(parse_feature_spec("fp4-kv").params.is_empty());
    }

    #[test]
    fn recipe_parsing() {
        let r = Recipe::parse(
            r#"
            [source]
            path = "m.gguf"
            [hardware]
            profile = "1x24GB"
            profiles = ["8xH100"]
            [[feature]]
            id = "kv-share"
            group = 2
            [[feature]]
            id = "fp4-kv"
            [export]
            formats = ["gguf"]
            "#,
        )
        .unwrap();
        assert_eq!(r.hardware_ids(), ["1x24GB", "8xH100"]);
        assert_eq!(r.features[0].params["group"], serde_json::json!(2));
        assert!(r.features[1].params.is_empty());
        assert!(Recipe::parse("[source]\npath = 1").is_err());
        assert!(Recipe::parse("[source]\npath = \"x\"\n[bogus]\n").is_err());
    }

    #[test]
    fn hardware_resolution() {
        assert_eq!(resolve_hardware(&[]).unwrap().len(), profiles().len());
        assert!(resolve_hardware(&["nope".into()]).is_err());
        assert_eq!(
            resolve_hardware(&["8xh100".into()]).unwrap()[0].id,
            "8xH100"
        );
    }
}
