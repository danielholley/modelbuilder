//! Recipes and plan assembly.
//!
//! A plan evaluates requested features (or the whole catalog) against one
//! model: detection, compatibility, the plugin's estimate, and compute priced
//! on each hardware profile.

use mb_features::estimate::{cost, cost_model_assumptions, ComputeEstimate};
use mb_features::hardware::{profile, profiles, HardwareProfile};
use mb_features::{
    catalog, feature, Compat, Context, Detection, Estimate, FeatureError, Params, Stage, TrunkUse,
};
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
/// path = "models/target.gguf"
///
/// [hardware]
/// profiles = ["1x24GB", "8xH100"]
///
/// [[feature]]
/// id = "mtp"
/// from = "models/base-model"
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

/// One stage in the order the whole plan should run.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct ScheduledStage {
    /// 1-based position.
    pub order: usize,
    pub feature: String,
    pub stage: String,
    pub trunk: TrunkUse,
    /// Why it runs at this position.
    pub reason: String,
}

/// All compatible features' stages in dependency order, with totals per profile.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct Schedule {
    pub stages: Vec<ScheduledStage>,
    /// The whole schedule priced per hardware profile (the peak is the largest stage's).
    pub totals: Vec<ComputeEstimate>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct Plan {
    pub model: String,
    pub hardware: Vec<HardwareProfile>,
    pub features: Vec<FeaturePlan>,
    pub schedule: Schedule,
    /// The commands that carry the plan out, in schedule order.
    pub pipeline: Pipeline,
    pub cost_model_assumptions: Vec<String>,
}

/// One command of the [`Pipeline`].
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct PipelineStep {
    /// 1-based position.
    pub order: usize,
    pub feature: String,
    /// The schedule stage this step trains, if any.
    pub stage: Option<String>,
    pub trunk: TrunkUse,
    /// `None`: modelbuilder has no runner for this step yet. `<...>` marks
    /// inputs to fill in.
    pub command: Option<String>,
    /// The model the step starts from.
    pub model: String,
    pub note: String,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct Pipeline {
    pub steps: Vec<PipelineStep>,
    /// The GGUF the last step leaves (the input model when nothing changes it).
    pub final_model: String,
    pub notes: Vec<String>,
}

fn file_stem(p: &str) -> String {
    std::path::Path::new(p)
        .file_stem()
        .map_or_else(|| p.to_string(), |s| s.to_string_lossy().into_owned())
}

/// Orders the compatible features' run steps restructure → adapt → frozen
/// (stable, like [`schedule`]) and chains the GGUFs they produce: each step
/// starts from the latest GGUF.
pub fn pipeline(ctx: &Context, features: &[FeaturePlan]) -> Result<Pipeline, PlanError> {
    let mut steps: Vec<(&str, mb_features::RunStep)> = Vec::new();
    for f in features.iter().filter(|f| f.compat.ok()) {
        for st in feature(f.id)?.pipeline(ctx, &f.params)? {
            steps.push((f.id, st));
        }
    }
    steps.sort_by_key(|(_, s)| s.trunk);
    let source = ctx.ir.raw.root.display().to_string();
    let mut model = source.clone();
    let mut notes = Vec::new();
    let mut hf_only: Option<String> = None;
    let mut changed = false;
    let mut out = Vec::new();
    for (i, (fid, st)) in steps.into_iter().enumerate() {
        let run = format!("runs/{fid}");
        let fill = |t: &str| {
            t.replace("{source}", &source)
                .replace("{model}", &model)
                .replace("{stem}", &file_stem(&model))
                .replace("{run}", &run)
        };
        let mut note = fill(&st.note);
        if let Some(h) = &hf_only {
            if st.command.as_deref().is_some_and(|c| c.contains("{model}")) {
                note = format!(
                    "{note} (starts from {model}: the earlier HF-only result {h} has no GGUF form)"
                );
            }
        }
        if st.trunk == TrunkUse::Frozen && changed && st.command.is_some() {
            note = format!("{note} Uses the trunk after every change before it.");
        }
        let command = st.command.as_deref().map(fill);
        let next = match &st.produces {
            mb_features::Produces::Gguf(p) => Some(fill(p)),
            mb_features::Produces::Hf(p) => {
                hf_only = Some(fill(p));
                None
            }
            mb_features::Produces::Nothing => None,
        };
        out.push(PipelineStep {
            order: i + 1,
            feature: fid.to_string(),
            stage: st.stage.clone(),
            trunk: st.trunk,
            command,
            model: model.clone(),
            note,
        });
        if let Some(n) = next {
            model = n;
            changed = true;
        }
        if hf_only.is_some() && st.trunk != TrunkUse::Frozen {
            changed = true;
        }
    }
    if out.iter().any(|s| {
        s.command.as_deref().is_some_and(|c| {
            c.contains("trunk-distill") && !c.contains("--teacher") && !c.contains("--loss ce")
        })
    }) && changed
    {
        notes.push("trunk-distill steps without --teacher distill from the model they start from with the change switched off (e.g. a plain KV cache), so they isolate their own change; heal-type steps pass --teacher with the original.".into());
    }
    if out.iter().any(|s| {
        s.command
            .as_deref()
            .is_some_and(|c| c.contains("--teacher"))
    }) {
        notes.push(
            "Steps with --teacher hold a second model in memory: the original they distill from."
                .into(),
        );
    }
    let missing: Vec<String> = out
        .iter()
        .filter(|s| s.command.is_none())
        .map(|s| format!("{}/{}", s.feature, s.stage.as_deref().unwrap_or("?")))
        .collect();
    if !missing.is_empty() {
        notes.push(format!("No runner yet for: {}.", missing.join(", ")));
    }
    Ok(Pipeline {
        steps: out,
        final_model: model,
        notes,
    })
}

fn reason(t: TrunkUse) -> &'static str {
    match t {
        TrunkUse::Restructure => "changes the trunk's structure, so it runs before anything trained against the trunk",
        TrunkUse::Adapt => "retrains trunk tensors in place, after structural changes (so it sees the final structure) and before frozen-trunk stages",
        TrunkUse::Frozen => "learns from the trunk's outputs, so it runs last: any later change to the trunk would invalidate it",
    }
}

/// Orders stages restructure → adapt → frozen. The sort is stable, so each
/// feature keeps its own stage order and features keep the request order.
pub fn schedule(
    features: &[FeaturePlan],
    inputs: &mb_features::CostInputs,
    hardware: &[&HardwareProfile],
) -> Schedule {
    let mut stages: Vec<(&FeaturePlan, &Stage)> = features
        .iter()
        .filter_map(|f| f.estimate.as_ref().map(|e| (f, e)))
        .flat_map(|(f, e)| e.stages.iter().map(move |s| (f, s)))
        .collect();
    stages.sort_by_key(|(_, s)| s.trunk);
    let ordered: Vec<Stage> = stages.iter().map(|(_, s)| (*s).clone()).collect();
    let mut notes = Vec::new();
    let has = |t: TrunkUse| stages.iter().any(|(_, s)| s.trunk == t);
    if has(TrunkUse::Frozen) && (has(TrunkUse::Restructure) || has(TrunkUse::Adapt)) {
        notes.push(
            "Frozen-trunk stages (draft heads) are trained against the final trunk: extract their features only after the trunk-changing stages finish.".into(),
        );
    }
    let adapters: std::collections::BTreeSet<&str> = stages
        .iter()
        .filter(|(_, s)| s.trunk == TrunkUse::Adapt)
        .map(|(f, _)| f.id)
        .collect();
    if adapters.len() > 1 {
        notes.push(format!(
            "{} each retrain trunk tensors: when they touch the same tensors, one combined run with all their losses is cheaper than running them in sequence.",
            adapters.into_iter().collect::<Vec<_>>().join(" and ")
        ));
    }
    if stages.is_empty() {
        notes.push("No compatible feature needs training.".into());
    }
    Schedule {
        stages: stages
            .iter()
            .enumerate()
            .map(|(i, (f, s))| ScheduledStage {
                order: i + 1,
                feature: f.id.to_string(),
                stage: s.name.clone(),
                trunk: s.trunk,
                reason: reason(s.trunk).into(),
            })
            .collect(),
        totals: hardware
            .iter()
            .map(|hw| cost(inputs, &ordered, hw))
            .collect(),
        notes,
    }
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
        schedule: schedule(&features, &inputs, hardware),
        pipeline: pipeline(ctx, &features)?,
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
