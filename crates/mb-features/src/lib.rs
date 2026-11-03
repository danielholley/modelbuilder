//! Feature plugins.
//!
//! Each feature answers, for a given model: is it already there
//! ([`Feature::detect`]), can it be applied ([`Feature::check_compat`]), what
//! does it cost and risk ([`Feature::estimate`], priced per hardware profile by
//! [`estimate::cost`]), what changes in the checkpoint
//! ([`Feature::surgery_outline`]), and what it means for runtimes
//! ([`Feature::export_notes`]).
//!
//! Plugins that implement a technique from a paper or codebase cite it in
//! [`Estimate::references`] and follow what the source specifies.

pub mod estimate;
mod features;
pub mod hardware;

use mb_analyze::Report;
use mb_ir::ModelIr;
use serde::de::DeserializeOwned;
use serde::Serialize;

pub use estimate::{
    Confidence, CostInputs, Effect, QualityRisk, Range, RiskLevel, Stage, TrunkUse,
};
pub use features::catalog;

/// Feature parameters, e.g. from a recipe's `[[feature]]` table.
pub type Params = serde_json::Map<String, serde_json::Value>;

#[derive(Debug, thiserror::Error)]
pub enum FeatureError {
    #[error("unknown feature `{0}`; known: {1}")]
    Unknown(String, String),
    #[error("{feature}: invalid parameters: {msg}")]
    Params { feature: &'static str, msg: String },
}

/// Parses a feature's parameters into its typed struct.
pub fn parse_params<T: DeserializeOwned>(
    feature: &'static str,
    params: &Params,
) -> Result<T, FeatureError> {
    serde_json::from_value(serde_json::Value::Object(params.clone())).map_err(|e| {
        FeatureError::Params {
            feature,
            msg: e.to_string(),
        }
    })
}

/// What a plugin sees: the model, its static analysis, and optionally a
/// reference model (e.g. the full-precision base a feature is ported from).
pub struct Context<'a> {
    pub ir: &'a ModelIr,
    pub report: Report,
    pub reference: Option<&'a ModelIr>,
}

impl<'a> Context<'a> {
    pub fn new(ir: &'a ModelIr, reference: Option<&'a ModelIr>) -> Self {
        Self {
            ir,
            report: mb_analyze::analyze(ir, None),
            reference,
        }
    }

    /// Inputs for the training cost model. Forward params exclude MTP modules
    /// and multimodal towers, which don't run in text training of the trunk.
    pub fn cost_inputs(&self) -> CostInputs {
        let p = &self.report.params;
        CostInputs {
            forward_params: p.total - p.mtp - p.multimodal,
            hidden: self.ir.hidden_size.unwrap_or(0),
            layers: self.ir.layers.len() as u64,
            vocab: self.ir.vocab_size.unwrap_or(0),
            source_bits: self.report.quantization.trunk_bits_per_param.max(1.0),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum Detection {
    Absent,
    Present(String),
    /// Something equivalent or partial exists.
    Partial(String),
}

#[derive(Clone, Debug, Default, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct Compat {
    /// Reasons the feature can't be applied as requested.
    pub blockers: Vec<String>,
    /// Applicable, but with caveats.
    pub warnings: Vec<String>,
}

impl Compat {
    pub fn ok(&self) -> bool {
        self.blockers.is_empty()
    }
}

/// A plugin's estimate before pricing on hardware.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct Estimate {
    pub effects: Vec<Effect>,
    pub stages: Vec<Stage>,
    pub risk: QualityRisk,
    pub assumptions: Vec<String>,
    pub confidence: Confidence,
    pub references: Vec<String>,
}

/// What the next step works on after a [`RunStep`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(tag = "kind", content = "path", rename_all = "snake_case")]
pub enum Produces {
    /// The model is unchanged (e.g. a separate drafter or sidecar).
    Nothing,
    /// A new GGUF that later steps should use (`{run}`-relative path).
    Gguf(String),
    /// A Hugging Face checkpoint that no GGUF runtime loads; later GGUF steps can't follow it.
    Hf(String),
}

/// One command that carries out part of a feature. In `command`, `{model}`
/// is the model the step works on, `{stem}` its file stem and `{run}` the
/// feature's run directory; `<...>` marks inputs the user supplies.
#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct RunStep {
    pub trunk: TrunkUse,
    /// The estimate stage this step trains, if any.
    pub stage: Option<String>,
    /// `None`: modelbuilder has no runner for this step yet.
    pub command: Option<String>,
    pub produces: Produces,
    pub note: String,
}

impl RunStep {
    pub fn run(
        trunk: TrunkUse,
        stage: Option<&str>,
        command: String,
        produces: Produces,
        note: &str,
    ) -> Self {
        Self {
            trunk,
            stage: stage.map(str::to_owned),
            command: Some(command),
            produces,
            note: note.into(),
        }
    }

    pub fn missing(trunk: TrunkUse, stage: &str, note: &str) -> Self {
        Self {
            trunk,
            stage: Some(stage.into()),
            command: None,
            produces: Produces::Nothing,
            note: note.into(),
        }
    }
}

pub trait Feature: Sync {
    fn id(&self) -> &'static str;
    fn title(&self) -> &'static str;
    fn summary(&self) -> &'static str;
    fn detect(&self, ctx: &Context) -> Detection;
    fn check_compat(&self, ctx: &Context, params: &Params) -> Result<Compat, FeatureError>;
    fn estimate(&self, ctx: &Context, params: &Params) -> Result<Estimate, FeatureError>;
    /// The tensor and metadata changes surgery would make, in words.
    fn surgery_outline(&self, ctx: &Context, params: &Params) -> Result<Vec<String>, FeatureError>;
    fn export_notes(&self, ctx: &Context, params: &Params) -> Result<Vec<String>, FeatureError>;
    /// The commands that apply the feature, in order. The planner orders
    /// them across features and chains the models they produce.
    fn pipeline(&self, _ctx: &Context, _params: &Params) -> Result<Vec<RunStep>, FeatureError> {
        Ok(vec![])
    }
}

/// Looks a feature up by id.
pub fn feature(id: &str) -> Result<&'static dyn Feature, FeatureError> {
    catalog()
        .iter()
        .copied()
        .find(|f| f.id() == id)
        .ok_or_else(|| {
            FeatureError::Unknown(
                id.to_string(),
                catalog()
                    .iter()
                    .map(|f| f.id())
                    .collect::<Vec<_>>()
                    .join(", "),
            )
        })
}
