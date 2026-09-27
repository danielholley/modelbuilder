//! The operations behind the dashboards.
//!
//! The web server (`mb-server`) and the terminal UI (`mb-tui`) are both thin
//! clients of this crate: every request and response type is defined here
//! (and exported to TypeScript for the web UI), and every operation is a plain
//! function over the core crates. Nothing here knows about HTTP or terminals.

pub mod fs;
pub mod jobs;

use std::path::{Path, PathBuf};

pub use mb_analyze::weights::WeightStatsReport;
pub use mb_analyze::Report;
pub use mb_features::hardware::HardwareProfile;
pub use mb_plan::Plan;

use mb_analyze::weights::WeightStatsOptions;
use mb_ir::{Layer, ModelIr, TensorInfo, TensorRole};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    /// The request can't be served as given (bad path, bad recipe, unknown feature).
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, ApiError>;

fn bad(e: impl std::fmt::Display) -> ApiError {
    ApiError::BadRequest(e.to_string())
}

/// Opens a checkpoint (a .gguf file or an HF directory) and builds its IR.
/// Only headers are read.
pub fn open(path: &Path) -> Result<(mb_formats::LoadedModel, ModelIr)> {
    let m = mb_formats::open(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    let ir = ModelIr::from_raw(m.raw.clone());
    Ok((m, ir))
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(default, deny_unknown_fields)]
pub struct InspectRequest {
    pub path: String,
    /// Context length for KV-cache totals (defaults to the model's max positions).
    pub context: Option<u64>,
    /// Also return the tensor index.
    pub tensors: bool,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct TensorRow {
    pub info: TensorInfo,
    pub role: TensorRole,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct LayerRow {
    pub layer: Layer,
    /// Short label, e.g. `gqa(24q/4kv,d256)+dense` (the CLI's layer pattern notation).
    pub label: String,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct InspectResponse {
    pub report: Report,
    /// Per-layer structure, in order.
    pub layers: Vec<LayerRow>,
    /// The tensor index, when requested.
    pub tensors: Option<Vec<TensorRow>>,
}

pub fn inspect(req: &InspectRequest) -> Result<InspectResponse> {
    let (_, ir) = open(Path::new(&req.path))?;
    let report = mb_analyze::analyze(&ir, req.context);
    let tensors = req.tensors.then(|| {
        ir.tensors()
            .map(|(t, r)| TensorRow {
                info: t.clone(),
                role: *r,
            })
            .collect()
    });
    Ok(InspectResponse {
        report,
        layers: ir
            .layers
            .iter()
            .map(|l| LayerRow {
                layer: l.clone(),
                label: mb_analyze::layer_label(l),
            })
            .collect(),
        tensors,
    })
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(default, deny_unknown_fields)]
pub struct StatsRequest {
    pub path: String,
    /// Only tensors whose name contains one of these (all if empty).
    pub only: Vec<String>,
    /// Compute K/V singular-value spectra per attention layer.
    pub kv_spectra: bool,
}

/// Streams the weights (one tensor at a time) and computes statistics. Slow
/// on large models: run it off any UI or async thread.
pub fn stats(req: &StatsRequest) -> Result<WeightStatsReport> {
    let (model, ir) = open(Path::new(&req.path))?;
    let opts = WeightStatsOptions {
        only: req.only.clone(),
        kv_spectra: req.kv_spectra,
    };
    mb_analyze::weights::weight_stats(&model, &ir, &opts).map_err(bad)
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(default, deny_unknown_fields)]
pub struct PlanRequest {
    /// The model; overrides the recipe's source.
    pub path: Option<String>,
    /// Recipe TOML text (the same format as `plan --recipe`).
    pub recipe: Option<String>,
    /// Feature specs, `id` or `id:key=value,...`, added to the recipe's.
    /// With no features at all, the whole catalog is evaluated.
    pub features: Vec<String>,
    /// Hardware profile ids; override the recipe's. Empty: all profiles.
    pub hardware: Vec<String>,
}

/// Same semantics as `modelbuilder plan`.
pub fn plan(req: &PlanRequest) -> Result<Plan> {
    let (source, mut requests, mut hw_ids) = match &req.recipe {
        Some(text) => {
            let recipe = mb_plan::Recipe::parse(text).map_err(bad)?;
            let hw = recipe.hardware_ids();
            (Some(recipe.source.path.clone()), recipe.features, hw)
        }
        None => (None, Vec::new(), Vec::new()),
    };
    let source = req
        .path
        .clone()
        .or(source)
        .filter(|p| !p.is_empty())
        .ok_or_else(|| bad("give a model path or a recipe"))?;
    requests.extend(req.features.iter().map(|f| mb_plan::parse_feature_spec(f)));
    if !req.hardware.is_empty() {
        hw_ids = req.hardware.clone();
    }
    let hw = mb_plan::resolve_hardware(&hw_ids).map_err(bad)?;
    let (_, ir) = open(Path::new(&source))?;
    // A feature's `from` names a reference model (e.g. the base to port an MTP head from).
    let reference = requests
        .iter()
        .find_map(|r| {
            r.params
                .get("from")
                .and_then(|v| v.as_str())
                .map(PathBuf::from)
        })
        .map(|p| open(&p).map(|(_, ir)| ir))
        .transpose()?;
    let ctx = mb_features::Context::new(&ir, reference.as_ref());
    mb_plan::plan(&ctx, &requests, &hw).map_err(bad)
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct CatalogFeature {
    pub id: String,
    pub title: String,
    pub summary: String,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct Catalog {
    pub features: Vec<CatalogFeature>,
    pub hardware: Vec<HardwareProfile>,
}

/// The feature catalog and hardware profiles (`modelbuilder features`).
pub fn catalog() -> Catalog {
    Catalog {
        features: mb_features::catalog()
            .iter()
            .map(|f| CatalogFeature {
                id: f.id().into(),
                title: f.title().into(),
                summary: f.summary().into(),
            })
            .collect(),
        hardware: mb_features::hardware::profiles().to_vec(),
    }
}
