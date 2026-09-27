//! Directory listing for the model picker: which entries are checkpoints,
//! recipes or job specs. Only names and sizes are read.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{ApiError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    Dir,
    /// A .gguf file.
    Gguf,
    /// A directory with config.json and .safetensors files (an HF checkpoint).
    HfModel,
    /// A .toml file (possibly a recipe).
    Recipe,
    /// A .json file named like a job spec (`job*.json`).
    JobSpec,
    /// A .jsonl file (possibly a job's events).
    Events,
    Other,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct DirEntry {
    pub name: String,
    pub path: String,
    pub kind: EntryKind,
    /// File size in bytes (files only).
    pub size: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
pub struct DirListing {
    pub path: String,
    pub parent: Option<String>,
    /// Directories first, then files, each sorted by name. Hidden entries are skipped.
    pub entries: Vec<DirEntry>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "ts", derive(ts_rs::TS))]
#[serde(default, deny_unknown_fields)]
pub struct ListRequest {
    /// Directory to list; the current directory if empty.
    pub path: String,
}

fn is_hf_model(dir: &Path) -> bool {
    dir.join("config.json").is_file()
        && std::fs::read_dir(dir).is_ok_and(|rd| {
            rd.flatten()
                .any(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        })
}

fn kind_of(path: &Path, is_dir: bool) -> EntryKind {
    if is_dir {
        return if is_hf_model(path) {
            EntryKind::HfModel
        } else {
            EntryKind::Dir
        };
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    match path.extension().and_then(|x| x.to_str()) {
        Some("gguf") => EntryKind::Gguf,
        Some("toml") => EntryKind::Recipe,
        Some("json") if name.starts_with("job") => EntryKind::JobSpec,
        Some("jsonl") => EntryKind::Events,
        _ => EntryKind::Other,
    }
}

pub fn list(req: &ListRequest) -> Result<DirListing> {
    let dir: PathBuf = if req.path.is_empty() {
        std::env::current_dir().map_err(|e| ApiError::Internal(e.to_string()))?
    } else {
        PathBuf::from(&req.path)
    };
    let dir = std::fs::canonicalize(&dir)
        .map_err(|e| ApiError::NotFound(format!("{}: {e}", dir.display())))?;
    let rd = std::fs::read_dir(&dir)
        .map_err(|e| ApiError::BadRequest(format!("{}: {e}", dir.display())))?;
    let mut entries: Vec<DirEntry> = rd
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                return None;
            }
            let path = e.path();
            // Follow symlinks: models are often linked into a working directory.
            let meta = std::fs::metadata(&path).ok()?;
            Some(DirEntry {
                kind: kind_of(&path, meta.is_dir()),
                size: meta.is_file().then_some(meta.len()),
                path: path.display().to_string(),
                name,
            })
        })
        .collect();
    entries.sort_by(|a, b| {
        let dir_a = matches!(a.kind, EntryKind::Dir | EntryKind::HfModel);
        let dir_b = matches!(b.kind, EntryKind::Dir | EntryKind::HfModel);
        dir_b.cmp(&dir_a).then_with(|| a.name.cmp(&b.name))
    });
    Ok(DirListing {
        parent: dir.parent().map(|p| p.display().to_string()),
        path: dir.display().to_string(),
        entries,
    })
}
