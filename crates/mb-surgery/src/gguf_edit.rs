//! Structural GGUF edits that need no retraining to write: removing layers
//! (depth pruning) and enabling YaRN RoPE scaling. Tensors are copied byte
//! for byte in their stored format and basis; only names and metadata change.

use std::borrow::Cow;
use std::path::Path;

use mb_formats::{gguf, LoadedModel, TensorToWrite};
use mb_ir::{MetaValue, Metadata, ModelIr, SourceFormat};

use crate::{SurgeryError, SurgeryReport, WrittenTensor};

type Kv = Vec<(String, MetaValue)>;

fn gguf_kv(ir: &ModelIr) -> Result<(Kv, String), SurgeryError> {
    if ir.raw.format != SourceFormat::Gguf {
        return Err(SurgeryError::Unsupported(
            "the input must be a GGUF file".into(),
        ));
    }
    let Metadata::Gguf { kv, .. } = &ir.raw.metadata else {
        unreachable!("GGUF inputs carry GGUF metadata")
    };
    let arch = kv
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .and_then(|(_, v)| v.as_str())
        .ok_or_else(|| SurgeryError::Incompatible("no general.architecture".into()))?
        .to_string();
    Ok((kv.clone(), arch))
}

fn check_output(ir: &ModelIr, out: &Path, overwrite: bool) -> Result<(), SurgeryError> {
    if !out.exists() {
        return Ok(());
    }
    let same = |p: &Path| std::fs::canonicalize(p).ok() == std::fs::canonicalize(out).ok();
    if ir.raw.files.iter().any(|p| same(p)) {
        return Err(SurgeryError::Output(
            "the output path is one of the input files".into(),
        ));
    }
    if !overwrite {
        return Err(SurgeryError::Output(format!(
            "{} exists (pass overwrite)",
            out.display()
        )));
    }
    Ok(())
}

/// Same integer width as `like`.
fn int_like(like: &MetaValue, v: u64) -> MetaValue {
    match like {
        MetaValue::U64(_) => MetaValue::U64(v),
        MetaValue::I32(_) => MetaValue::I32(v as i32),
        MetaValue::I64(_) => MetaValue::I64(v as i64),
        _ => MetaValue::U32(v as u32),
    }
}

fn set(kv: &mut Kv, key: &str, v: MetaValue) {
    match kv.iter_mut().find(|(k, _)| k == key) {
        Some(slot) => slot.1 = v,
        None => kv.push((key.to_string(), v)),
    }
}

/// `blk.{i}.rest` → `(i, rest)`.
fn layer_of(name: &str) -> Option<(u32, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (i, rest) = rest.split_once('.')?;
    Some((i.parse().ok()?, rest))
}

/// The new name of a layer tensor, or `None` if its layer is removed.
fn renumber(name: &str, start: u32, count: u32) -> Option<String> {
    match layer_of(name) {
        Some((i, _)) if (start..start + count).contains(&i) => None,
        Some((i, rest)) if i >= start + count => Some(format!("blk.{}.{rest}", i - count)),
        _ => Some(name.to_string()),
    }
}

/// Removes layers `start..start + count` and renumbers the ones above.
///
/// Per-layer metadata arrays (length `block_count`) lose the removed entries;
/// string arrays of tensor names (e.g. which weights are rotated) are
/// renumbered. An integer layer-pattern key (`*interval*`, `*pattern*`) must
/// divide `count`, so the pattern of mixer types is unchanged.
pub fn prune_layers(
    model: &LoadedModel,
    ir: &ModelIr,
    start: u32,
    count: u32,
    out: &Path,
    overwrite: bool,
) -> Result<SurgeryReport, SurgeryError> {
    let (mut kv, arch) = gguf_kv(ir)?;
    check_output(ir, out, overwrite)?;
    let bc_key = format!("{arch}.block_count");
    let bc_val = kv
        .iter()
        .find(|(k, _)| *k == bc_key)
        .map(|(_, v)| v.clone())
        .ok_or_else(|| SurgeryError::Incompatible(format!("no {bc_key}")))?;
    let n = bc_val.as_u64().unwrap_or(0) as u32;
    let trunk = ir.layers.len() as u32;
    if count == 0 || start + count > trunk || count >= trunk {
        return Err(SurgeryError::Incompatible(format!(
            "cannot remove layers {start}..{} from a {trunk}-layer trunk",
            start + count
        )));
    }
    let mut changed = vec![format!("{bc_key}: {n} → {}", n - count)];
    let prefix = format!("{arch}.");
    for (k, v) in kv.iter_mut() {
        if !k.starts_with(&prefix) || *k == bc_key {
            continue;
        }
        match v {
            MetaValue::Array { values, .. } if values.len() == n as usize => {
                let mut i = 0u32;
                values.retain(|_| {
                    let keep = !(start..start + count).contains(&i);
                    i += 1;
                    keep
                });
                changed.push(format!("{k}: per-layer array, {count} entries removed"));
            }
            _ if k.contains("interval") || k.contains("pattern") => {
                if let Some(p) = v.as_u64().filter(|&p| p > 1) {
                    if u64::from(count) % p != 0 {
                        return Err(SurgeryError::Incompatible(format!(
                            "{k} = {p}: removing {count} layers (not a multiple of {p}) would break the layer pattern"
                        )));
                    }
                }
            }
            _ => {}
        }
    }
    for (k, v) in kv.iter_mut() {
        if let MetaValue::Array { values, .. } = v {
            if values
                .iter()
                .any(|x| x.as_str().is_some_and(|s| layer_of(s).is_some()))
            {
                values.retain_mut(|x| match x.as_str().map(|s| renumber(s, start, count)) {
                    Some(None) => false,
                    Some(Some(s)) => {
                        *x = MetaValue::String(s);
                        true
                    }
                    None => true,
                });
                changed.push(format!("{k}: tensor names renumbered"));
            }
        }
    }
    set(&mut kv, &bc_key, int_like(&bc_val, u64::from(n - count)));
    set(
        &mut kv,
        "modelbuilder.pruned_layers",
        MetaValue::String(format!("{start}..{}", start + count)),
    );
    changed.push("modelbuilder.pruned_layers (added)".into());

    let mut tensors = Vec::new();
    let mut written = Vec::new();
    for t in &ir.raw.tensors {
        let Some(name) = renumber(&t.name, start, count) else {
            continue;
        };
        if name != t.name {
            written.push(WrittenTensor {
                name: name.clone(),
                dtype: t.dtype.to_string(),
                shape: t.shape.clone(),
                source: format!("target:{}", t.name),
                transform: "renamed, bytes copied".into(),
            });
        }
        tensors.push(TensorToWrite {
            name,
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: Cow::Borrowed(model.tensor_bytes(t)?),
        });
    }
    let removed = ir.raw.tensors.len() - tensors.len();
    gguf::write(out, &kv, &tensors)?;
    Ok(SurgeryReport {
        output: out.to_owned(),
        bytes: file_len(out)?,
        tensors: written,
        metadata: changed,
        notes: vec![format!(
            "Removed layers {start}..{} ({removed} tensors); every other tensor copied byte for byte.",
            start + count
        )],
    })
}

/// Enables YaRN RoPE scaling: `original_context × factor` tokens.
pub fn set_yarn(
    model: &LoadedModel,
    ir: &ModelIr,
    factor: f32,
    original_context: Option<u64>,
    out: &Path,
    overwrite: bool,
) -> Result<SurgeryReport, SurgeryError> {
    let (mut kv, arch) = gguf_kv(ir)?;
    check_output(ir, out, overwrite)?;
    if factor.is_nan() || factor <= 1.0 {
        return Err(SurgeryError::Incompatible(
            "factor must be greater than 1".into(),
        ));
    }
    let ctx_key = format!("{arch}.context_length");
    let ctx_val = kv
        .iter()
        .find(|(k, _)| *k == ctx_key)
        .map(|(_, v)| v.clone());
    let orig_key = format!("{arch}.rope.scaling.original_context_length");
    let existing_orig = kv
        .iter()
        .find(|(k, _)| *k == orig_key)
        .and_then(|(_, v)| v.as_u64());
    let orig = original_context
        .or(existing_orig)
        .or_else(|| ctx_val.as_ref().and_then(MetaValue::as_u64))
        .ok_or_else(|| {
            SurgeryError::Incompatible(format!("no {ctx_key}; pass the original context"))
        })?;
    let new = (orig as f64 * f64::from(factor)).round() as u64;
    set(
        &mut kv,
        &format!("{arch}.rope.scaling.type"),
        MetaValue::String("yarn".into()),
    );
    set(
        &mut kv,
        &format!("{arch}.rope.scaling.factor"),
        MetaValue::F32(factor),
    );
    set(&mut kv, &orig_key, MetaValue::U32(orig as u32));
    let like = ctx_val.unwrap_or(MetaValue::U32(0));
    set(&mut kv, &ctx_key, int_like(&like, new));
    let tensors: Vec<TensorToWrite> = ir
        .raw
        .tensors
        .iter()
        .map(|t| {
            Ok(TensorToWrite {
                name: t.name.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                data: Cow::Borrowed(model.tensor_bytes(t)?),
            })
        })
        .collect::<Result<_, SurgeryError>>()?;
    gguf::write(out, &kv, &tensors)?;
    Ok(SurgeryReport {
        output: out.to_owned(),
        bytes: file_len(out)?,
        tensors: vec![],
        metadata: vec![
            format!("{arch}.rope.scaling.type = yarn"),
            format!("{arch}.rope.scaling.factor = {factor}"),
            format!("{orig_key} = {orig}"),
            format!("{ctx_key}: → {new}"),
        ],
        notes: vec![
            "Metadata only; every tensor copied byte for byte.".into(),
            "llama.cpp derives YaRN's attention factor and ramp from these keys; override with --yarn-* flags if needed.".into(),
        ],
    })
}

fn file_len(p: &Path) -> Result<u64, SurgeryError> {
    Ok(std::fs::metadata(p)
        .map_err(|e| SurgeryError::Output(e.to_string()))?
        .len())
}

#[cfg(test)]
mod tests {
    use super::renumber;

    #[test]
    fn renumbering() {
        assert_eq!(renumber("blk.2.attn_q.weight", 2, 2), None);
        assert_eq!(
            renumber("blk.5.attn_q.weight", 2, 2).unwrap(),
            "blk.3.attn_q.weight"
        );
        assert_eq!(renumber("blk.1.x", 2, 2).unwrap(), "blk.1.x");
        assert_eq!(renumber("output.weight", 2, 2).unwrap(), "output.weight");
    }
}
