//! Normalized model intermediate representation (IR).
//!
//! Format readers (`mb-formats`) produce a [`RawModel`]: the tensor index plus
//! whatever metadata the checkpoint carries. [`ModelIr::from_raw`] turns that
//! into a format-independent description of the architecture that analyzers,
//! planners and feature plugins all work from.
//!
//! Nothing in this crate performs IO or touches tensor data.

mod config;
mod dtype;
mod ir;
mod naming;
mod raw;

pub use config::{ConfigView, Key};
pub use dtype::{DType, GgmlType};
pub use ir::{
    AttentionKind, AttentionSpec, FfnSpec, Layer, LinearAttentionSpec, Mixer, MlaSpec, ModelIr,
    MoeSpec, MtpInfo, RopeInfo,
};
pub use naming::{classify, Component, TensorKind, TensorRole};
pub use raw::{AuxFiles, MetaType, MetaValue, Metadata, RawModel, SourceFormat, TensorInfo};
