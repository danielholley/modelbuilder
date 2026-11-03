"""Job spec, version 1. Mirrors ``schema/job-spec.v1.schema.json``; tests keep them in sync."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Hyper(_Strict):
    lr: float = Field(gt=0)
    steps: int = Field(ge=1)
    seq_len: int = Field(ge=4)
    batch_seqs: int = Field(ge=1)
    warmup_steps: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.0, ge=0)
    grad_clip: float | None = 1.0
    log_every: int = Field(default=10, ge=1)
    eval_every: int = Field(default=100, ge=1)
    seed: int = 0
    dtype: Literal["float32", "bfloat16"] = "float32"


class MtpAlign(_Strict):
    reference_config: str
    init_head: str | None = None
    frozen_tensors: str
    embedding_tensor: str
    lm_head_tensor: str
    features: str
    eval_fraction: float = Field(default=0.05, ge=0, le=0.5)


class TrunkDistill(_Strict):
    model: str
    texts: str
    trainable: list[str] = Field(min_length=1)
    kv_format: Literal["q8_0", "q4_0", "nvfp4"] | None = None
    kv_share_group: int | None = Field(default=None, ge=2)
    weight_fakequant: bool = True
    teacher: str | None = None
    loss: Literal["kl", "ce"] = "kl"
    device_map: Literal["auto", "balanced", "sequential"] | None = None
    shard: Literal["fsdp"] | None = None
    lora_rank: int | None = Field(default=None, ge=1)
    lora_alpha: float | None = Field(default=None, gt=0)
    eval_fraction: float = Field(default=0.05, ge=0, le=0.5)


class DraftAlign(_Strict):
    kind: Literal["medusa", "eagle", "dspark"]
    heads: int = Field(default=4, ge=1)
    blocks: int = Field(default=1, ge=1)
    share_lm_head: bool = False
    reference_config: str | None = None
    # Medusa and EAGLE: precomputed features and the frozen tensors.
    frozen_tensors: str | None = None
    embedding_tensor: str | None = None
    lm_head_tensor: str | None = None
    features: str | None = None
    # DSpark: an online target (HF directory) and texts.
    target: str | None = None
    texts: str | None = None
    block_size: int = Field(default=7, ge=2)
    target_layer_ids: list[int] | None = None
    mask_token_id: int | None = Field(default=None, ge=0)
    markov_rank: int = Field(default=256, ge=0)
    num_anchors: int = Field(default=512, ge=1)
    confidence_head: bool = True
    eval_fraction: float = Field(default=0.05, ge=0, le=0.5)

    @model_validator(mode="after")
    def _inputs(self) -> DraftAlign:
        if self.kind == "eagle" and self.reference_config is None:
            raise ValueError("an EAGLE drafter needs reference_config")
        if self.kind == "dspark":
            if self.target is None or self.texts is None:
                raise ValueError("a DSpark drafter needs target and texts")
            if self.confidence_head and self.markov_rank == 0:
                raise ValueError("the DSpark confidence head reads the Markov embedding: markov_rank must be > 0")
        elif None in (self.frozen_tensors, self.embedding_tensor, self.lm_head_tensor, self.features):
            raise ValueError(f"{self.kind} needs frozen_tensors, embedding_tensor, lm_head_tensor and features")
        return self


KINDS = ("mtp_align", "trunk_distill", "draft_align")


class Stage(_Strict):
    name: str = Field(min_length=1)
    kind: Literal["mtp_align", "trunk_distill", "draft_align"]
    mtp_align: MtpAlign | None = None
    trunk_distill: TrunkDistill | None = None
    draft_align: DraftAlign | None = None
    hyper: Hyper

    @model_validator(mode="after")
    def _kind_payload(self) -> Stage:
        for kind in KINDS:
            if self.kind == kind and getattr(self, kind) is None:
                raise ValueError(f"kind {kind} requires a {kind} table")
        return self


class JobSpec(_Strict):
    schema_version: Literal[1]
    job_id: str = Field(min_length=1)
    created_by: str | None = None
    backend: Literal["torch", "mlx"]
    device: Literal["auto", "cpu", "cuda", "mps"]
    hardware_profile: str | None = None
    output_dir: str = Field(min_length=1)
    stages: list[Stage] = Field(min_length=1)

    @classmethod
    def load(cls, path: Path) -> JobSpec:
        return cls.model_validate_json(Path(path).read_text())


def resolve(base: Path, p: str | None) -> Path | None:
    """Spec paths are absolute or relative to the spec file's directory."""
    if p is None:
        return None
    q = Path(p)
    return q if q.is_absolute() else (base / q)
