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
    eval_fraction: float = Field(default=0.05, ge=0, le=0.5)


class Stage(_Strict):
    name: str = Field(min_length=1)
    kind: Literal["mtp_align", "trunk_distill"]
    mtp_align: MtpAlign | None = None
    trunk_distill: TrunkDistill | None = None
    hyper: Hyper

    @model_validator(mode="after")
    def _kind_payload(self) -> Stage:
        if self.kind == "mtp_align" and self.mtp_align is None:
            raise ValueError("kind mtp_align requires an mtp_align table")
        if self.kind == "trunk_distill" and self.trunk_distill is None:
            raise ValueError("kind trunk_distill requires a trunk_distill table")
        return self


class JobSpec(_Strict):
    schema_version: Literal[1]
    job_id: str = Field(min_length=1)
    created_by: str | None = None
    backend: Literal["torch"]
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
