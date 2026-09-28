"""Training backends. A backend runs the stages of a job spec.

Only ``torch`` exists so far. It runs frozen-trunk stages from precomputed
trunk features, which makes it independent of how the trunk is stored
(e.g. PrismML's ternary GGUF, which PyTorch can't load directly).
"""

from __future__ import annotations

from pathlib import Path

from modelbuilder_train.events import EventWriter
from modelbuilder_train.spec import JobSpec, resolve


def run_torch(spec: JobSpec, spec_dir: Path, ev: EventWriter) -> dict[str, str]:
    from modelbuilder_train import dist
    from modelbuilder_train.mtp.train import pick_device, run_mtp_align

    device = dist.init(pick_device(spec.device))
    out_dir = resolve(spec_dir, spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for stage in spec.stages:
        ev.stage = stage.name
        if stage.kind == "trunk_distill":
            from modelbuilder_train.hf.distill import run_trunk_distill

            outputs.update(
                run_trunk_distill(
                    stage.trunk_distill, stage.hyper, lambda p: resolve(spec_dir, p), out_dir, device, ev, spec.job_id
                )
            )
        elif stage.kind == "mtp_align":
            outputs.update(
                run_mtp_align(
                    stage.mtp_align, stage.hyper, lambda p: resolve(spec_dir, p), out_dir, device, ev, spec.job_id
                )
            )
        else:  # pragma: no cover - the spec model rejects unknown kinds
            raise ValueError(f"stage kind {stage.kind} is not supported by the torch backend")
    ev.stage = None
    dist.barrier()
    dist.shutdown()
    return outputs


BACKENDS = {"torch": run_torch}
