"""The JSON Schemas in schema/ and the pydantic models must agree."""

import copy
import io
import json

import jsonschema
import pytest
from conftest import SCHEMA
from pydantic import ValidationError

from modelbuilder_train.events import EventWriter
from modelbuilder_train.spec import JobSpec

SPEC_SCHEMA = json.loads((SCHEMA / "job-spec.v1.schema.json").read_text())
EVENT_SCHEMA = json.loads((SCHEMA / "events.v1.schema.json").read_text())
EXAMPLE = json.loads((SCHEMA / "examples" / "mtp-align.job.json").read_text())


def both_accept(doc: dict) -> None:
    jsonschema.validate(doc, SPEC_SCHEMA)
    JobSpec.model_validate(doc)


def both_reject(doc: dict) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, SPEC_SCHEMA)
    with pytest.raises(ValidationError):
        JobSpec.model_validate(doc)


def test_example_spec_is_valid():
    both_accept(EXAMPLE)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(extra=1),
        lambda d: d.update(schema_version=2),
        lambda d: d.update(backend="jax"),
        lambda d: d.update(stages=[]),
        lambda d: d["stages"][0].pop("mtp_align"),
        lambda d: d["stages"][0].update(kind="unknown"),
        lambda d: d["stages"][0]["hyper"].update(lr=0),
        lambda d: d["stages"][0]["hyper"].update(seq_len=2),
        lambda d: d["stages"][0]["hyper"].update(dtype="float16"),
        lambda d: d["stages"][0]["mtp_align"].update(eval_fraction=0.9),
        lambda d: d["stages"][0]["mtp_align"].update(bogus="x"),
    ],
)
def test_invalid_specs_are_rejected_by_both(mutate):
    doc = copy.deepcopy(EXAMPLE)
    mutate(doc)
    both_reject(doc)


def test_example_events_are_valid():
    lines = (SCHEMA / "examples" / "events.jsonl").read_text().splitlines()
    kinds = set()
    for line in lines:
        rec = json.loads(line)
        jsonschema.validate(rec, EVENT_SCHEMA)
        kinds.add(rec["event"])
    assert kinds == {"started", "progress", "eval", "checkpoint", "finished", "error"}


def test_event_writer_output_is_valid():
    buf = io.StringIO()
    ev = EventWriter(buf)
    ev.stage = "s"
    ev.emit("progress", step=1, steps=2, loss=0.5)
    ev.emit("finished", status="ok", outputs={"a": "b"})
    for line in buf.getvalue().splitlines():
        jsonschema.validate(json.loads(line), EVENT_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"schema_version": 1, "event": "progress", "time": 0.0}, EVENT_SCHEMA)
