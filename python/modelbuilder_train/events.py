"""JSONL progress events, version 1 (``schema/events.v1.schema.json``)."""

from __future__ import annotations

import json
import sys
import time
from typing import IO, Any

from modelbuilder_train.spec import SCHEMA_VERSION


class EventWriter:
    """Writes one JSON object per line and flushes, so a reader sees events live."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        self.stream = stream or sys.stdout
        self.stage: str | None = None

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        rec = {"schema_version": SCHEMA_VERSION, "event": event, "time": time.time(), "stage": self.stage, **fields}
        self.stream.write(json.dumps(rec) + "\n")
        self.stream.flush()
        return rec
