"""Structured JSONL logger for WebSocket training runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


SAVED_DIR = Path(__file__).resolve().parents[5] / "saved"


class TrainingRunLogger:
    """Append structured training events to a JSONL file per run."""

    def __init__(self, variant: str) -> None:
        self.variant = variant
        stamp = time.strftime("%Y%m%dT%H%M%S")
        self.run_id = f"{stamp}_{variant}"
        self.base_dir = SAVED_DIR / "training_logs" / variant
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / f"{self.run_id}.jsonl"

    def log(self, event: dict[str, Any]) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "variant": self.variant,
            "runId": self.run_id,
            "event": event.get("type"),
            "payload": event.get("payload", {}),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

