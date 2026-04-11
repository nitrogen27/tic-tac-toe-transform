from __future__ import annotations

import json
from pathlib import Path

import pytest

from gomoku_api.ws import training_worker


class _DummyLogger:
    _counter = 0

    def __init__(self, variant: str, base_dir: Path) -> None:
        type(self)._counter += 1
        self.variant = variant
        self.run_id = f"test-run-{type(self)._counter}"
        self.path = base_dir / f"{self.run_id}_{variant}.jsonl"

    def log(self, event: dict) -> None:
        record = {
            "ts": "2026-04-11T00:00:00",
            "variant": self.variant,
            "runId": self.run_id,
            "event": event.get("type"),
            "payload": event.get("payload", {}),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


@pytest.mark.asyncio
async def test_training_worker_marks_background_terminal_state_on_success(tmp_path, monkeypatch) -> None:
    request_path = tmp_path / "request.json"
    meta_path = tmp_path / "meta.json"
    cancel_path = tmp_path / "cancel.flag"
    request_path.write_text(json.dumps({"payload": {}}), encoding="utf-8")
    monkeypatch.setattr(training_worker, "TrainingRunLogger", lambda variant: _DummyLogger(variant, tmp_path))

    async def fake_train_variant(variant, callback, **kwargs):
        await callback({"type": "train.done", "payload": {"variant": variant, "evaluationQueued": True}})
        return {"deferredEvaluator": {"variant": variant}}

    async def fake_tail(context, callback):
        await callback({"type": "background_train.started", "payload": {"variant": context["variant"]}})
        await callback({"type": "background_train.done", "payload": {"variant": context["variant"]}})
        return {"variant": context["variant"]}

    monkeypatch.setattr(training_worker, "train_variant", fake_train_variant)
    monkeypatch.setattr(training_worker, "run_deferred_evaluator_tail", fake_tail)

    exit_code = await training_worker._run_worker(
        variant="ttt5",
        request_path=request_path,
        meta_path=meta_path,
        cancel_path=cancel_path,
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert meta["finalEvent"] == "background_train.done"


@pytest.mark.asyncio
async def test_training_worker_reports_background_error_instead_of_train_error(tmp_path, monkeypatch) -> None:
    request_path = tmp_path / "request.json"
    meta_path = tmp_path / "meta.json"
    cancel_path = tmp_path / "cancel.flag"
    request_path.write_text(json.dumps({"payload": {}}), encoding="utf-8")
    monkeypatch.setattr(training_worker, "TrainingRunLogger", lambda variant: _DummyLogger(variant, tmp_path))

    async def fake_train_variant(variant, callback, **kwargs):
        await callback({"type": "train.done", "payload": {"variant": variant, "evaluationQueued": True}})
        return {"deferredEvaluator": {"variant": variant}}

    async def failing_tail(context, callback):
        raise RuntimeError("boom")

    monkeypatch.setattr(training_worker, "train_variant", fake_train_variant)
    monkeypatch.setattr(training_worker, "run_deferred_evaluator_tail", failing_tail)

    exit_code = await training_worker._run_worker(
        variant="ttt5",
        request_path=request_path,
        meta_path=meta_path,
        cancel_path=cancel_path,
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    log_path = Path(meta["logPath"])
    logged_events = [
        json.loads(line)["event"]
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert exit_code == 1
    assert meta["finalEvent"] == "background_train.error"
    assert "background_train.error" in logged_events
    assert "train.error" not in logged_events
