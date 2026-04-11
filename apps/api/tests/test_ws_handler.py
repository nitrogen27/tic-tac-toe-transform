from __future__ import annotations

import asyncio
import json

import pytest

from gomoku_api.ws import handler


class DummyWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_json(self, message: dict) -> None:
        self.messages.append(message)


@pytest.mark.asyncio
async def test_train_on_games_rejects_duplicate_variant(monkeypatch) -> None:
    ws = DummyWebSocket()
    monkeypatch.setattr(handler, "_is_training_active", lambda variant: True)

    await handler._dispatch(ws, "train_on_games", {"variant": "ttt5"})

    assert ws.messages[-1] == {
        "type": "train.rejected",
        "payload": {"variant": "ttt5", "reason": "already running"},
    }


@pytest.mark.asyncio
async def test_get_training_status_dispatch_returns_current_state(monkeypatch) -> None:
    ws = DummyWebSocket()
    monkeypatch.setattr(
        handler,
        "_build_training_status",
        lambda variant: {
            "variant": variant,
            "active": True,
            "backgroundActive": False,
            "anyActive": True,
            "runId": "run-1",
            "lastEvent": "train.progress",
            "payload": {"phase": "self_play_train"},
        },
    )

    await handler._dispatch(ws, "get_training_status", {"variant": "ttt5"})

    assert ws.messages[-1] == {
        "type": "training.status",
        "payload": {
            "variant": "ttt5",
            "active": True,
            "backgroundActive": False,
            "anyActive": True,
            "runId": "run-1",
            "lastEvent": "train.progress",
            "payload": {"phase": "self_play_train"},
        },
    }


@pytest.mark.asyncio
async def test_finish_game_auto_train_queues_corpus_analysis_for_ttt5(monkeypatch) -> None:
    ws = DummyWebSocket()
    monkeypatch.setattr(handler._game_service, "finish_game", lambda game_id, winner: {"gameId": game_id, "winner": winner})
    queued: list[dict] = []

    async def fake_queue_finished_game_analysis(ws_obj, **kwargs):
        queued.append(kwargs)
        return True

    monkeypatch.setattr(handler, "_queue_finished_game_analysis", fake_queue_finished_game_analysis)

    await handler._dispatch(ws, "finish_game", {
        "variant": "ttt5",
        "gameId": "g1",
        "winner": 1,
        "autoTrain": True,
    })

    assert ws.messages[0] == {
        "type": "game.finished",
        "payload": {"gameId": "g1", "winner": 1},
    }
    assert queued == [{
        "game_id": "g1",
        "variant": "ttt5",
        "trigger_training": True,
        "epochs": 3,
        "batch_size": 256,
        "data_count": 1000,
    }]


@pytest.mark.asyncio
async def test_generate_engine_dataset_dispatch(monkeypatch) -> None:
    ws = DummyWebSocket()

    async def fake_generate_engine_dataset(variant: str, count: int, cb, *, phase_focus=None, backend="auto"):
        suffix = "builtin" if backend == "auto" else backend
        return f"C:/tmp/{variant}_{suffix}.json"

    monkeypatch.setattr(handler, "generate_engine_dataset", fake_generate_engine_dataset)

    await handler._dispatch(ws, "generate_engine_dataset", {"variant": "ttt5", "count": 1500})

    assert ws.messages[-1] == {
        "type": "dataset.done",
        "payload": {
            "path": "C:/tmp/ttt5_builtin.json",
            "count": 1500,
            "variant": "ttt5",
            "mode": "engine",
            "backend": "auto",
        },
    }


@pytest.mark.asyncio
async def test_run_training_diagnostics_dispatch(monkeypatch) -> None:
    ws = DummyWebSocket()

    async def fake_run_training_diagnostics(variant: str, **kwargs):
        return {"variant": variant, "tinyOverfitPassed": True, "datasetSize": kwargs["dataset_limit"]}

    monkeypatch.setattr(handler, "run_training_diagnostics", fake_run_training_diagnostics)

    await handler._dispatch(ws, "run_training_diagnostics", {"variant": "ttt5", "datasetLimit": 192})

    assert ws.messages[-1] == {
        "type": "training.diagnostics",
        "payload": {"variant": "ttt5", "tinyOverfitPassed": True, "datasetSize": 192},
    }


@pytest.mark.asyncio
async def test_run_training_schedules_deferred_evaluator(monkeypatch) -> None:
    events: list[dict] = []

    async def fake_cb(event: dict) -> None:
        events.append(event)

    async def fake_train_variant(variant: str, cb, **kwargs):
        await cb({"type": "train.done", "payload": {"variant": variant, "evaluationQueued": True}})
        return {"deferredEvaluator": {"variant": variant}}

    async def fake_run_deferred_evaluator_tail(context: dict, cb) -> dict:
        await cb({"type": "background_train.started", "payload": {"variant": context["variant"], "message": "bg start"}})
        await cb({"type": "background_train.done", "payload": {"variant": context["variant"], "message": "bg done"}})
        return {"variant": context["variant"]}

    monkeypatch.setattr(handler, "train_variant", fake_train_variant)
    monkeypatch.setattr(handler, "run_deferred_evaluator_tail", fake_run_deferred_evaluator_tail)

    await handler._run_training("ttt5", fake_cb)
    await asyncio.sleep(0)

    assert [event["type"] for event in events] == [
        "train.done",
        "background_train.started",
        "background_train.done",
    ]
    assert "ttt5" not in handler._active_background_eval_tasks


@pytest.mark.asyncio
async def test_is_training_active_considers_background_evaluator() -> None:
    task = asyncio.create_task(asyncio.sleep(0.05))
    handler._active_background_eval_tasks["ttt5"] = task
    try:
        assert handler._is_training_active("ttt5") is True
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        handler._active_background_eval_tasks.pop("ttt5", None)


def test_build_training_status_restores_chart_histories(tmp_path, monkeypatch) -> None:
    log_dir = tmp_path / "saved" / "training_logs" / "ttt5"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "20260411T999999_ttt5.jsonl"
    log_path.write_text(
        "\n".join([
            json.dumps({
                "event": "train.progress",
                "runId": "run-1",
                "payload": {
                    "phase": "turbo_train",
                    "metricsHistory": [{"epoch": 1, "loss": 1.2, "acc": 55.0, "mae": 0.4}],
                },
            }),
            json.dumps({
                "event": "train.progress",
                "runId": "run-1",
                "payload": {
                    "phase": "exam",
                    "cycle": 2,
                    "winrateVsAlgorithm": 0.75,
                    "decisiveWinRate": 0.5,
                    "drawRate": 0.25,
                    "winrateAsP1": 1.0,
                    "winrateAsP2": 0.5,
                    "balancedSideWinrate": 0.5,
                    "arenaWins": 3,
                    "arenaLosses": 1,
                    "arenaDraws": 0,
                },
            }),
            json.dumps({
                "event": "train.done",
                "runId": "run-1",
                "payload": {
                    "variant": "ttt5",
                    "cycles": 2,
                    "winrateVsAlgorithm": 0.8,
                    "confirmWinrateAsP1": 1.0,
                    "confirmWinrateAsP2": 0.6,
                    "confirmBalancedSideWinrate": 0.6,
                    "confirmDecisiveWinRate": 0.7,
                    "confirmDrawRate": 0.1,
                    "confirmWins": 8,
                    "confirmLosses": 2,
                    "confirmDraws": 0,
                },
            }),
        ]) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(handler, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(handler.TrainingWorkerManager, "is_active", lambda self: False)

    status = handler._build_training_status("ttt5")

    assert status["runId"] == "run-1"
    assert status["metricsHistory"] == [{"epoch": 1, "loss": 1.2, "acc": 55.0, "mae": 0.4}]
    assert status["winrateHistory"] == [{
        "cycle": 2,
        "winrate": 0.75,
        "decisiveWinRate": 0.5,
        "drawRate": 0.25,
        "winrateAsP1": 1.0,
        "winrateAsP2": 0.5,
        "balancedSideWinrate": 0.5,
        "tacticalOverrideRate": None,
        "valueGuidedRate": None,
        "modelPolicyRate": None,
        "wins": 3,
        "losses": 1,
        "draws": 0,
    }]
    assert status["confirmWinrateHistory"] == [{
        "cycle": 3,
        "winrate": 0.8,
        "decisiveWinRate": 0.7,
        "drawRate": 0.1,
        "winrateAsP1": 1.0,
        "winrateAsP2": 0.6,
        "balancedSideWinrate": 0.6,
        "tacticalOverrideRate": None,
        "valueGuidedRate": None,
        "modelPolicyRate": None,
        "wins": 8,
        "losses": 2,
        "draws": 0,
    }]
