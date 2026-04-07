from __future__ import annotations

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
