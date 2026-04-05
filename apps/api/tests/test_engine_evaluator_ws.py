from __future__ import annotations

import pytest

from gomoku_api.ws.engine_evaluator import EngineEvaluator


@pytest.mark.asyncio
async def test_best_move_with_value_parses_engine_payload() -> None:
    evaluator = EngineEvaluator(binary_path="missing")

    async def fake_send_request(payload):
        return {"bestMove": 7, "value": 1.25}

    class DummyProcess:
        returncode = None

    evaluator._send_request = fake_send_request  # type: ignore[method-assign]
    evaluator._process = DummyProcess()  # type: ignore[assignment]

    move, value = await evaluator.best_move_with_value([0] * 25, 1, 5, 4)

    assert move == 7
    assert value == 1.0
