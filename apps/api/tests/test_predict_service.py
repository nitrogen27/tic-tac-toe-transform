from __future__ import annotations

import torch

from gomoku_api.ws import predict_service


def _policy_index(move: int, board_size: int = 5) -> int:
    row, col = divmod(move, board_size)
    return row * 16 + col


class DummyModel(torch.nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self._logits = torch.tensor(logits, dtype=torch.float32).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self._logits.to(x.device) + (self.anchor * 0).view(1, 1)
        logits = logits.expand(x.shape[0], -1)
        value = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device) + (self.anchor * 0).view(1, 1)
        return logits, value


def test_model_predict_blocks_immediate_loss(monkeypatch) -> None:
    board = [
        2, 0, 0, 0, 0,
        2, 0, 0, 0, 0,
        2, 0, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ]
    logits = [-8.0] * 256
    logits[_policy_index(24)] = 10.0  # model prefers a bad move
    model = DummyModel(logits)
    monkeypatch.setattr(predict_service, "_get_model", lambda variant: model)

    result = predict_service._model_predict(board, current=1, variant="ttt5", board_size=5)

    assert result["move"] == 15
    assert result["tacticalReason"] == "block_immediate"
    assert result["tacticalOverride"] is True
    assert result["opponentThreatsBefore"] >= 1


def test_model_predict_prefers_immediate_win_over_anything_else(monkeypatch) -> None:
    board = [
        1, 1, 1, 0, 0,
        2, 2, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ]
    logits = [-8.0] * 256
    logits[_policy_index(24)] = 10.0
    model = DummyModel(logits)
    monkeypatch.setattr(predict_service, "_get_model", lambda variant: model)

    result = predict_service._model_predict(board, current=1, variant="ttt5", board_size=5)

    assert result["move"] == 3
    assert result["tacticalReason"] == "immediate_win"
    assert result["tacticalOverride"] is True


def test_model_predict_creates_forcing_threat_when_policy_prefers_quiet_move(monkeypatch) -> None:
    board = [
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,
    ]
    logits = [-8.0] * 256
    logits[_policy_index(24)] = 10.0
    model = DummyModel(logits)
    monkeypatch.setattr(predict_service, "_get_model", lambda variant: model)

    result = predict_service._model_predict(board, current=1, variant="ttt5", board_size=5)

    assert result["move"] == 12
    assert result["tacticalReason"] == "create_forcing_threat"
    assert result["tacticalOverride"] is True
    assert result["forcingThreatsAfterMove"] >= 2
