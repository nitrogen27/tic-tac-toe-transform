from __future__ import annotations

import torch
from pathlib import Path

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


def test_model_predict_uses_value_head_when_safe_moves_are_tactically_equal(monkeypatch) -> None:
    board = [
        1, 0, 0, 0, 0,
        0, 2, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 2, 0,
        0, 0, 0, 0, 1,
    ]
    logits = [-8.0] * 256
    logits[_policy_index(1)] = 10.0   # policy prefers move 1
    logits[_policy_index(12)] = 6.0   # but value will prefer center move 12
    model = DummyModel(logits)
    monkeypatch.setattr(predict_service, "_get_model", lambda variant: model)
    monkeypatch.setattr(
        predict_service,
        "_evaluate_afterstate_values",
        lambda model, board, current, board_size, candidate_moves: {move: (0.9 if move == 12 else -0.2) for move in candidate_moves},
    )

    result = predict_service._model_predict(board, current=1, variant="ttt5", board_size=5)

    assert result["move"] == 12
    assert result["tacticalReason"] == "policy_value"
    assert result["valueGuided"] is True
    assert result["afterstateValue"] == 0.9


def test_model_predict_pure_mode_skips_hybrid_tactical_override(monkeypatch) -> None:
    board = [
        2, 0, 0, 0, 0,
        2, 0, 0, 0, 0,
        2, 0, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ]
    logits = [-8.0] * 256
    logits[_policy_index(24)] = 10.0
    logits[_policy_index(15)] = 1.0
    model = DummyModel(logits)
    monkeypatch.setattr(predict_service, "_get_model", lambda variant: model)
    monkeypatch.setattr(
        predict_service,
        "_evaluate_afterstate_values",
        lambda model, board, current, board_size, candidate_moves: {move: (0.9 if move == 24 else -0.1) for move in candidate_moves},
    )

    result = predict_service._model_predict(board, current=1, variant="ttt5", board_size=5, decision_mode="pure")

    assert result["move"] == 24
    assert result["decisionMode"] == "pure"
    assert result["tacticalReason"] in {"model_policy", "policy_value"}
    assert result["tacticalOverride"] is False


def test_model_predict_pure_mode_without_checkpoint_does_not_use_strong_tactical_fallback(monkeypatch) -> None:
    board = [
        2, 0, 0, 0, 0,
        2, 0, 0, 0, 0,
        2, 0, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ]
    monkeypatch.setattr(predict_service, "_get_model", lambda variant: None)

    result = predict_service._model_predict(board, current=1, variant="ttt5", board_size=5, decision_mode="pure")

    assert result["decisionMode"] == "pure"
    assert result["fallback"] is True
    assert result["tacticalReason"] == "no_model_checkpoint"
    assert result["move"] != 15


def test_get_model_falls_back_to_candidate_checkpoint(tmp_path, monkeypatch) -> None:
    saved_dir = tmp_path / "saved"
    variant_dir = saved_dir / "ttt5_resnet"
    variant_dir.mkdir(parents=True)

    from trainer_lab.config import ModelConfig
    from trainer_lab.models.resnet import PolicyValueResNet

    cfg = ModelConfig()
    model = PolicyValueResNet(
        in_channels=cfg.in_channels,
        res_filters=96,
        res_blocks=8,
        policy_filters=cfg.policy_filters,
        value_fc=160,
        board_max=cfg.board_max,
    )
    torch.save(model.state_dict(), variant_dir / "candidate.pt")

    monkeypatch.setattr(predict_service, "SAVED_DIR", Path(saved_dir))
    predict_service.clear_cached_model("ttt5")

    loaded = predict_service._get_model("ttt5")

    assert loaded is not None
