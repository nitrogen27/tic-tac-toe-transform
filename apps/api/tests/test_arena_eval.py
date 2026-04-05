from __future__ import annotations

import torch

from gomoku_api.ws import arena_eval


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        logits = torch.zeros((batch, 256), dtype=torch.float32, device=x.device) + (self.anchor * 0).view(1, 1)
        value = torch.zeros((batch, 1), dtype=torch.float32, device=x.device) + (self.anchor * 0).view(1, 1)
        return logits, value


def test_arena_result_reports_decisive_and_draw_rates() -> None:
    result = arena_eval.ArenaResult(wins_a=3, wins_b=1, draws=2, total=6)

    assert result.winrate_a == 4 / 6
    assert result.decisive_winrate_a == 3 / 6
    assert result.draw_rate == 2 / 6
    payload = result.to_dict()
    assert payload["decisiveWinrateA"] == round(3 / 6, 4)
    assert payload["drawRate"] == round(2 / 6, 4)


def test_model_greedy_move_uses_shared_loaded_model_decision(monkeypatch) -> None:
    board = [0] * 25
    model = DummyModel()
    calls: list[tuple[int, int, str]] = []

    def fake_loaded_model_decision(board_arg, current_arg, board_size_arg, win_len_arg, model_arg, *, decision_mode="hybrid"):
        calls.append((current_arg, board_size_arg, decision_mode))
        assert board_arg == board
        assert model_arg is model
        return {"move": 12}

    monkeypatch.setattr(arena_eval, "_loaded_model_decision", fake_loaded_model_decision)

    move = arena_eval._model_greedy_move(
        board,
        board_size=5,
        win_len=4,
        current=1,
        model=model,
        device=torch.device("cpu"),
    )

    assert move == 12
    assert calls == [(1, 5, "hybrid")]
