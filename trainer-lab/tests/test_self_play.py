"""Tests for self-play pipeline components."""

from __future__ import annotations

import json

import pytest
import torch

from trainer_lab.config import ModelConfig, SelfPlayConfig
from trainer_lab.models.resnet import PolicyValueResNet
from trainer_lab.self_play.player import GameState, MCTSNode, SelfPlayPlayer, mcts_search
from trainer_lab.self_play.replay_buffer import ReplayBuffer

# Small model config for fast tests
SMALL_CFG = ModelConfig(res_filters=32, res_blocks=2, policy_filters=2, value_fc=32)


def _make_small_model() -> PolicyValueResNet:
    return PolicyValueResNet(
        in_channels=SMALL_CFG.in_channels,
        res_filters=SMALL_CFG.res_filters,
        res_blocks=SMALL_CFG.res_blocks,
        policy_filters=SMALL_CFG.policy_filters,
        value_fc=SMALL_CFG.value_fc,
        board_max=SMALL_CFG.board_max,
    )


# ---------------------------------------------------------------------------
# GameState tests
# ---------------------------------------------------------------------------


class TestGameState:
    def test_initial_legal_moves(self):
        state = GameState(15)
        assert len(state.legal_moves()) == 225

    def test_initial_legal_moves_7x7(self):
        state = GameState(7)
        assert len(state.legal_moves()) == 49

    def test_apply_move(self):
        state = GameState(15)
        new = state.apply_move(112)  # center
        assert new.board[7][7] == 1
        assert new.current_player == 2
        assert new.move_count == 1
        assert len(new.legal_moves()) == 224

    def test_not_terminal_initially(self):
        state = GameState(15)
        terminal, winner = state.is_terminal()
        assert not terminal
        assert winner == 0

    def test_horizontal_win(self):
        state = GameState(15)
        # Place 5 in a row for player 1 at row 7
        for c in range(5):
            state = state.apply_move(7 * 15 + c)  # player 1
            if c < 4:
                state = state.apply_move(8 * 15 + c)  # player 2
        terminal, winner = state.is_terminal()
        assert terminal
        assert winner == 1

    def test_vertical_win(self):
        state = GameState(7)
        # Player 1 fills column 0, player 2 fills column 1
        for r in range(5):
            state = state.apply_move(r * 7 + 0)  # p1 col 0
            if r < 4:
                state = state.apply_move(r * 7 + 1)  # p2 col 1
        terminal, winner = state.is_terminal()
        assert terminal
        assert winner == 1

    def test_draw_on_full_board(self):
        # Fill a 3x3 board (win_length=5 so no one can win)
        state = GameState(3)
        for i in range(9):
            state = state.apply_move(i)
        terminal, winner = state.is_terminal()
        assert terminal
        assert winner == 0

    def test_encoder_dict(self):
        state = GameState(15)
        state = state.apply_move(112)
        d = state.to_encoder_dict()
        assert d["board_size"] == 15
        assert d["current_player"] == 2
        assert d["board"][7][7] == 1


# ---------------------------------------------------------------------------
# MCTSNode tests
# ---------------------------------------------------------------------------


class TestMCTSNode:
    def test_leaf_initially(self):
        node = MCTSNode(parent=None, move=-1, prior=1.0)
        assert node.is_leaf()
        assert node.visit_count == 0

    def test_q_value_zero_initially(self):
        node = MCTSNode(parent=None, move=0, prior=0.5)
        assert node.q_value == 0.0

    def test_q_value_after_visits(self):
        node = MCTSNode(parent=None, move=0, prior=0.5)
        node.visit_count = 4
        node.value_sum = 2.0
        assert node.q_value == 0.5

    def test_ucb_score_with_prior(self):
        parent = MCTSNode(parent=None, move=-1, prior=1.0)
        parent.visit_count = 100
        child = MCTSNode(parent=parent, move=0, prior=0.5)
        child.visit_count = 0
        # UCB = 0 + 1.5 * 0.5 * sqrt(100) / 1 = 7.5
        assert child.ucb_score(c_puct=1.5) == pytest.approx(7.5, abs=0.01)

    def test_best_child(self):
        parent = MCTSNode(parent=None, move=-1, prior=1.0)
        parent.visit_count = 10
        c1 = MCTSNode(parent=parent, move=0, prior=0.1)
        c2 = MCTSNode(parent=parent, move=1, prior=0.9)
        parent.children = [c1, c2]
        assert parent.best_child().move == 1  # higher prior → higher UCB initially


# ---------------------------------------------------------------------------
# MCTS search smoke test
# ---------------------------------------------------------------------------


class TestMCTSSearch:
    def test_returns_valid_policy(self):
        model = _make_small_model()
        state = GameState(7)
        policy, value = mcts_search(state, model, torch.device("cpu"), num_simulations=8)
        assert len(policy) == 49
        assert sum(policy) == pytest.approx(1.0, abs=0.01)
        assert all(p >= 0 for p in policy)


# ---------------------------------------------------------------------------
# SelfPlayPlayer smoke test
# ---------------------------------------------------------------------------


class TestSelfPlayPlayer:
    def test_play_game_returns_positions(self):
        model = _make_small_model()
        cfg = SelfPlayConfig(simulations=4, games=1)
        player = SelfPlayPlayer(model, cfg, torch.device("cpu"))
        positions = player.play_game(board_size=7)
        assert len(positions) > 0
        pos = positions[0]
        assert "board_size" in pos
        assert "board" in pos
        assert "policy" in pos
        assert "value" in pos
        assert len(pos["policy"]) == 49


# ---------------------------------------------------------------------------
# ReplayBuffer tests
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(SelfPlayConfig(replay_buffer_max=100))
        for i in range(10):
            buf.add({"id": i})
        assert len(buf) == 10

    def test_overflow(self):
        buf = ReplayBuffer(SelfPlayConfig(replay_buffer_max=50))
        for i in range(100):
            buf.add({"id": i})
        assert len(buf) == 50

    def test_sample_returns_correct_count(self):
        buf = ReplayBuffer(SelfPlayConfig(replay_buffer_max=100))
        for i in range(20):
            buf.add({"id": i})
        sampled = buf.sample(5)
        assert len(sampled) == 5

    def test_save_and_load(self, tmp_path):
        buf1 = ReplayBuffer(SelfPlayConfig(replay_buffer_max=100))
        for i in range(10):
            buf1.add({"id": i, "value": float(i)})
        path = tmp_path / "buf.json"
        buf1.save(path)

        buf2 = ReplayBuffer(SelfPlayConfig(replay_buffer_max=100))
        buf2.load(path)
        assert len(buf2) == 10

    def test_load_nonexistent_file(self, tmp_path):
        buf = ReplayBuffer(SelfPlayConfig(replay_buffer_max=100))
        buf.load(tmp_path / "nonexistent.json")
        assert len(buf) == 0
