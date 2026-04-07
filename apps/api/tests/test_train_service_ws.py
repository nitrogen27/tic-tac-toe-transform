from __future__ import annotations

import random

import pytest

from gomoku_api.ws.train_service_ws import (
    _build_repair_pool,
    _build_turbo_pool,
    _build_train_pool,
    _checkpoint_selection_score,
    _choose_rapid_cycle_strategy,
    _compute_target_sanity_metrics,
    _generate_engine_labeled_positions,
    _merge_failure_bank,
    _merge_position_bank,
    _generate_tactical_curriculum_positions,
    _load_offline_dataset_positions,
    _policy_cell_index,
    _resolve_engine_sampling_bounds,
    _sample_engine_position,
    _soft_policy_from_engine_hints,
    _variant_model_hparams,
)
from trainer_lab.config import ModelConfig


@pytest.mark.asyncio
async def test_generate_tactical_curriculum_positions_returns_one_hot_targets() -> None:
    events: list[dict] = []

    async def callback(event: dict) -> None:
        events.append(event)

    positions = await _generate_tactical_curriculum_positions(
        count=8,
        board_size=9,
        win_len=5,
        callback=callback,
    )

    assert len(positions) == 8
    assert any(event["type"] == "dataset.progress" for event in events)
    for pos in positions:
        assert sum(pos["policy"]) == pytest.approx(1.0, abs=1e-6)
        move = max(range(len(pos["policy"])), key=pos["policy"].__getitem__)
        assert pos["policy"][move] == 1.0


@pytest.mark.asyncio
async def test_generate_tactical_curriculum_positions_respects_motif_filter() -> None:
    async def callback(_event: dict) -> None:
        return None

    positions = await _generate_tactical_curriculum_positions(
        count=12,
        board_size=5,
        win_len=4,
        callback=callback,
        motif_filter="block",
    )

    assert len(positions) == 12
    assert all(pos["motif"] == "block" for pos in positions)


def test_variant_model_hparams_scale_up_ttt5() -> None:
    cfg = ModelConfig()
    ttt5_profile, ttt5 = _variant_model_hparams(5, cfg, variant="ttt5")
    gomoku15_profile, gomoku15 = _variant_model_hparams(15, cfg, variant="gomoku15")

    assert ttt5_profile == "standard"
    assert gomoku15_profile == "standard"
    assert ttt5 == (96, 8, 160)
    assert gomoku15[0] >= 128
    assert gomoku15[1] >= 8


def test_variant_model_hparams_supports_ttt5_small_profile() -> None:
    cfg = ModelConfig()
    profile, ttt5_small = _variant_model_hparams(5, cfg, variant="ttt5", model_profile="small")

    assert profile == "small"
    assert ttt5_small == (64, 6, 128)


def test_policy_cell_index_maps_inside_padded_grid() -> None:
    assert _policy_cell_index(0, 5) == 0
    assert _policy_cell_index(6, 5) == 17


def test_build_train_pool_keeps_seed_samples() -> None:
    latest = [{"id": f"latest-{idx}"} for idx in range(8)]
    replay = [{"id": f"replay-{idx}"} for idx in range(8)]
    seed = [{"id": f"seed-{idx}"} for idx in range(4)]

    pool = _build_train_pool(latest, replay, data_count=10, seed_positions=seed)

    ids = {item["id"] for item in pool}
    assert len(pool) == 10
    assert any(item.startswith("latest-") for item in ids)
    assert any(item.startswith("seed-") for item in ids)


def test_merge_failure_bank_deduplicates_and_caps() -> None:
    pos_a = {"board_size": 5, "board": [[0] * 5 for _ in range(5)], "current_player": 1, "last_move": None}
    pos_b = {"board_size": 5, "board": [[0, 1, 0, 0, 0]] + [[0] * 5 for _ in range(4)], "current_player": 2, "last_move": [0, 1]}
    merged = _merge_failure_bank([pos_a], [pos_a, pos_b], max_size=2)

    assert len(merged) == 2
    assert merged[-1]["current_player"] == 2


def test_merge_position_bank_deduplicates_d4_equivalents_and_increases_weight() -> None:
    base = {
        "board_size": 5,
        "board": [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "current_player": 1,
        "last_move": [1, 1],
        "policy": [0.0] * 256,
        "value": 0.5,
        "source": "engine",
    }
    base["policy"][2] = 1.0

    rotated = {
        "board_size": 5,
        "board": [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "current_player": 1,
        "last_move": [1, 3],
        "policy": [0.0] * 256,
        "value": 0.75,
        "source": "engine",
    }
    rotated["policy"][13] = 1.0

    merged = _merge_position_bank([base], [rotated], max_size=8)

    assert len(merged) == 1
    assert merged[0]["mergeCount"] == 2
    assert merged[0]["sampleWeight"] > 1.0


def test_merge_position_bank_keeps_policy_target_legal_after_d4_canonicalization() -> None:
    base = {
        "board_size": 5,
        "board": [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "current_player": 1,
        "last_move": [1, 3],
        "policy": [0.0] * 256,
        "value": 0.5,
        "source": "engine",
    }
    base["policy"][13] = 1.0

    merged = _merge_position_bank([], [base], max_size=8)
    sanity = _compute_target_sanity_metrics(merged)

    assert len(merged) == 1
    assert sanity["legalTargetRate"] == 100.0


def test_build_repair_pool_prioritizes_failure_samples() -> None:
    failure = [{"id": f"failure-{idx}"} for idx in range(10)]
    anchor = [{"id": f"anchor-{idx}"} for idx in range(10)]
    tactical = [{"id": f"tactical-{idx}"} for idx in range(10)]

    pool = _build_repair_pool(failure, anchor, tactical, data_count=12)
    ids = [item["id"] for item in pool]

    assert len(pool) == 12
    assert sum(item.startswith("failure-") for item in ids) >= 6


def test_build_repair_pool_includes_user_corpus_positions() -> None:
    failure = [{"id": f"failure-{idx}", "sampleWeight": 1.0} for idx in range(10)]
    anchor = [{"id": f"anchor-{idx}", "sampleWeight": 1.0} for idx in range(10)]
    tactical = [{"id": f"tactical-{idx}", "sampleWeight": 1.0} for idx in range(10)]
    user = [{"id": f"user-{idx}", "sampleWeight": 1.5} for idx in range(10)]

    pool = _build_repair_pool(failure, anchor, tactical, user, data_count=20)
    ids = [item["id"] for item in pool]

    assert len(pool) == 20
    assert any(item.startswith("user-") for item in ids)


def test_build_turbo_pool_includes_user_corpus_positions() -> None:
    anchor = [{"id": f"anchor-{idx}", "sampleWeight": 1.0} for idx in range(20)]
    tactical = [{"id": f"tactical-{idx}", "sampleWeight": 1.0} for idx in range(20)]
    failure = [{"id": f"failure-{idx}", "sampleWeight": 1.0} for idx in range(20)]
    user = [{"id": f"user-{idx}", "sampleWeight": 1.5} for idx in range(20)]

    pool = _build_turbo_pool(anchor, tactical, failure, user, data_count=24, tactical_ratio=0.20)
    ids = [item["id"] for item in pool]

    assert len(pool) == 24
    assert any(item.startswith("user-") for item in ids)


def test_resolve_engine_sampling_bounds_biases_mid_and_late_for_ttt5() -> None:
    total_cells = 25

    assert _resolve_engine_sampling_bounds(5, total_cells, phase_focus="mid") == (9, 16)
    assert _resolve_engine_sampling_bounds(5, total_cells, phase_focus="late") == (17, 24)


def test_sample_engine_position_with_mid_focus_stays_in_mid_bucket() -> None:
    rng = random.Random(7)

    sampled = None
    for _ in range(64):
        sampled = _sample_engine_position(5, 4, rng=rng, phase_focus="mid")
        if sampled is not None:
            break

    assert sampled is not None
    board, _current, _last_move = sampled
    occupied = sum(1 for cell in board if cell != 0)
    assert 9 <= occupied <= 16


def test_choose_rapid_cycle_strategy_biases_engine_generation_toward_midgame() -> None:
    strategy = _choose_rapid_cycle_strategy(
        {
            "frozenBlockAcc": 92.0,
            "frozenWinAcc": 95.0,
            "frozenMidAcc": 48.0,
            "frozenLateAcc": 68.0,
            "holdoutDeltaAcc": 0.5,
        },
        corrected_rate=0.12,
        failure_bank_size=16,
        engine_per_cycle=50,
    )

    assert strategy["engineFocus"] == "mid"
    assert strategy["engineCount"] > 50
    assert strategy["tacticalRatio"] <= 0.35
    assert strategy["failureSlice"] == 384


def test_choose_rapid_cycle_strategy_biases_conversion_and_weak_side() -> None:
    strategy = _choose_rapid_cycle_strategy(
        {
            "frozenBlockAcc": 94.0,
            "frozenWinAcc": 96.0,
            "frozenMidAcc": 61.0,
            "frozenLateAcc": 67.0,
            "holdoutDeltaAcc": 0.0,
        },
        corrected_rate=0.30,
        failure_bank_size=12,
        engine_per_cycle=50,
        exam_summary={
            "winrateAsP1": 0.25,
            "winrateAsP2": 0.75,
            "decisiveWinRate": 0.05,
            "drawRate": 0.55,
        },
    )

    assert strategy["engineCurrentPlayerFocus"] == 1
    assert strategy["conversionFocus"] is True
    assert strategy["engineCount"] > 50
    assert strategy["tacticalRatio"] <= 0.32


@pytest.mark.asyncio
async def test_generate_engine_labeled_positions_can_focus_side_and_conversion(monkeypatch) -> None:
    samples = [
        ([0] * 25, 1, -1),
        ([0] * 25, 2, -1),
        ([0] * 25, 2, -1),
    ]

    def fake_sample_engine_position(board_size: int, win_len: int, *, rng=None, phase_focus=None):
        if not samples:
            return None
        board, current, last_move = samples.pop(0)
        return list(board), current, last_move

    class FakeEngine:
        async def analyze_position(self, board, current, board_size, win_len):
            move = next(idx for idx, cell in enumerate(board) if cell == 0)
            value = 0.8 if current == 2 else 0.1
            return {"bestMove": move, "value": value}

        async def suggest_moves(self, board, current, board_size, win_len, *, top_n=5):
            move = next(idx for idx, cell in enumerate(board) if cell == 0)
            return [{"move": move, "score": 1.0}]

    async def callback(_event: dict) -> None:
        return None

    monkeypatch.setattr("gomoku_api.ws.train_service_ws._sample_engine_position", fake_sample_engine_position)

    positions = await _generate_engine_labeled_positions(
        2,
        5,
        4,
        callback,
        FakeEngine(),
        current_player_focus=2,
        min_value=0.45,
        source="engine_conversion",
        boost_weight=1.35,
    )

    assert len(positions) == 2
    assert all(pos["current_player"] == 2 for pos in positions)
    assert all(pos["value"] >= 0.45 for pos in positions)
    assert all(pos["source"] == "engine_conversion" for pos in positions)
    assert all(pos["sampleWeight"] == pytest.approx(1.35) for pos in positions)


def test_soft_policy_from_engine_hints_creates_distribution_not_one_hot() -> None:
    board = [0] * 25
    hints = [
        {"move": 12, "score": 10.0},
        {"move": 7, "score": 9.0},
        {"move": 17, "score": 8.5},
    ]

    policy = _soft_policy_from_engine_hints(12, board, 5, hints)

    assert abs(sum(policy) - 1.0) < 1e-6
    assert policy[(12 // 5) * 16 + (12 % 5)] > policy[(17 // 5) * 16 + (17 % 5)] > 0.0
    assert sum(1 for value in policy if value > 0) >= 3


def test_checkpoint_selection_score_prefers_decisive_wins_over_draw_heavy_line() -> None:
    draw_heavy = {
        "winrate": 0.25,
        "decisiveWinRate": 0.0,
        "drawRate": 0.5,
    }
    winning = {
        "winrate": 0.5,
        "decisiveWinRate": 0.5,
        "drawRate": 0.0,
    }

    assert _checkpoint_selection_score(winning, 9) > _checkpoint_selection_score(draw_heavy, 9)


def test_checkpoint_selection_score_uses_side_balance_as_tiebreaker() -> None:
    skewed = {
        "winrate": 0.5,
        "decisiveWinRate": 0.5,
        "drawRate": 0.0,
        "winrateAsP1": 0.75,
        "winrateAsP2": 0.25,
    }
    balanced = {
        "winrate": 0.5,
        "decisiveWinRate": 0.5,
        "drawRate": 0.0,
        "winrateAsP1": 0.5,
        "winrateAsP2": 0.5,
    }

    assert _checkpoint_selection_score(balanced, 9) > _checkpoint_selection_score(skewed, 9)


def test_load_offline_dataset_positions_prefers_rapfi_dataset(tmp_path, monkeypatch) -> None:
    datasets_dir = tmp_path / "saved" / "datasets"
    datasets_dir.mkdir(parents=True)
    (datasets_dir / "gomoku15_engine.json").write_text('[{"source":"engine"}]', encoding="utf-8")
    (datasets_dir / "gomoku15_rapfi.json").write_text('[{"source":"rapfi"}]', encoding="utf-8")

    monkeypatch.setattr("gomoku_api.ws.train_service_ws.SAVED_DIR", tmp_path / "saved")

    positions, path, dataset_type = _load_offline_dataset_positions("gomoku15")

    assert dataset_type == "rapfi"
    assert path is not None and path.name == "gomoku15_rapfi.json"
    assert positions[0]["source"] == "rapfi"


def test_load_offline_dataset_positions_honors_preferred_backend(tmp_path, monkeypatch) -> None:
    datasets_dir = tmp_path / "saved" / "datasets"
    datasets_dir.mkdir(parents=True)
    (datasets_dir / "gomoku15_engine.json").write_text('[{"source":"engine"}]', encoding="utf-8")
    (datasets_dir / "gomoku15_rapfi.json").write_text('[{"source":"rapfi"}]', encoding="utf-8")

    monkeypatch.setattr("gomoku_api.ws.train_service_ws.SAVED_DIR", tmp_path / "saved")

    positions, path, dataset_type = _load_offline_dataset_positions("gomoku15", preferred_backend="builtin")

    assert dataset_type == "engine"
    assert path is not None and path.name == "gomoku15_engine.json"
    assert positions[0]["source"] == "engine"
