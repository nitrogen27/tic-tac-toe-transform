from __future__ import annotations

import random

import pytest

from gomoku_api.ws.train_service_ws import (
    _build_exact_ttt5_validation_pack,
    _build_pure_gap_relabel_candidate,
    _build_repair_pool,
    _build_turbo_pool,
    _build_train_pool,
    _build_frozen_benchmark_suites,
    _checkpoint_selection_score,
    _choose_rapid_cycle_strategy,
    _compute_target_sanity_metrics,
    _evaluate_decision_suite,
    _generate_engine_labeled_positions,
    _merge_failure_bank,
    _merge_position_bank,
    _position_bank_importance,
    _generate_tactical_curriculum_positions,
    _load_offline_dataset_positions,
    _policy_cell_index,
    _resolve_engine_sampling_bounds,
    _run_engine_exam,
    _sample_engine_position,
    _selfplay_mixed_source_weights,
    _soft_policy_from_engine_hints,
    _variant_model_hparams,
)
from trainer_lab.config import ModelConfig


def test_build_exact_ttt5_validation_pack_contains_legal_targets() -> None:
    positions = _build_exact_ttt5_validation_pack()

    assert len(positions) >= 8
    assert all(pos["source"] == "exact" for pos in positions)
    assert any(bool(pos.get("conversionTarget")) for pos in positions)
    for pos in positions:
        board = [cell for row in pos["board"] for cell in row]
        policy_index = next(i for i, weight in enumerate(pos["policy"]) if weight > 0.5)
        row, col = divmod(policy_index, 16)
        move = row * 5 + col
        assert board[move] == 0
        assert sum(pos["policy"]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_build_frozen_benchmark_suites_adds_exact_pack_for_ttt5() -> None:
    suites = await _build_frozen_benchmark_suites("ttt5", 5, 4, None)

    assert "exact" in suites
    assert len(suites["exact"]) >= 8


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


def test_selfplay_mixed_source_weights_shift_toward_self_play() -> None:
    early = _selfplay_mixed_source_weights(1, 10)
    late = _selfplay_mixed_source_weights(10, 10)

    assert sum(early.values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(late.values()) == pytest.approx(1.0, abs=1e-6)
    assert late["self_play"] > early["self_play"]
    assert late["anchor"] < early["anchor"]


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


def test_choose_rapid_cycle_strategy_responds_to_pure_gap_on_p1() -> None:
    strategy = _choose_rapid_cycle_strategy(
        {
            "frozenBlockAcc": 92.0,
            "frozenWinAcc": 95.0,
            "frozenMidAcc": 60.0,
            "frozenLateAcc": 65.0,
            "holdoutDeltaAcc": 0.0,
        },
        corrected_rate=0.30,
        failure_bank_size=12,
        engine_per_cycle=50,
        exam_summary={
            "winrateAsP1": 0.50,
            "winrateAsP2": 0.75,
            "decisiveWinRate": 0.15,
            "drawRate": 0.45,
            "pureGapRate": 0.35,
            "pureGapRateAsP1": 0.40,
            "pureGapRateAsP2": 0.10,
        },
    )

    assert strategy["engineCurrentPlayerFocus"] == 1
    assert strategy["conversionFocus"] is True
    assert strategy["failureSlice"] == 384
    assert strategy["engineCount"] > 50


def test_choose_rapid_cycle_strategy_uses_pure_frozen_recalls() -> None:
    strategy = _choose_rapid_cycle_strategy(
        {
            "frozenBlockAcc": 95.0,
            "frozenWinAcc": 95.0,
            "pureFrozenBlockRecall": 30.0,
            "pureFrozenWinRecall": 85.0,
            "frozenMidAcc": 62.0,
            "frozenLateAcc": 66.0,
            "holdoutDeltaAcc": 0.0,
        },
        corrected_rate=0.25,
        failure_bank_size=12,
        engine_per_cycle=50,
    )

    assert strategy["tacticalFocus"] == "block"
    assert strategy["tacticalRatio"] >= 0.65
    assert strategy["failureSlice"] >= 320


def test_choose_rapid_cycle_strategy_responds_to_exact_trap_recall() -> None:
    strategy = _choose_rapid_cycle_strategy(
        {
            "frozenBlockAcc": 95.0,
            "frozenWinAcc": 95.0,
            "pureFrozenBlockRecall": 92.0,
            "pureFrozenWinRecall": 92.0,
            "pureExactTrapRecall": 62.5,
            "frozenMidAcc": 62.0,
            "frozenLateAcc": 66.0,
            "holdoutDeltaAcc": 0.0,
        },
        corrected_rate=0.25,
        failure_bank_size=12,
        engine_per_cycle=50,
    )

    assert strategy["failureSlice"] >= 384
    assert strategy["tacticalRatio"] >= 0.65
    assert strategy["conversionFocus"] is True


def test_choose_rapid_cycle_strategy_enters_tactical_rescue_mode_on_zero_pure_win_recall() -> None:
    strategy = _choose_rapid_cycle_strategy(
        {
            "frozenBlockAcc": 95.0,
            "frozenWinAcc": 95.0,
            "pureFrozenBlockRecall": 68.75,
            "pureFrozenWinRecall": 0.0,
            "frozenMidAcc": 62.0,
            "frozenLateAcc": 66.0,
            "holdoutDeltaAcc": 0.0,
        },
        corrected_rate=0.25,
        failure_bank_size=12,
        engine_per_cycle=50,
    )

    assert strategy["tacticalFocus"] == "win"
    assert strategy["tacticalRatio"] >= 0.75
    assert strategy["failureSlice"] >= 448
    assert strategy["conversionFocus"] is True


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
    assert strategy["playerFocusRatio"] >= 0.45


def test_position_bank_importance_boosts_any_focused_side() -> None:
    p1 = _position_bank_importance({"sampleWeight": 1.0, "playerFocus": 1})
    p2 = _position_bank_importance({"sampleWeight": 1.0, "playerFocus": 2})
    neutral = _position_bank_importance({"sampleWeight": 1.0, "playerFocus": 0})

    assert p1 > neutral
    assert p2 > neutral
    assert p1 == pytest.approx(p2)


def test_position_bank_importance_boosts_engine_side_and_conversion_sources() -> None:
    base = _position_bank_importance({"sampleWeight": 1.0, "source": "engine"})
    side_focus = _position_bank_importance({"sampleWeight": 1.0, "source": "engine_side_focus"})
    conversion = _position_bank_importance({"sampleWeight": 1.0, "source": "engine_conversion"})
    side_conversion = _position_bank_importance({"sampleWeight": 1.0, "source": "engine_side_conversion"})

    assert side_focus > base
    assert conversion > side_focus
    assert side_conversion > conversion


def test_build_turbo_pool_respects_focus_player_ratio() -> None:
    anchor = [{"id": f"a1-{i}", "playerFocus": 1, "sampleWeight": 10.0} for i in range(30)]
    anchor.extend({"id": f"a2-{i}", "playerFocus": 2, "sampleWeight": 1.0} for i in range(30))
    tactical = [{"id": f"t1-{i}", "playerFocus": 1, "sampleWeight": 10.0} for i in range(12)]
    tactical.extend({"id": f"t2-{i}", "playerFocus": 2, "sampleWeight": 1.0} for i in range(12))
    failure = [{"id": f"f1-{i}", "playerFocus": 1, "sampleWeight": 10.0} for i in range(12)]
    failure.extend({"id": f"f2-{i}", "playerFocus": 2, "sampleWeight": 1.0} for i in range(12))

    pool = _build_turbo_pool(
        anchor,
        tactical,
        failure,
        data_count=24,
        tactical_ratio=0.25,
        focus_player=2,
        focus_ratio=0.5,
    )

    focused = sum(1 for pos in pool if int(pos.get("playerFocus", 0) or 0) == 2)
    assert focused >= 8


def test_build_turbo_pool_reserves_side_conversion_slice() -> None:
    anchor = [{"id": f"a1-{i}", "playerFocus": 1, "sampleWeight": 1.0} for i in range(40)]
    anchor.extend(
        {"id": f"ac2-{i}", "playerFocus": 2, "conversionTarget": True, "source": "engine_side_conversion", "sampleWeight": 2.0}
        for i in range(12)
    )
    tactical = [{"id": f"t1-{i}", "playerFocus": 1, "sampleWeight": 1.0} for i in range(12)]
    failure = [{"id": f"f1-{i}", "playerFocus": 1, "sampleWeight": 1.0} for i in range(12)]

    pool = _build_turbo_pool(
        anchor,
        tactical,
        failure,
        data_count=24,
        tactical_ratio=0.20,
        focus_player=2,
        focus_ratio=0.45,
        focus_conversion_ratio=0.25,
    )

    focus_conversion = sum(
        1 for pos in pool
        if int(pos.get("playerFocus", 0) or 0) == 2 and bool(pos.get("conversionTarget"))
    )
    assert focus_conversion >= 4


def test_build_turbo_pool_keeps_counter_side_conversion_slice() -> None:
    anchor = [
        {"id": f"a1c-{i}", "playerFocus": 1, "conversionTarget": True, "source": "engine_conversion", "sampleWeight": 1.5}
        for i in range(12)
    ]
    anchor.extend(
        {"id": f"a2c-{i}", "playerFocus": 2, "conversionTarget": True, "source": "engine_side_conversion", "sampleWeight": 2.0}
        for i in range(12)
    )
    tactical = [{"id": f"t-{i}", "playerFocus": 1, "sampleWeight": 1.0} for i in range(12)]
    failure = [{"id": f"f-{i}", "playerFocus": 2, "sampleWeight": 1.0} for i in range(12)]

    pool = _build_turbo_pool(
        anchor,
        tactical,
        failure,
        data_count=24,
        tactical_ratio=0.20,
        focus_player=2,
        focus_ratio=0.45,
        focus_conversion_ratio=0.25,
        counter_conversion_ratio=0.10,
    )

    counter_conversion = sum(
        1 for pos in pool
        if int(pos.get("playerFocus", 0) or 0) == 1 and bool(pos.get("conversionTarget"))
    )
    assert counter_conversion >= 2


def test_build_repair_pool_respects_focus_player_ratio() -> None:
    failure = [{"id": f"f1-{i}", "playerFocus": 1, "sampleWeight": 10.0} for i in range(40)]
    failure.extend({"id": f"f2-{i}", "playerFocus": 2, "sampleWeight": 1.0} for i in range(40))
    anchor = [{"id": f"a1-{i}", "playerFocus": 1, "sampleWeight": 10.0} for i in range(20)]
    anchor.extend({"id": f"a2-{i}", "playerFocus": 2, "sampleWeight": 1.0} for i in range(20))
    tactical = [{"id": f"t1-{i}", "playerFocus": 1, "sampleWeight": 10.0} for i in range(16)]
    tactical.extend({"id": f"t2-{i}", "playerFocus": 2, "sampleWeight": 1.0} for i in range(16))

    pool = _build_repair_pool(
        failure,
        anchor,
        tactical,
        data_count=32,
        focus_player=2,
        focus_ratio=0.5,
    )

    focused = sum(1 for pos in pool if int(pos.get("playerFocus", 0) or 0) == 2)
    assert focused >= 12


def test_build_repair_pool_reserves_side_conversion_slice() -> None:
    failure = [{"id": f"f1-{i}", "playerFocus": 1, "sampleWeight": 1.0} for i in range(40)]
    failure.extend(
        {"id": f"fc2-{i}", "playerFocus": 2, "conversionTarget": True, "source": "failure_conversion", "sampleWeight": 2.0}
        for i in range(12)
    )
    anchor = [{"id": f"a1-{i}", "playerFocus": 1, "sampleWeight": 1.0} for i in range(20)]
    tactical = [{"id": f"t1-{i}", "playerFocus": 1, "sampleWeight": 1.0} for i in range(16)]

    pool = _build_repair_pool(
        failure,
        anchor,
        tactical,
        data_count=32,
        focus_player=2,
        focus_ratio=0.5,
        focus_conversion_ratio=0.25,
    )

    focus_conversion = sum(
        1 for pos in pool
        if int(pos.get("playerFocus", 0) or 0) == 2 and bool(pos.get("conversionTarget"))
    )
    assert focus_conversion >= 6


def test_choose_rapid_cycle_strategy_uses_side_conversion_failures() -> None:
    strategy = _choose_rapid_cycle_strategy(
        {
            "frozenBlockAcc": 94.0,
            "frozenWinAcc": 96.0,
            "frozenMidAcc": 61.0,
            "frozenLateAcc": 67.0,
            "holdoutDeltaAcc": 0.0,
        },
        corrected_rate=0.30,
        failure_bank_size=24,
        engine_per_cycle=50,
        exam_summary={
            "winrateAsP1": 0.50,
            "winrateAsP2": 0.50,
            "balancedSideWinrate": 0.10,
            "decisiveWinRate": 0.05,
            "drawRate": 0.50,
            "conversionFailuresAsP1": 3,
            "conversionFailuresAsP2": 11,
        },
    )

    assert strategy["engineCurrentPlayerFocus"] == 2
    assert strategy["conversionFocus"] is True
    assert strategy["playerFocusRatio"] >= 0.55
    assert strategy["focusConversionRatio"] >= 0.30
    assert strategy["counterConversionRatio"] > 0.0
    assert strategy["failureSlice"] >= 384


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


def test_evaluate_decision_suite_collects_pure_win_failures(monkeypatch) -> None:
    position = {
        "board_size": 5,
        "board": [
            [1, 1, 1, 0, 0],
            [2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "current_player": 1,
        "last_move": [1, 1],
        "policy": [0.0] * 256,
        "value": 1.0,
        "motif": "win",
        "sampleWeight": 1.0,
    }
    position["policy"][_policy_cell_index(3, 5)] = 1.0

    def fake_loaded_model_decision(*args, **kwargs):
        return {"move": 24}

    monkeypatch.setattr("gomoku_api.ws.predict_service._loaded_model_decision", fake_loaded_model_decision)

    metrics, failures = _evaluate_decision_suite(
        model=None,
        positions=[position],
        board_size=5,
        win_len=4,
        decision_mode="pure",
        suite_name="win",
        collect_failures=True,
    )

    assert metrics["total"] == 1
    assert metrics["correct"] == 0
    assert failures[0]["motif"] == "pure_missed_win"
    assert failures[0]["source"] == "failure_pure_gap"
    assert failures[0]["pureMissedWinInOne"] is True


def test_evaluate_decision_suite_collects_pure_block_failures(monkeypatch) -> None:
    position = {
        "board_size": 5,
        "board": [
            [2, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [2, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "current_player": 1,
        "last_move": [2, 2],
        "policy": [0.0] * 256,
        "value": 0.2,
        "motif": "block",
        "sampleWeight": 1.0,
    }
    position["policy"][_policy_cell_index(15, 5)] = 1.0

    def fake_loaded_model_decision(*args, **kwargs):
        return {"move": 24}

    monkeypatch.setattr("gomoku_api.ws.predict_service._loaded_model_decision", fake_loaded_model_decision)

    metrics, failures = _evaluate_decision_suite(
        model=None,
        positions=[position],
        board_size=5,
        win_len=4,
        decision_mode="pure",
        suite_name="block",
        collect_failures=True,
    )

    assert metrics["total"] == 1
    assert metrics["correct"] == 0
    assert failures[0]["motif"] == "pure_missed_block"
    assert failures[0]["source"] == "failure_pure_gap"
    assert failures[0]["pureMissedBlockInOne"] is True


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


def test_checkpoint_selection_score_prefers_lower_pure_gap_on_equal_hybrid_strength() -> None:
    higher_gap = {
        "winrate": 0.75,
        "decisiveWinRate": 0.5,
        "drawRate": 0.25,
        "winrateAsP1": 0.75,
        "winrateAsP2": 0.75,
        "pureGapRate": 0.45,
        "pureGapRateAsP1": 0.50,
        "pureGapRateAsP2": 0.40,
    }
    lower_gap = {
        "winrate": 0.75,
        "decisiveWinRate": 0.5,
        "drawRate": 0.25,
        "winrateAsP1": 0.75,
        "winrateAsP2": 0.75,
        "pureGapRate": 0.15,
        "pureGapRateAsP1": 0.10,
        "pureGapRateAsP2": 0.20,
    }

    assert _checkpoint_selection_score(lower_gap, 9) > _checkpoint_selection_score(higher_gap, 9)


def test_checkpoint_selection_score_prefers_stronger_challenger_vs_champion() -> None:
    weaker_vs_champion = {
        "winrate": 0.80,
        "decisiveWinRate": 0.60,
        "drawRate": 0.20,
        "winrateAsP1": 0.80,
        "winrateAsP2": 0.80,
        "winrateVsChampion": 0.45,
        "decisiveWinRateVsChampion": 0.20,
    }
    stronger_vs_champion = {
        "winrate": 0.72,
        "decisiveWinRate": 0.50,
        "drawRate": 0.22,
        "winrateAsP1": 0.72,
        "winrateAsP2": 0.72,
        "winrateVsChampion": 0.60,
        "decisiveWinRateVsChampion": 0.35,
    }

    assert _checkpoint_selection_score(stronger_vs_champion, 9) > _checkpoint_selection_score(weaker_vs_champion, 9)


def test_build_pure_gap_relabel_candidate_captures_tactical_gap() -> None:
    state = {
        "board_size": 5,
        "board": [[0] * 5 for _ in range(5)],
        "current_player": 1,
        "last_move": None,
    }
    hybrid = {
        "move": 12,
        "tacticalReason": "block_immediate",
        "winningPressure": 0,
        "forcingThreatsAfterMove": 0,
        "searchScore": 0.0,
        "unsafeMovesFiltered": 2,
    }
    pure = {
        "move": 24,
        "tacticalReason": "model_policy",
        "winningPressure": 0,
    }

    candidate = _build_pure_gap_relabel_candidate(state, hybrid, pure)

    assert candidate is not None
    assert candidate["source"] == "failure_pure_gap"
    assert candidate["motif"] == "pure_missed_block"
    assert candidate["conversionTarget"] is False
    assert candidate["pureMissedBlockInOne"] is True
    assert candidate["pureMissedWinInOne"] is False


def test_build_pure_gap_relabel_candidate_keeps_conversion_cases_high_priority() -> None:
    state = {
        "board_size": 5,
        "board": [[0] * 5 for _ in range(5)],
        "current_player": 1,
        "last_move": None,
    }
    hybrid = {
        "move": 12,
        "tacticalReason": "press_winning_advantage",
        "winningPressure": 3,
        "forcingThreatsAfterMove": 2,
        "searchScore": 1.0,
        "unsafeMovesFiltered": 0,
    }
    pure = {
        "move": 1,
        "tacticalReason": "model_policy",
        "winningPressure": 0,
    }

    candidate = _build_pure_gap_relabel_candidate(state, hybrid, pure)

    assert candidate is not None
    assert candidate["source"] == "failure_conversion"
    assert candidate["motif"] == "conversion"
    assert candidate["conversionTarget"] is True


def test_build_pure_gap_relabel_candidate_marks_immediate_win_as_pure_missed_win() -> None:
    state = {
        "board_size": 5,
        "board": [[0] * 5 for _ in range(5)],
        "current_player": 1,
        "last_move": None,
    }
    hybrid = {
        "move": 4,
        "tacticalReason": "immediate_win",
        "winningPressure": 4,
        "forcingThreatsAfterMove": 1,
        "searchScore": 0.0,
        "unsafeMovesFiltered": 0,
    }
    pure = {
        "move": 12,
        "tacticalReason": "model_policy",
        "winningPressure": 0,
    }

    candidate = _build_pure_gap_relabel_candidate(state, hybrid, pure)

    assert candidate is not None
    assert candidate["source"] == "failure_conversion"
    assert candidate["motif"] == "pure_missed_win"
    assert candidate["conversionTarget"] is True
    assert candidate["pureMissedWinInOne"] is True
    assert candidate["pureMissedBlockInOne"] is False


@pytest.mark.asyncio
async def test_run_engine_exam_pure_gap_side_rates_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    async def callback(_event: dict) -> None:
        return None

    class FakeEngine:
        async def best_move(self, board: list[int], _current: int, _board_size: int, _win_len: int) -> int:
            for idx, cell in enumerate(board):
                if cell == 0:
                    return idx
            return -1

    def fake_hybrid_decision(
        board: list[int],
        _board_size: int,
        _win_len: int,
        _current: int,
        _model: object,
        _device: object,
    ) -> dict[str, object]:
        legal = [idx for idx, cell in enumerate(board) if cell == 0]
        return {
            "move": legal[0],
            "tacticalOverride": True,
            "valueGuided": False,
            "tacticalReason": "reject_unsafe_model_move",
            "unsafeMovesFiltered": 1,
        }

    def fake_pure_decision(
        board: list[int],
        _current: int,
        _board_size: int,
        _win_len: int,
        _model: object,
        *,
        decision_mode: str = "hybrid",
    ) -> dict[str, object]:
        assert decision_mode == "pure"
        legal = [idx for idx, cell in enumerate(board) if cell == 0]
        return {
            "move": legal[-1],
            "tacticalOverride": False,
            "valueGuided": False,
            "tacticalReason": "model_policy",
            "unsafeMovesFiltered": 0,
        }

    monkeypatch.setattr("gomoku_api.ws.arena_eval._model_greedy_decision", fake_hybrid_decision)
    monkeypatch.setattr("gomoku_api.ws.predict_service._loaded_model_decision", fake_pure_decision)

    _result, _relabeled, summary = await _run_engine_exam(
        model=object(),
        board_size=3,
        win_len=3,
        device="cpu",
        callback=callback,
        engine_eval=FakeEngine(),
        variant="ttt3",
        cycle=1,
        total_cycles=1,
        num_pairs=1,
        collect_failures=False,
    )

    assert summary["pureGapCount"] > 0
    assert 0.0 <= summary["pureGapRate"] <= 1.0
    assert 0.0 <= summary["pureGapRateAsP1"] <= 1.0
    assert 0.0 <= summary["pureGapRateAsP2"] <= 1.0


def test_choose_rapid_cycle_strategy_focuses_tactical_curriculum_on_pure_missed_wins() -> None:
    strategy = _choose_rapid_cycle_strategy(
        {
            "frozenBlockAcc": 92.0,
            "frozenWinAcc": 92.0,
            "holdoutDeltaAcc": 0.5,
        },
        corrected_rate=0.10,
        failure_bank_size=16,
        engine_per_cycle=50,
        exam_summary={
            "winrateAsP1": 0.5,
            "winrateAsP2": 0.5,
            "decisiveWinRate": 0.10,
            "drawRate": 0.45,
            "pureGapRate": 0.35,
            "pureGapRateAsP1": 0.40,
            "pureGapRateAsP2": 0.20,
            "pureMissedWinCount": 8,
            "pureMissedBlockCount": 3,
        },
    )

    assert strategy["tacticalFocus"] == "win"
    assert strategy["tacticalRatio"] >= 0.60
    assert strategy["conversionFocus"] is True


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
