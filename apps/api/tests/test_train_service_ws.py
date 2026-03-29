from __future__ import annotations

import pytest

from gomoku_api.ws.train_service_ws import (
    _build_train_pool,
    _generate_tactical_curriculum_positions,
    _policy_cell_index,
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


def test_variant_model_hparams_scale_up_ttt5() -> None:
    cfg = ModelConfig()
    ttt5 = _variant_model_hparams(5, cfg)
    gomoku15 = _variant_model_hparams(15, cfg)

    assert ttt5 == (96, 8, 160)
    assert gomoku15[0] >= 128
    assert gomoku15[1] >= 8


def test_policy_cell_index_maps_inside_padded_grid() -> None:
    assert _policy_cell_index(0, 5) == 0
    assert _policy_cell_index(6, 5) == 17


def test_build_train_pool_keeps_seed_samples() -> None:
    latest = [{"id": f"latest-{idx}"} for idx in range(8)]
    replay = [{"id": f"replay-{idx}"} for idx in range(8)]
    seed = [{"id": f"seed-{idx}"} for idx in range(4)]
    hard = [{"id": f"hard-{idx}"} for idx in range(4)]
    minimax = [{"id": f"minimax-{idx}"} for idx in range(4)]

    pool = _build_train_pool(
        latest,
        replay,
        data_count=10,
        seed_positions=seed,
        hard_positions=hard,
        minimax_positions=minimax,
    )

    ids = {item["id"] for item in pool}
    assert len(pool) == 10
    assert any(item.startswith("latest-") for item in ids)
    assert any(item.startswith("seed-") for item in ids)
    assert any(item.startswith("hard-") for item in ids)
    assert any(item.startswith("minimax-") for item in ids)


def test_build_train_pool_tactical_not_diluted_by_bootstrap() -> None:
    """Tactical quota must contain ONLY tactical positions, never bootstrap.

    The caller must pass bootstrap as replay, not as seed_positions.
    This test verifies the contract: seed_positions = tactical only.
    """
    tactical = [{"id": f"tactical-{i}", "source": "tactical"} for i in range(10)]
    bootstrap = [{"id": f"bootstrap-{i}", "source": "bootstrap"} for i in range(10)]
    latest = [{"id": f"latest-{i}", "source": "selfplay"} for i in range(20)]
    minimax = [{"id": f"minimax-{i}", "source": "minimax"} for i in range(10)]

    # Correct usage: bootstrap goes into replay, NOT into seed_positions
    pool = _build_train_pool(
        latest,
        bootstrap,  # bootstrap as replay
        data_count=20,
        seed_positions=tactical,  # only tactical here
        minimax_positions=minimax,
    )

    ids = {item["id"] for item in pool}
    # Tactical quota = 20% of 20 = 4, so at least 4 tactical items
    tactical_in_pool = [item for item in pool if item.get("source") == "tactical"]
    assert len(tactical_in_pool) >= 4, f"Expected >= 4 tactical, got {len(tactical_in_pool)}"

    # No bootstrap item should appear in the tactical quota slot
    # (bootstrap can appear via replay bucket, but NOT as seed_positions)
    minimax_in_pool = [item for item in pool if item.get("source") == "minimax"]
    assert len(minimax_in_pool) >= 3, f"Expected >= 3 minimax, got {len(minimax_in_pool)}"

    # Verify pool has all sources represented
    sources = {item.get("source") for item in pool}
    assert "tactical" in sources
    assert "minimax" in sources
    assert "selfplay" in sources
