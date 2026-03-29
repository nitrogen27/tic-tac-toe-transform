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

    pool = _build_train_pool(latest, replay, data_count=10, seed_positions=seed)

    ids = {item["id"] for item in pool}
    assert len(pool) == 10
    assert any(item.startswith("latest-") for item in ids)
    assert any(item.startswith("seed-") for item in ids)
