from __future__ import annotations

from trainer_lab.self_play.mixed_replay import MixedReplay


def test_mixed_replay_add_replace_and_summary() -> None:
    replay = MixedReplay(
        total_capacity=100,
        source_limits={"anchor": 20, "self_play": 40, "failure": 20},
    )

    replay.add_many("anchor", [{"id": i} for i in range(10)])
    replay.add_many("self_play", [{"id": f"sp-{i}"} for i in range(12)])
    replay.replace("failure", [{"id": "f-1"}, {"id": "f-2"}])

    summary = replay.summary()

    assert len(replay) == 24
    assert summary["sources"]["anchor"] == 10
    assert summary["sources"]["self_play"] == 12
    assert summary["sources"]["failure"] == 2


def test_mixed_replay_sample_uses_requested_source_weights() -> None:
    replay = MixedReplay(
        total_capacity=200,
        source_limits={"anchor": 50, "self_play": 50},
    )
    replay.add_many("anchor", [{"id": f"a-{i}"} for i in range(30)])
    replay.add_many("self_play", [{"id": f"sp-{i}"} for i in range(30)])

    sample = replay.sample(40, source_weights={"anchor": 0.1, "self_play": 0.9})
    self_play_count = sum(item.get("replayBucket") == "self_play" for item in sample)

    assert len(sample) == 40
    assert self_play_count >= 24


def test_mixed_replay_save_and_load_round_trip(tmp_path) -> None:
    replay = MixedReplay(
        total_capacity=100,
        source_limits={"anchor": 20, "self_play": 30, "user": 20},
    )
    replay.add_many("anchor", [{"id": 1}])
    replay.add_many("self_play", [{"id": 2}])
    replay.add_many("user", [{"id": 3}])

    path = tmp_path / "mixed_replay.json"
    replay.save(path)

    restored = MixedReplay(total_capacity=100)
    restored.load(path)

    assert len(restored) == 3
    assert restored.summary()["sources"]["anchor"] == 1
    assert restored.summary()["sources"]["self_play"] == 1
    assert restored.summary()["sources"]["user"] == 1
