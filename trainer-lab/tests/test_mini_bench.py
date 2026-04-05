from __future__ import annotations

from trainer_lab.training.mini_bench import run_mini_benchmark


def test_run_mini_benchmark_returns_metrics() -> None:
    result = run_mini_benchmark(steps=2, warmup_steps=1, batch_size=4, board_size=7, device="cpu")

    assert result["device"] == "cpu"
    assert result["steps"] == 2
    assert result["batchSize"] == 4
    assert result["avgStepMs"] > 0
    assert result["samplesPerSec"] > 0
    assert result["modelParams"] > 0
    assert "torchCompile" in result
