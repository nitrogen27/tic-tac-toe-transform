from __future__ import annotations

from gomoku_api.ws import gpu_info
from gomoku_api.ws.subprocess_utils import windows_hidden_subprocess_kwargs


def test_query_nvidia_smi_parses_telemetry(monkeypatch) -> None:
    sample = (
        "2026/03/29 11:08:37.870, 100, 2, 5880, 6144, 68.03, 95.00, 1965, 7001, 78\n"
    )

    class Result:
        stdout = sample

    monkeypatch.setattr(gpu_info.shutil, "which", lambda _name: "nvidia-smi")
    monkeypatch.setattr(gpu_info.subprocess, "run", lambda *args, **kwargs: Result())

    parsed = gpu_info._query_nvidia_smi()

    assert parsed == {
        "timestamp": "2026/03/29 11:08:37.870",
        "utilizationGpu": 100,
        "utilizationMemory": 2,
        "memoryUsedMB": 5880,
        "memoryTotalMB": 6144,
        "powerDrawW": 68.03,
        "powerLimitW": 95.0,
        "clockSmMHz": 1965,
        "clockMemMHz": 7001,
        "temperatureC": 78,
    }


def test_query_nvidia_smi_uses_hidden_subprocess_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Result:
        stdout = "2026/03/29 11:08:37.870, 0, 0, 0, 6144, 0.0, 95.0, 0, 0, 30\n"

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(gpu_info.shutil, "which", lambda _name: "nvidia-smi")
    monkeypatch.setattr(gpu_info.subprocess, "run", fake_run)

    gpu_info._query_nvidia_smi()

    expected = windows_hidden_subprocess_kwargs()
    for key, value in expected.items():
        assert key in captured
        if key == "startupinfo":
            assert getattr(captured[key], "wShowWindow", None) == getattr(value, "wShowWindow", None)
            assert getattr(captured[key], "dwFlags", None) == getattr(value, "dwFlags", None)
        else:
            assert captured.get(key) == value
