from __future__ import annotations

from gomoku_api.ws.engine_evaluator import EngineEvaluator
from gomoku_api.ws.oracle_backends import create_oracle_evaluator, normalize_oracle_backend
from gomoku_api.ws.rapfi_adapter import RapfiAdapter, rapfi_supports_variant


def test_normalize_oracle_backend_maps_aliases() -> None:
    assert normalize_oracle_backend("engine") == "builtin"
    assert normalize_oracle_backend("builtin") == "builtin"
    assert normalize_oracle_backend("rapfi") == "rapfi"


def test_normalize_oracle_backend_auto_prefers_rapfi_when_available(monkeypatch) -> None:
    monkeypatch.setattr("gomoku_api.ws.oracle_backends.settings.rapfi_enabled", True)
    monkeypatch.setattr("gomoku_api.ws.oracle_backends.settings.rapfi_binary", "rapfi.exe")

    assert normalize_oracle_backend("auto", board_size=15, win_len=5) == "rapfi"
    assert normalize_oracle_backend("auto", board_size=5, win_len=4) == "builtin"


def test_rapfi_supports_only_standard_five_in_row() -> None:
    assert rapfi_supports_variant(15, 5) is True
    assert rapfi_supports_variant(5, 4) is False


def test_create_oracle_evaluator_falls_back_for_ttt5(monkeypatch) -> None:
    monkeypatch.setattr("gomoku_api.ws.oracle_backends.settings.rapfi_enabled", True)
    monkeypatch.setattr("gomoku_api.ws.oracle_backends.settings.rapfi_binary", "rapfi.exe")

    evaluator, resolved = create_oracle_evaluator(5, 4, backend="rapfi", role="teacher")

    assert resolved == "builtin"
    assert isinstance(evaluator, EngineEvaluator)


def test_create_oracle_evaluator_returns_rapfi_for_standard_gomoku(monkeypatch) -> None:
    monkeypatch.setattr("gomoku_api.ws.oracle_backends.settings.rapfi_enabled", True)
    monkeypatch.setattr("gomoku_api.ws.oracle_backends.settings.rapfi_binary", "rapfi.exe")

    evaluator, resolved = create_oracle_evaluator(15, 5, backend="rapfi", role="confirm")

    assert resolved == "rapfi"
    assert isinstance(evaluator, RapfiAdapter)
