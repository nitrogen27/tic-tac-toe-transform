"""Tests for engine and health endpoints."""

from __future__ import annotations


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_engine_info(client):
    resp = client.get("/engine/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "version" in data
    assert isinstance(data["supportedBoardSizes"], list)
    assert isinstance(data["capabilities"], list)


def test_analyze_fallback(client):
    """With no real engine binary the adapter returns a center-move fallback."""
    payload = {
        "position": {
            "boardSize": 15,
            "winLength": 5,
            "currentPlayer": 1,
            "cells": [0] * 225,
            "lastMove": -1,
        },
        "topK": 3,
        "timeLimitMs": 500,
        "includePv": True,
    }
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["bestMove"] == 112  # center of 15x15
    assert data["source"] == "alpha_beta"


def test_best_move_fallback(client):
    payload = {
        "position": {
            "boardSize": 9,
            "winLength": 5,
            "currentPlayer": 1,
            "cells": [0] * 81,
        },
        "timeLimitMs": 200,
    }
    resp = client.post("/best-move", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["move"] == 40  # center of 9x9
    assert data["row"] == 4
    assert data["col"] == 4


def test_suggest_fallback(client):
    payload = {
        "position": {
            "boardSize": 7,
            "winLength": 5,
            "currentPlayer": -1,
            "cells": [0] * 49,
        },
        "topK": 3,
    }
    resp = client.post("/suggest", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["suggestions"]) >= 1
