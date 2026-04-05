"""Tests for training job endpoints."""

from __future__ import annotations

from pathlib import Path


def test_create_job(client):
    payload = {"variant": "gomoku15", "epochs": 3, "batchSize": 64}
    resp = client.post("/train/jobs", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert "jobId" in data
    assert data["variant"] == "gomoku15"
    assert data["status"] in ("queued", "running")


def test_get_job(client):
    resp = client.post("/train/jobs", json={"variant": "gomoku15", "epochs": 1})
    job_id = resp.json()["jobId"]

    resp = client.get(f"/train/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["jobId"] == job_id


def test_get_job_404(client):
    resp = client.get("/train/jobs/nonexistent-id")
    assert resp.status_code == 404


def test_list_jobs(client):
    client.post("/train/jobs", json={"variant": "test1"})
    client.post("/train/jobs", json={"variant": "test2"})

    resp = client.get("/train/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 2


def test_cancel_job(client):
    resp = client.post("/train/jobs", json={"variant": "gomoku15", "epochs": 30})
    job_id = resp.json()["jobId"]

    resp = client.delete(f"/train/jobs/{job_id}")
    assert resp.status_code == 204

    resp = client.get(f"/train/jobs/{job_id}")
    assert resp.json()["status"] == "cancelled"


def test_create_engine_dataset_route(client, monkeypatch, tmp_path: Path):
    async def fake_generate_engine_dataset(variant: str, count: int):
        path = tmp_path / f"{variant}_engine.json"
        path.write_text("[]", encoding="utf-8")
        return path

    monkeypatch.setattr("gomoku_api.routers.training.generate_engine_dataset", fake_generate_engine_dataset)

    resp = client.post("/train/datasets/engine", json={"variant": "ttt5", "count": 1234})
    assert resp.status_code == 200
    data = resp.json()
    assert data["variant"] == "ttt5"
    assert data["count"] == 1234
    assert data["mode"] == "engine"
    assert data["path"].endswith("ttt5_engine.json")


def test_run_diagnostics_route(client, monkeypatch):
    async def fake_run_training_diagnostics(variant: str, **kwargs):
        return {
            "variant": variant,
            "datasetSize": kwargs["dataset_limit"],
            "tinyOverfitPassed": True,
        }

    monkeypatch.setattr("gomoku_api.routers.training.run_training_diagnostics", fake_run_training_diagnostics)

    resp = client.post("/train/diagnostics", json={"variant": "ttt5", "datasetLimit": 128, "tinySteps": 16})
    assert resp.status_code == 200
    data = resp.json()
    assert data["variant"] == "ttt5"
    assert data["datasetSize"] == 128
    assert data["tinyOverfitPassed"] is True
