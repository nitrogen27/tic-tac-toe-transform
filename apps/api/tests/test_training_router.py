"""Tests for training job endpoints."""

from __future__ import annotations


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
