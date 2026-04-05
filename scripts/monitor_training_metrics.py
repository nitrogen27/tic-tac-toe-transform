from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import websockets


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_stat(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "mtimeIso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
    }


async def ws_connect(uri: str):
    ws = await websockets.connect(uri, max_size=10_000_000)
    try:
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
    except Exception:
        first = {}
    return ws, first


async def fetch_gpu_info(ws) -> dict[str, Any]:
    await ws.send(json.dumps({"type": "get_gpu_info"}))
    deadline = time.time() + 10
    while time.time() < deadline:
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        msg = json.loads(raw)
        if msg.get("type") == "gpu.info":
            return msg.get("payload", {}) or {}
    return {}


def make_snapshot(
    *,
    variant: str,
    gpu: dict[str, Any],
    manifest: dict[str, Any],
    candidate_path: Path,
    champion_path: Path,
    manifest_path: Path,
    api_log_path: Path,
) -> dict[str, Any]:
    telemetry = gpu.get("telemetry") or {}
    last_history = (manifest.get("history") or [])
    last_entry = last_history[-1] if last_history else {}
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "variant": variant,
        "gpu": {
            "available": gpu.get("available"),
            "backend": gpu.get("backend"),
            "name": gpu.get("name"),
            "utilizationGpu": telemetry.get("utilizationGpu"),
            "utilizationMemory": telemetry.get("utilizationMemory"),
            "powerDrawW": telemetry.get("powerDrawW"),
            "memoryUsedMB": telemetry.get("memoryUsedMB"),
            "memoryTotalMB": telemetry.get("memoryTotalMB"),
            "temperatureC": telemetry.get("temperatureC"),
            "timestamp": telemetry.get("timestamp"),
        },
        "files": {
            "candidate": file_stat(candidate_path),
            "champion": file_stat(champion_path),
            "manifest": file_stat(manifest_path),
            "apiLog": file_stat(api_log_path),
        },
        "manifest": {
            "currentChampionGeneration": manifest.get("current_champion_generation"),
            "historyCount": len(last_history),
            "lastReason": last_entry.get("reason"),
            "lastWinrateVsAlgorithm": last_entry.get("winrateVsAlgorithm"),
            "lastBlockAccuracy": last_entry.get("blockAccuracy"),
            "lastWinAccuracy": last_entry.get("winAccuracy"),
        },
    }


async def monitor(
    *,
    variant: str,
    interval: float,
    duration: float | None,
    output_path: Path,
    uri: str,
) -> None:
    root = repo_root()
    variant_dir = root / "saved" / f"{variant}_resnet"
    candidate_path = variant_dir / "candidate.pt"
    champion_path = variant_dir / "champion.pt"
    manifest_path = variant_dir / "manifest.json"
    api_log_path = root / ".runtime" / "legacy-ui-api" / "api.log"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ws, first = await ws_connect(uri)
    try:
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "event": "monitor.start",
                "variant": variant,
                "uri": uri,
                "first": first,
            }, ensure_ascii=False) + "\n")

            started = time.time()
            while True:
                if duration is not None and (time.time() - started) >= duration:
                    break

                gpu = await fetch_gpu_info(ws)
                manifest = read_manifest(manifest_path)
                snapshot = make_snapshot(
                    variant=variant,
                    gpu=gpu,
                    manifest=manifest,
                    candidate_path=candidate_path,
                    champion_path=champion_path,
                    manifest_path=manifest_path,
                    api_log_path=api_log_path,
                )
                fh.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
                fh.flush()
                await asyncio.sleep(interval)

            fh.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "event": "monitor.stop",
                "variant": variant,
            }, ensure_ascii=False) + "\n")
    finally:
        await ws.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor active training metrics into JSONL")
    parser.add_argument("--variant", default="ttt5")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--uri", default="ws://127.0.0.1:8080/")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = repo_root()
    if args.out:
        out = Path(args.out)
    else:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        out = root / ".runtime" / "training-metrics" / args.variant / f"{stamp}.jsonl"

    asyncio.run(
        monitor(
            variant=args.variant,
            interval=args.interval,
            duration=args.duration,
            output_path=out,
            uri=args.uri,
        )
    )


if __name__ == "__main__":
    main()
