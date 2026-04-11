from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import websockets

TERMINAL_TRAINING_EVENTS = {
    "train.error",
    "train.cancelled",
    "promotion.rejected",
    "background_train.done",
    "background_train.error",
}


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


def _iso_local(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def latest_training_log(root: Path, variant: str) -> Path | None:
    log_dir = root / "saved" / "training_logs" / variant
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return None

    active_logs: list[Path] = []
    for log_path in logs:
        last_obj = read_last_jsonl_object(log_path)
        event = str(last_obj.get("event") or "")
        if event and event not in TERMINAL_TRAINING_EVENTS:
            active_logs.append(log_path)

    if active_logs:
        return max(active_logs, key=lambda p: p.stat().st_mtime)
    return logs[0]


def read_last_jsonl_object(path: Path) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            return json.loads(raw)
        except Exception:
            continue
    return {}


def compact_training_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    keep_keys = {
        "phase",
        "stage",
        "variant",
        "cycle",
        "totalCycles",
        "iteration",
        "totalIterations",
        "percent",
        "overallPercent",
        "overallEta",
        "game",
        "totalGames",
        "generated",
        "total",
        "epoch",
        "totalEpochs",
        "epochPercent",
        "step",
        "totalSteps",
        "batch",
        "totalBatches",
        "loss",
        "accuracy",
        "acc",
        "policyTop1Acc",
        "mae",
        "holdoutPolicyAcc",
        "holdoutPolicyKL",
        "frozenBlockAcc",
        "frozenWinAcc",
        "frozenExactAcc",
        "frozenMidAcc",
        "frozenLateAcc",
        "pureFrozenWinRecall",
        "pureFrozenBlockRecall",
        "hybridFrozenWinRecall",
        "hybridFrozenBlockRecall",
        "pureExactTrapRecall",
        "hybridExactTrapRecall",
        "pureWorstTrapFamilyRecall",
        "hybridWorstTrapFamilyRecall",
        "pureWorstTrapFamily",
        "hybridWorstTrapFamily",
        "pureP1TrapRecall",
        "pureP2TrapRecall",
        "hybridP1TrapRecall",
        "hybridP2TrapRecall",
        "exactPackSize",
        "exactPackFamilyCount",
        "servingReady",
        "servingSource",
        "servingGeneration",
        "winrateVsAlgorithm",
        "winrateVsChampion",
        "decisiveWinRateVsChampion",
        "drawRateVsChampion",
        "winrateVsPreviousCheckpoint",
        "decisiveWinRateVsPreviousCheckpoint",
        "drawRateVsPreviousCheckpoint",
        "acceptedVsPreviousCheckpoint",
        "decisiveWinRate",
        "drawRate",
        "winrateAsP1",
        "winrateAsP2",
        "balancedSideWinrate",
        "pureGapRate",
        "pureMissedWinCount",
        "pureMissedBlockCount",
        "promoted",
        "promotionPending",
        "evaluationQueued",
        "promotionDecision",
        "reason",
        "message",
        "elapsed",
        "eta",
        "heartbeatTs",
        "heartbeatFresh",
        "secondsSinceUpdate",
        "speed",
        "speedUnit",
        "positions",
        "positionsCollected",
        "selfPlayReplaySamples",
        "selfPlayMinReplaySamples",
        "mixedReplaySources",
        "mixedReplayWeights",
        "selfPlayStats",
        "arenaWins",
        "arenaLosses",
        "arenaDraws",
        "progressTrend",
        "deltaWinrate",
        "samplesPerSec",
        "gpuPowerW",
        "gpuUtilization",
        "gpuTemperatureC",
        "teacherBackendResolved",
        "confirmBackendResolved",
        "selectedCheckpointCycle",
        "selectedCheckpointWinrate",
        "selectedCheckpointBalancedSideWinrate",
        "selectedCheckpointWinrateVsChampion",
        "selectedCheckpointWinrateVsPreviousCheckpoint",
        "confirmWinrateAsP1",
        "confirmWinrateAsP2",
        "confirmBalancedSideWinrate",
        "confirmPureGapRate",
        "confirmPureMissedWinCount",
        "confirmPureMissedBlockCount",
    }
    compact = {key: payload.get(key) for key in keep_keys if key in payload}
    return compact


def read_training_state(root: Path, variant: str) -> dict[str, Any]:
    log_path = latest_training_log(root, variant)
    if log_path is None:
        return {
            "active": False,
            "logPath": None,
            "logFile": None,
            "logMtimeIso": None,
            "lastEvent": None,
            "runId": None,
            "eventTs": None,
            "payload": {},
        }
    stat = log_path.stat()
    last_obj = read_last_jsonl_object(log_path)
    event = str(last_obj.get("event") or "")
    payload = last_obj.get("payload") or {}
    active = event not in TERMINAL_TRAINING_EVENTS
    return {
        "active": active,
        "logPath": str(log_path),
        "logFile": log_path.name,
        "logMtimeIso": _iso_local(stat.st_mtime),
        "lastEvent": event or None,
        "runId": last_obj.get("runId"),
        "eventTs": last_obj.get("ts"),
        "payload": compact_training_payload(payload),
    }


async def fetch_runtime_state_once(uri: str, variant: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        ws = await websockets.connect(uri, max_size=10_000_000)
    except Exception:
        return {}, {}, {}
    first: dict[str, Any] = {}
    gpu_payload: dict[str, Any] = {}
    training_payload: dict[str, Any] = {}
    try:
        try:
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
        except Exception:
            first = {}
        await ws.send(json.dumps({"type": "get_gpu_info"}))
        await ws.send(json.dumps({"type": "get_training_status", "payload": {"variant": variant}}))
        deadline = time.time() + 5
        while time.time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            if msg.get("type") == "gpu.info":
                gpu_payload = msg.get("payload", {}) or {}
            elif msg.get("type") == "training.status":
                training_payload = msg.get("payload", {}) or {}
            if gpu_payload and training_payload:
                return gpu_payload, training_payload, first
    except Exception:
        return gpu_payload, training_payload, first
    finally:
        await ws.close()
    return gpu_payload, training_payload, first


def make_snapshot(
    *,
    variant: str,
    gpu: dict[str, Any],
    manifest: dict[str, Any],
    candidate_path: Path,
    working_candidate_path: Path,
    champion_path: Path,
    manifest_path: Path,
    api_log_path: Path,
    training_state: dict[str, Any],
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
            "candidateWorking": file_stat(working_candidate_path),
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
        "training": training_state,
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
    working_candidate_path = variant_dir / "candidate_working.pt"
    champion_path = variant_dir / "champion.pt"
    manifest_path = variant_dir / "manifest.json"
    api_log_path = root / ".runtime" / "legacy-ui-api" / "api.log"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first: dict[str, Any] = {}
    with output_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "event": "monitor.start",
                "variant": variant,
                "uri": uri,
                "first": {},
            }, ensure_ascii=False) + "\n")

            started = time.time()
            while True:
                if duration is not None and (time.time() - started) >= duration:
                    break

                gpu, server_training_state, first_msg = await fetch_runtime_state_once(uri, variant)
                if not first and first_msg:
                    first = first_msg
                manifest = read_manifest(manifest_path)
                training_state = server_training_state or read_training_state(root, variant)
                if training_state:
                    training_state = {
                        **training_state,
                        "payload": compact_training_payload(training_state.get("payload") or {}),
                    }
                snapshot = make_snapshot(
                    variant=variant,
                    gpu=gpu,
                    manifest=manifest,
                    candidate_path=candidate_path,
                    working_candidate_path=working_candidate_path,
                    champion_path=champion_path,
                    manifest_path=manifest_path,
                    api_log_path=api_log_path,
                    training_state=training_state,
                )
                fh.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
                fh.flush()
                await asyncio.sleep(interval)

            fh.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "event": "monitor.stop",
                "variant": variant,
            }, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor active training metrics into JSONL")
    parser.add_argument("--variant", default="ttt5")
    parser.add_argument("--interval", type=float, default=1.0)
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
