from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from gomoku_api.ws.train_service_ws import run_deferred_evaluator_tail, train_variant
from gomoku_api.ws.training_run_logger import TrainingRunLogger


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


async def _run_worker(
    *,
    variant: str,
    request_path: Path,
    meta_path: Path,
    cancel_path: Path,
) -> int:
    request_payload = _read_json(request_path)
    kwargs = dict(request_payload.get("payload") or {})
    logger = TrainingRunLogger(variant)

    meta = _read_json(meta_path) if meta_path.exists() else {}
    meta.update(
        {
            "variant": variant,
            "pid": os.getpid(),
            "active": True,
            "runId": logger.run_id,
            "logPath": str(logger.path),
            "workerStartedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    _write_json(meta_path, meta)
    cancel_path.unlink(missing_ok=True)

    async def callback(event: dict[str, Any]) -> None:
        logger.log(event)
        if cancel_path.exists():
            raise asyncio.CancelledError

    exit_code = 0
    final_event = "worker.done"
    try:
        result = await train_variant(variant, callback, **kwargs)
        deferred = result.get("deferredEvaluator") if isinstance(result, dict) else None
    except asyncio.CancelledError:
        final_event = "train.cancelled"
        logger.log({"type": "train.cancelled", "payload": {"variant": variant}})
        exit_code = 2
    except Exception as exc:
        final_event = "train.error"
        logger.log({"type": "train.error", "payload": {"variant": variant, "error": str(exc)}})
        exit_code = 1
    else:
        final_event = "train.done"
        if deferred:
            try:
                await run_deferred_evaluator_tail(deferred, callback)
                final_event = "background_train.done"
            except asyncio.CancelledError:
                final_event = "background_train.error"
                logger.log({
                    "type": "background_train.error",
                    "payload": {
                        "variant": variant,
                        "message": "Фоновая оценка отменена.",
                    },
                })
                exit_code = 2
            except Exception as exc:
                final_event = "background_train.error"
                logger.log({
                    "type": "background_train.error",
                    "payload": {
                        "variant": variant,
                        "message": f"Фоновая оценка завершилась с ошибкой: {exc}",
                        "error": str(exc),
                    },
                })
                exit_code = 1
    finally:
        meta = _read_json(meta_path) if meta_path.exists() else {}
        meta.update(
            {
                "variant": variant,
                "pid": os.getpid(),
                "active": False,
                "endedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "finalEvent": final_event,
                "exitCode": exit_code,
                "runId": logger.run_id,
                "logPath": str(logger.path),
            }
        )
        _write_json(meta_path, meta)
        cancel_path.unlink(missing_ok=True)
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Detached training worker for TTT/Gomoku models.")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--cancel-file", required=True)
    args = parser.parse_args()

    exit_code = asyncio.run(
        _run_worker(
            variant=args.variant,
            request_path=Path(args.request),
            meta_path=Path(args.meta),
            cancel_path=Path(args.cancel_file),
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
