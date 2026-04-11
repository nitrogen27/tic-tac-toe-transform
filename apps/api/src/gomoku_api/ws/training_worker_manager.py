from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gomoku_api.ws.subprocess_utils import windows_hidden_subprocess_kwargs


REPO_ROOT = Path(__file__).resolve().parents[5]
RUNTIME_DIR = REPO_ROOT / ".runtime" / "training-workers"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _is_pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        synchronize = 0x00100000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information | synchronize,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))  # type: ignore[attr-defined]
            if not ok:
                return False
            return int(exit_code.value) == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


@dataclass
class TrainingWorkerManager:
    variant: str

    @property
    def variant_dir(self) -> Path:
        return RUNTIME_DIR / self.variant

    @property
    def meta_path(self) -> Path:
        return self.variant_dir / "worker.json"

    @property
    def request_path(self) -> Path:
        return self.variant_dir / "request.json"

    @property
    def cancel_path(self) -> Path:
        return self.variant_dir / "cancel.flag"

    @property
    def stdout_path(self) -> Path:
        return self.variant_dir / "worker.out.log"

    @property
    def stderr_path(self) -> Path:
        return self.variant_dir / "worker.err.log"

    def read_meta(self) -> dict[str, Any]:
        return _read_json(self.meta_path)

    def is_active(self) -> bool:
        meta = self.read_meta()
        pid = int(meta.get("pid") or 0)
        alive = _is_pid_alive(pid)
        if meta and not alive and meta.get("active"):
            meta["active"] = False
            meta["endedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            _write_json(self.meta_path, meta)
        return alive

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.is_active():
            raise RuntimeError(f"training worker already running for {self.variant}")

        self.variant_dir.mkdir(parents=True, exist_ok=True)
        self.cancel_path.unlink(missing_ok=True)
        _write_json(
            self.request_path,
            {
                "variant": self.variant,
                "payload": payload,
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        required_paths = [
            str(REPO_ROOT / "apps" / "api" / "src"),
            str(REPO_ROOT / "trainer-lab" / "src"),
        ]
        env["PYTHONPATH"] = os.pathsep.join([*required_paths, existing_pythonpath] if existing_pythonpath else required_paths)

        with self.stdout_path.open("ab") as stdout_fh, self.stderr_path.open("ab") as stderr_fh:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "gomoku_api.ws.training_worker",
                    "--variant",
                    self.variant,
                    "--request",
                    str(self.request_path),
                    "--meta",
                    str(self.meta_path),
                    "--cancel-file",
                    str(self.cancel_path),
                ],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=stdout_fh,
                stderr=stderr_fh,
                close_fds=True,
                **windows_hidden_subprocess_kwargs(detached=True, new_process_group=True),
            )

        meta = {
            "variant": self.variant,
            "pid": proc.pid,
            "active": True,
            "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "requestPath": str(self.request_path),
            "stdoutPath": str(self.stdout_path),
            "stderrPath": str(self.stderr_path),
        }
        _write_json(self.meta_path, meta)
        return meta

    def request_cancel(self, timeout_seconds: float = 8.0) -> bool:
        meta = self.read_meta()
        pid = int(meta.get("pid") or 0)
        if not _is_pid_alive(pid):
            return False

        self.cancel_path.write_text("cancel\n", encoding="utf-8")
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if not _is_pid_alive(pid):
                updated = self.read_meta()
                updated["active"] = False
                updated["endedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                _write_json(self.meta_path, updated)
                return True
            time.sleep(0.25)

        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            check=False,
            capture_output=True,
            text=True,
            **windows_hidden_subprocess_kwargs(),
        )
        updated = self.read_meta()
        updated["active"] = False
        updated["endedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        updated["forcedStop"] = True
        _write_json(self.meta_path, updated)
        return True
