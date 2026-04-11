from __future__ import annotations

import os
import subprocess
from typing import Any


def windows_hidden_subprocess_kwargs(
    *,
    detached: bool = False,
    new_process_group: bool = False,
) -> dict[str, Any]:
    """Return Windows-only kwargs that suppress flashing console windows.

    We use this for child processes spawned during training/monitoring
    (`nvidia-smi`, engine workers, detached trainer). On non-Windows platforms
    this safely returns an empty mapping.
    """
    if os.name != "nt":
        return {}

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if detached:
        creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    if new_process_group:
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    kwargs: dict[str, Any] = {}
    if creationflags:
        kwargs["creationflags"] = creationflags

    startupinfo_factory = getattr(subprocess, "STARTUPINFO", None)
    if startupinfo_factory is not None:
        startupinfo = startupinfo_factory()
        startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        kwargs["startupinfo"] = startupinfo

    return kwargs
