"""Fixed-size replay buffer for self-play training positions."""

from __future__ import annotations

import json
import logging
import random
from collections import deque
from pathlib import Path
from typing import Any

from trainer_lab.config import SelfPlayConfig

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Ring-buffer that stores the most recent training positions."""

    def __init__(self, config: SelfPlayConfig | None = None) -> None:
        self.config = config or SelfPlayConfig()
        self._buffer: deque[dict] = deque(maxlen=self.config.replay_buffer_max)

    def add(self, position: dict) -> None:
        """Add a single position record to the buffer."""
        self._buffer.append(position)

    def add_many(self, positions: list[dict]) -> None:
        """Bulk-add position records."""
        for p in positions:
            self._buffer.append(p)

    def sample(self, n: int) -> list[dict]:
        """Sample *n* random positions (with replacement if n > len)."""
        if len(self._buffer) == 0:
            return []
        return random.choices(list(self._buffer), k=n)

    def save(self, path: str | Path) -> None:
        """Save buffer contents to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(list(self._buffer), f)
        logger.info("Replay buffer saved: %d positions → %s", len(self._buffer), path)

    def load(self, path: str | Path) -> None:
        """Load positions from a JSON file into the buffer."""
        path = Path(path)
        if not path.exists():
            return
        with open(path, "r") as f:
            data = json.load(f)
        for item in data:
            self._buffer.append(item)
        logger.info("Replay buffer loaded: %d positions from %s", len(data), path)

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
