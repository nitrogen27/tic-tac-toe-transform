"""Fixed-size replay buffer for self-play training positions."""

from __future__ import annotations

import random
from collections import deque
from typing import Any

from trainer_lab.config import SelfPlayConfig


class ReplayBuffer:
    """Ring-buffer that stores the most recent training positions.

    Phase-6 stub — the API is functional but minimal.
    """

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

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
