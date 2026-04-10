"""Source-aware mixed replay for AlphaZero-style training.

This replay manager is designed for projects that combine multiple position
sources instead of relying on raw self-play alone:

- anchor / teacher positions
- tactical curriculum positions
- failure-bank / repair positions
- user-game relabel positions
- self-play visit-target positions

Each source gets its own ring buffer, while sampling can draw a configurable
mixture across sources.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_LIMITS: dict[str, int] = {
    "anchor": 15_000,
    "tactical": 10_000,
    "failure": 10_000,
    "user": 7_500,
    "self_play": 20_000,
}

DEFAULT_SAMPLE_WEIGHTS: dict[str, float] = {
    "anchor": 0.25,
    "tactical": 0.20,
    "failure": 0.20,
    "user": 0.15,
    "self_play": 0.20,
}


class MixedReplay:
    """Ring-buffer replay partitioned by source bucket."""

    def __init__(
        self,
        *,
        total_capacity: int = 50_000,
        source_limits: dict[str, int] | None = None,
        default_sample_weights: dict[str, float] | None = None,
    ) -> None:
        if total_capacity <= 0:
            raise ValueError("total_capacity must be positive")
        raw_limits = dict(source_limits or DEFAULT_SOURCE_LIMITS)
        if not raw_limits:
            raw_limits = dict(DEFAULT_SOURCE_LIMITS)

        limit_sum = sum(max(int(v), 0) for v in raw_limits.values())
        if limit_sum <= 0:
            raise ValueError("source_limits must contain at least one positive capacity")

        # Scale limits down if callers provide a total cap smaller than the sum of limits.
        if limit_sum > total_capacity:
            scale = total_capacity / float(limit_sum)
            scaled = {
                source: max(1, int(math.floor(limit * scale)))
                for source, limit in raw_limits.items()
                if limit > 0
            }
            scaled_sum = sum(scaled.values())
            if scaled_sum < total_capacity and scaled:
                first_key = next(iter(scaled))
                scaled[first_key] += total_capacity - scaled_sum
            raw_limits = scaled

        self.total_capacity = int(total_capacity)
        self.source_limits = {source: int(limit) for source, limit in raw_limits.items() if int(limit) > 0}
        self.default_sample_weights = dict(default_sample_weights or DEFAULT_SAMPLE_WEIGHTS)
        self._buffers: dict[str, deque[dict[str, Any]]] = {
            source: deque(maxlen=limit)
            for source, limit in self.source_limits.items()
        }

    def _ensure_bucket(self, source: str) -> str:
        bucket = str(source or "other").strip() or "other"
        if bucket not in self._buffers:
            # Dynamically added sources reuse a small fallback capacity.
            fallback_limit = max(256, min(2_000, self.total_capacity // 10))
            self.source_limits[bucket] = fallback_limit
            self._buffers[bucket] = deque(maxlen=fallback_limit)
        return bucket

    def add(self, source: str, position: dict[str, Any]) -> None:
        bucket = self._ensure_bucket(source)
        record = dict(position)
        record.setdefault("source", source)
        record["replayBucket"] = bucket
        self._buffers[bucket].append(record)

    def add_many(self, source: str, positions: list[dict[str, Any]]) -> None:
        for position in positions:
            self.add(source, position)

    def replace(self, source: str, positions: list[dict[str, Any]]) -> None:
        bucket = self._ensure_bucket(source)
        self._buffers[bucket].clear()
        for position in positions:
            self.add(bucket, position)

    def clear(self) -> None:
        for bucket in self._buffers.values():
            bucket.clear()

    def size_by_source(self) -> dict[str, int]:
        return {source: len(buffer) for source, buffer in self._buffers.items()}

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self),
            "capacity": self.total_capacity,
            "sources": self.size_by_source(),
            "sourceLimits": dict(self.source_limits),
            "defaultSampleWeights": dict(self.default_sample_weights),
        }

    def sample(
        self,
        n: int,
        *,
        source_weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        if n <= 0:
            return []

        available = {source: list(buffer) for source, buffer in self._buffers.items() if len(buffer) > 0}
        if not available:
            return []

        weights = dict(self.default_sample_weights)
        if source_weights:
            weights.update(source_weights)

        positive = {source: max(float(weights.get(source, 0.0)), 0.0) for source in available}
        weight_sum = sum(positive.values())
        if weight_sum <= 0.0:
            positive = {source: 1.0 for source in available}
            weight_sum = float(len(positive))

        normalized = {source: value / weight_sum for source, value in positive.items()}
        quotas: dict[str, int] = {}
        remainder: list[tuple[float, str]] = []
        assigned = 0
        for source, weight in normalized.items():
            exact = n * weight
            quota = int(math.floor(exact))
            quotas[source] = quota
            assigned += quota
            remainder.append((exact - quota, source))

        for _, source in sorted(remainder, reverse=True):
            if assigned >= n:
                break
            quotas[source] += 1
            assigned += 1

        sampled: list[dict[str, Any]] = []
        for source, quota in quotas.items():
            if quota <= 0:
                continue
            sampled.extend(random.choices(available[source], k=quota))

        while len(sampled) < n:
            source = random.choices(list(available.keys()), weights=[normalized[s] for s in available], k=1)[0]
            sampled.append(random.choice(available[source]))

        random.shuffle(sampled)
        return sampled[:n]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_capacity": self.total_capacity,
            "source_limits": self.source_limits,
            "default_sample_weights": self.default_sample_weights,
            "buffers": {source: list(buffer) for source, buffer in self._buffers.items()},
        }
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.info("Mixed replay saved: %d positions → %s", len(self), target)

    def load(self, path: str | Path) -> None:
        source_path = Path(path)
        if not source_path.exists():
            return
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if payload.get("source_limits"):
            self.source_limits = {source: int(limit) for source, limit in payload["source_limits"].items() if int(limit) > 0}
        if payload.get("default_sample_weights"):
            self.default_sample_weights = dict(payload["default_sample_weights"])
        self._buffers = {
            source: deque(payload.get("buffers", {}).get(source, []), maxlen=limit)
            for source, limit in self.source_limits.items()
        }
        # Preserve extra buckets from old states even if they were not declared in source_limits.
        for source, items in payload.get("buffers", {}).items():
            if source in self._buffers:
                continue
            limit = max(256, min(2_000, self.total_capacity // 10))
            self.source_limits[source] = limit
            self._buffers[source] = deque(items, maxlen=limit)
        logger.info("Mixed replay loaded: %d positions from %s", len(self), source_path)

    def __len__(self) -> int:
        return sum(len(buffer) for buffer in self._buffers.values())
