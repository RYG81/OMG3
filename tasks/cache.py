"""Bounded, process-memory response cache for local Ollama tasks."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheStats:
    entries: int
    bytes: int
    hits: int
    misses: int
    max_entries: int
    max_bytes: int


class MemoryResponseCache:
    """Thread-safe LRU cache that never writes prompts or responses to disk."""

    def __init__(self, max_entries: int = 128, max_bytes: int = 64 * 1024 * 1024):
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max(1024, int(max_bytes))
        self._items: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self._misses += 1
                return None
            self._items.move_to_end(key)
            self._hits += 1
            return item[0]

    def put(self, key: str, value: Any, size_bytes: int) -> None:
        size = max(0, int(size_bytes))
        if size > self.max_bytes:
            return
        with self._lock:
            previous = self._items.pop(key, None)
            if previous:
                self._bytes -= previous[1]
            self._items[key] = (value, size)
            self._bytes += size
            while len(self._items) > self.max_entries or self._bytes > self.max_bytes:
                _old_key, (_old_value, old_size) = self._items.popitem(last=False)
                self._bytes -= old_size

    def clear(self) -> int:
        with self._lock:
            count = len(self._items)
            self._items.clear()
            self._bytes = 0
            return count

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                entries=len(self._items),
                bytes=self._bytes,
                hits=self._hits,
                misses=self._misses,
                max_entries=self.max_entries,
                max_bytes=self.max_bytes,
            )


TASK_RESPONSE_CACHE = MemoryResponseCache()
