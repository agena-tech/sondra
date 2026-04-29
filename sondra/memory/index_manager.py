from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .persistent_memory import PersistentMemoryStore


@dataclass
class MemorySyncResult:
    trigger: str
    indexed_count: int
    health: dict[str, int | float | str | bool]


class MemoryIndexManager:
    def __init__(self, store: PersistentMemoryStore):
        self.store = store

    def probe(self) -> dict[str, int | float | str | bool]:
        return self.store.memory_health()

    def status(self) -> dict[str, int | float | str | bool]:
        return self.probe()

    def sync(self, trigger: str = "manual", limit: int = 300) -> MemorySyncResult:
        indexed = self.store.sync_indices(mode=trigger, limit=limit)
        return MemorySyncResult(
            trigger=str(trigger or "manual"),
            indexed_count=int(indexed),
            health=self.probe(),
        )

    def maybe_sync(self, trigger: str = "interval", limit: int = 250) -> MemorySyncResult:
        indexed = self.store.maybe_sync(trigger=trigger, limit=limit)
        return MemorySyncResult(
            trigger=str(trigger or "interval"),
            indexed_count=int(indexed),
            health=self.probe(),
        )

    def reindex(self, limit: int = 5000) -> MemorySyncResult:
        indexed = self.store.reindex_embeddings(limit=limit)
        return MemorySyncResult(
            trigger="reindex",
            indexed_count=int(indexed),
            health=self.probe(),
        )

    def clear_query_cache(self) -> dict[str, Any]:
        self.store.clear_query_cache()
        return {"ok": True, "health": self.probe()}
