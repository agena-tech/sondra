from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from .episodic_memory import EpisodicMemory
from .index_manager import MemoryIndexManager
from .persistent_memory import PersistentMemoryStore


class MemoryRuntime:
    def __init__(self) -> None:
        self.store: PersistentMemoryStore | None = None
        self.index_manager: MemoryIndexManager | None = None
        self.episodic = EpisodicMemory()

    def initialize(
        self,
        *,
        model: str | None = None,
        base_dir: str | None = None,
        sync_limit: int = 300,
    ) -> None:
        self.store = PersistentMemoryStore(base_dir=base_dir)
        self.store.configure_llm_extraction(enabled=False, model=model)
        self.index_manager = MemoryIndexManager(self.store)
        with contextlib.suppress(Exception):
            self.index_manager.sync(trigger="session_start", limit=sync_limit)

    def clear_persistent_handles(self) -> None:
        self.store = None
        self.index_manager = None

    def reset(
        self,
        *,
        model: str | None = None,
        sync_limit: int = 300,
    ) -> None:
        if not self.store:
            raise RuntimeError("Memory runtime is not initialized.")

        memory_dir = Path(str(self.store.memory_dir))
        db_path = Path(str(self.store.db_path))

        self.clear_persistent_handles()
        for suffix in ("", "-wal", "-shm"):
            with contextlib.suppress(FileNotFoundError):
                Path(str(db_path) + suffix).unlink()

        self.initialize(model=model, base_dir=str(memory_dir), sync_limit=sync_limit)

    def maybe_sync(self, limit: int = 250) -> None:
        if not self.index_manager:
            return
        with contextlib.suppress(Exception):
            self.index_manager.maybe_sync(trigger="interval", limit=limit)
        with contextlib.suppress(Exception):
            self.index_manager.maybe_sync(trigger="watch", limit=limit)

    def latest_emotion_summary(
        self,
        role: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.store:
            return None

        payload = self.store.get_latest_emotion_signal(role=role, session_id=session_id)
        if not isinstance(payload, dict):
            return None

        scores = payload.get("scores", {})
        if not isinstance(scores, dict):
            scores = {}

        return {
            "payload": payload,
            "message_id": int(payload.get("message_id", 0) or 0),
            "role": str(payload.get("role", "") or ""),
            "created_at": str(payload.get("created_at", "") or ""),
            "scores": scores,
            "top_emotion": str(payload.get("top_emotion", "neutral") or "neutral"),
            "top_score": float(payload.get("top_score", 0.0) or 0.0),
            "confidence": float(payload.get("confidence", 0.35) or 0.35),
        }
