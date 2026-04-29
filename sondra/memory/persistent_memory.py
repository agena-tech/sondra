import logging
import os
import re
import sqlite3
import time
import contextlib
from hashlib import blake2b
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
import requests

from .importance_engine import Amygdala
from .context_builder import build_auto_context as build_auto_context_impl
from .context_builder import quick_score as quick_score_impl
from .retrieval_engine import search_conversation as search_conversation_impl
from .retrieval_engine import search_conversation_structured as search_conversation_structured_impl
from .retrieval_engine import search_ranked_hits as search_ranked_hits_impl
from .signal_catalog import get_memory_signal_catalog, normalize_signal_text

logger = logging.getLogger(__name__)

OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

def get_relative_time_label(timestamp: datetime) -> str:
    now = datetime.now(UTC)
    delta = now - timestamp
    days = max(0, int(delta.days))

    if days == 0:
        return "TODAY"
    if days == 1:
        return "YESTERDAY"
    if days < 7:
        return f"{days} DAYS AGO"
    if days < 30:
        return f"{days // 7} WEEKS AGO"
    if days < 365:
        return f"{days // 30} MONTHS AGO"
    return f"{days // 365} YEARS AGO"


@dataclass
class ConversationMessage:
    id: int
    timestamp: str
    role: str
    content: str
    session_id: str


@dataclass
class MemorySearchHit:
    id: int
    timestamp: str
    role: str
    content: str
    body: str
    lexical_score: float
    recency_score: float
    fts_score: float
    vector_score: float
    temporal_score: float
    final_score: float


@dataclass
class ScheduledTask:
    id: int
    task_text: str
    schedule_time: str
    scheduled_for: str
    status: str
    created_at: str
    schedule_type: str = "once"
    cron_expression: str = ""
    dispatched_at: str = ""
    worker_name: str = ""
    worker_id: str = ""
    session_id: str = ""
    owner_agent_id: str = ""

    @property
    def next_run(self) -> str:
        return str(self.scheduled_for or "")

    @property
    def last_run(self) -> str:
        return str(self.dispatched_at or "")


class PersistentMemoryStore:
    LEGACY_SESSION_ID = "__legacy__"
    DEFAULT_EMBED_DIM = 96
    MAX_SEARCH_OUTPUT_CHARS = 2000
    RECENT_SEMANTIC_PREFIX = "[RECENT]"
    EPISODIC_TTL_DAYS = 3
    SEMANTIC_DECAY_FLOOR = 0.3

    def __init__(self, base_dir: str | None = None):
        self.signal_catalog = get_memory_signal_catalog()
        self._memory_config = self._load_memory_config()
        preferred_dir = base_dir or self._cfg_str("SONDRA_MEMORY_DIR", "/workspace/memory")
        self.memory_dir = self._resolve_memory_dir(preferred_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.memory_dir / "memory.db"
        self.memory_logs_dir = self._resolve_logs_dir(self._cfg_str("SONDRA_MEMORY_LOG_DIR", "logs"))
        self.memory_debug_log_path = self.memory_logs_dir / "memory_debug.log"
        self.memory_events_jsonl_path = self.memory_logs_dir / "memory_events.jsonl"
        
        # Search settings
        self.hybrid_enabled = self._cfg_bool("SONDRA_MEMORY_HYBRID_ENABLED", True)
        self.mmr_enabled = self._cfg_bool("SONDRA_MEMORY_MMR_ENABLED", True)
        self.temporal_decay_enabled = self._cfg_bool("SONDRA_MEMORY_TEMPORAL_DECAY", True)
        self.embedding_dim = self._cfg_int("SONDRA_MEMORY_EMBED_DIM", self.DEFAULT_EMBED_DIM, minimum=32)
        self.embed_timeout_sec = self._cfg_int("SONDRA_MEMORY_EMBED_TIMEOUT_SEC", 3, minimum=1)
        self.mmr_lambda = self._cfg_float("SONDRA_MEMORY_MMR_LAMBDA", 0.72, minimum=0.0, maximum=1.0)
        self.max_candidates = self._cfg_int("SONDRA_MEMORY_MAX_CANDIDATES", 1200, minimum=100)
        self.query_cache_enabled = self._cfg_bool("SONDRA_MEMORY_QUERY_CACHE", True)
        self.query_cache_ttl_sec = self._cfg_int("SONDRA_MEMORY_QUERY_CACHE_TTL_SEC", 180, minimum=30)
        self.memory_query_cache: dict[str, dict[str, Any]] = {}
        self.memory_embedding_cache: dict[str, list[float]] = {}
        self.memory_feedback_history: dict[int, list[bool]] = {}
        self.max_search_output_chars = self._cfg_int(
            "SONDRA_MEMORY_MAX_CHARS",
            self.MAX_SEARCH_OUTPUT_CHARS,
            minimum=400,
            maximum=20000,
        )

        # Embedding settings
        self.embed_provider = self._cfg_str("SONDRA_MEMORY_EMBED_PROVIDER", "local").lower()
        self.embed_model = self._cfg_str("SONDRA_MEMORY_EMBED_MODEL", "nomic-embed-text")
        self.embed_base_url = self._cfg_str("SONDRA_MEMORY_EMBED_BASE_URL", "http://127.0.0.1:11434")
        self.sync_mode = self._cfg_str("SONDRA_MEMORY_SYNC_MODE", "off").lower()
        self.sync_interval_sec = self._cfg_int("SONDRA_MEMORY_SYNC_INTERVAL_SEC", 90, minimum=30)

        self._last_embed_provider_used: str = "local"
        self._last_embed_error: str = ""
        self._last_sync_run: datetime = datetime.fromtimestamp(0, tz=UTC)
        self._vector_available: bool = True
        self.amygdala = Amygdala()
        self.signal_mode = "prompt"
        self.prompt_memory_trigger_enabled = True
        self.emotion_tracking_enabled = True
        self.overflow_policy = "last_20_messages"

        logger.info("[MEMORY] Using DB path: %s", self.db_path)
        self._init_db()
        with contextlib.suppress(Exception):
            self.prune_conversation_noise()
        with contextlib.suppress(Exception):
            self.prune_semantic_memory_noise()
        with contextlib.suppress(Exception):
            self.prune_profile_facts_noise()
        with contextlib.suppress(Exception):
            self.prune_expired_semantic()

    def _signal_list(self, file_name: str, *path: str) -> list[Any]:
        return self.signal_catalog.get_list(file_name, *path)

    def _signal_map(self, file_name: str, *path: str) -> dict[str, Any]:
        return self.signal_catalog.get_mapping(file_name, *path)

    def _signal_value(self, file_name: str, *path: str, default: Any = None) -> Any:
        return self.signal_catalog.get_value(file_name, *path, default=default)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(UTC).isoformat()

    def _normalize_session_id(self, session_id: str | None = None) -> str:
        value = str(session_id or "").strip()
        return value if value else self.LEGACY_SESSION_ID

    def _load_memory_config(self) -> dict[str, str]:
        config: dict[str, str] = {}
        config_path = Path(__file__).resolve().parents[2] / "config" / "memory_config.yaml"
        with contextlib.suppress(Exception):
            for raw_line in config_path.read_text(encoding="utf-8").splitlines():
                line = str(raw_line or "").strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                clean_key = str(key or "").strip()
                if not clean_key:
                    continue
                clean_value = str(value or "").strip()
                if clean_value.startswith(("'", '"')) and clean_value.endswith(("'", '"')) and len(clean_value) >= 2:
                    clean_value = clean_value[1:-1]
                config[clean_key] = clean_value
                config[clean_key.upper()] = clean_value
        return config

    def _cfg_raw(self, key: str) -> str:
        candidate = str(key or "").strip()
        upper_candidate = candidate.upper()
        keys = [candidate, upper_candidate]

        for raw_key in keys:
            raw = self._memory_config.get(raw_key, "")
            if raw:
                return str(raw).strip()
        return ""

    def _cfg_str(self, key: str, default: str) -> str:
        raw = self._cfg_raw(key)
        return raw if raw else str(default)

    def _cfg_bool(self, key: str, default: bool) -> bool:
        raw = self._cfg_raw(key).lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    def _cfg_int(self, key: str, default: int, minimum: int = 1, maximum: int = 100000) -> int:
        raw = self._cfg_raw(key)
        if not raw:
            return default
        try:
            value = int(raw)
        except Exception:
            return default
        return max(minimum, min(value, maximum))

    def _cfg_float(self, key: str, default: float, minimum: float = 0.0, maximum: float = 10.0) -> float:
        raw = self._cfg_raw(key)
        if not raw:
            return default
        try:
            value = float(raw)
        except Exception:
            return default
        return max(minimum, min(value, maximum))

    def _bool_env(self, key: str, default: bool) -> bool:
        raw = str(os.getenv(key, "")).strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    def _int_env(self, key: str, default: int, minimum: int = 1, maximum: int = 100000) -> int:
        raw = str(os.getenv(key, "")).strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except Exception:
            return default
        return max(minimum, min(value, maximum))

    def _float_env(self, key: str, default: float, minimum: float = 0.0, maximum: float = 10.0) -> float:
        raw = str(os.getenv(key, "")).strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except Exception:
            return default
        return max(minimum, min(value, maximum))

    def _resolve_memory_dir(self, preferred: str) -> Path:
        preferred_path = Path(preferred)
        if not preferred_path.is_absolute():
            preferred_path = Path(__file__).resolve().parents[2] / preferred_path
        try:
            preferred_path.mkdir(parents=True, exist_ok=True)
            return preferred_path
        except Exception:
            fallback = Path(__file__).resolve().parents[2] / "memory"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def _resolve_logs_dir(self, preferred: str) -> Path:
        preferred_path = Path(str(preferred or "logs").strip() or "logs")
        if not preferred_path.is_absolute():
            preferred_path = Path(__file__).resolve().parents[2] / preferred_path
        try:
            preferred_path.mkdir(parents=True, exist_ok=True)
            return preferred_path
        except Exception:
            fallback = self.memory_dir / "logs"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def _log_memory_update_event(self, event: dict[str, Any]) -> None:
        ts = str(event.get("ts") or self.now_iso())
        record = {"ts": ts, **{k: v for k, v in dict(event or {}).items() if k != "ts"}}

        with contextlib.suppress(Exception):
            with self.memory_events_jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        with contextlib.suppress(Exception):
            old_val = float(event.get("old", 0.0) or 0.0)
            new_val = float(event.get("new", 0.0) or 0.0)
            confidence_val = float(event.get("confidence", 0.0) or 0.0)
            delta_val = float(event.get("delta", 0.0) or 0.0)
            reason = str(event.get("reason", "") or "")
            memory_id = int(event.get("memory_id", 0) or 0)
            with self.memory_debug_log_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    f"[{ts}] id={memory_id} {old_val:.6f}->{new_val:.6f} "
                    f"conf={confidence_val:.2f} delta={delta_val:+.6f} reason={reason}\n"
                )

    @staticmethod
    def _read_jsonl_tail(path: Path, limit: int = 1, max_bytes: int = 65536) -> list[dict[str, Any]]:
        target_limit = max(1, int(limit))
        if not path.exists():
            return []
        try:
            with path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                if size <= 0:
                    return []
                read_size = min(size, max(1024, int(max_bytes)))
                fh.seek(max(0, size - read_size))
                payload = fh.read(read_size)
        except Exception:
            return []

        lines = payload.decode("utf-8", errors="replace").splitlines()
        events: list[dict[str, Any]] = []
        for line in reversed(lines):
            raw = str(line or "").strip()
            if not raw:
                continue
            try:
                loaded = json.loads(raw)
            except Exception:
                continue
            if isinstance(loaded, dict):
                events.append(loaded)
                if len(events) >= target_limit:
                    break
        return list(reversed(events))

    @staticmethod
    def _normalize_signal_text(text: str) -> str:
        return normalize_signal_text(text)

    def _contains_question_hint(self, lowered_value: str, tokens: list[str]) -> bool:
        value = self._normalize_signal_text(lowered_value)
        if not value:
            return True
        question_phrases = self._signal_list("prompt_memory", "question_phrases")
        question_tokens = set(self._signal_list("prompt_memory", "question_tokens"))
        if any(phrase in value for phrase in question_phrases):
            return True
        token_values = [self._normalize_signal_text(token) for token in tokens if self._normalize_signal_text(token)]
        if len(token_values) <= 6 and any(token in question_tokens for token in token_values):
            return True
        return False

    def _feedback_signal_from_text(self, text: Any) -> float:
        normalized = self._normalize_signal_text(str(text or ""))
        if not normalized:
            return 0.5
        negative_phrases = self._signal_list("prompt_memory", "feedback", "negative_phrases")
        positive_phrases = self._signal_list("prompt_memory", "feedback", "positive_phrases")
        if any(phrase in normalized for phrase in negative_phrases):
            return 0.0
        if any(phrase in normalized for phrase in positive_phrases):
            return 1.0
        return 0.5

    def _classify_failure_reason(self, failure_reason: str | None) -> str:
        normalized = self._normalize_signal_text(failure_reason or "")
        if not normalized:
            return ""
        contradiction_signals = self._signal_list("prompt_memory", "failure_reason", "contradiction_signals")
        transient_signals = self._signal_list("prompt_memory", "failure_reason", "transient_signals")
        logic_signals = self._signal_list("prompt_memory", "failure_reason", "logic_signals")
        if any(signal in normalized for signal in contradiction_signals):
            return "contradiction"
        if any(signal in normalized for signal in logic_signals):
            return "logic"
        if any(signal in normalized for signal in transient_signals):
            return "transient"
        return "generic"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        with contextlib.suppress(Exception):
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _table_columns(self, conn: sqlite3.Connection, table_name: str) -> set[str]:
        return {
            str(row[1] or "").strip().lower()
            for row in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        }

    def _legacy_session_id_for_existing_rows(self, conn: sqlite3.Connection) -> str:
        with contextlib.suppress(Exception):
            rows = conn.execute(
                """
                SELECT DISTINCT session_id
                FROM conversation_messages
                WHERE session_id IS NOT NULL AND TRIM(session_id) != ''
                LIMIT 2
                """
            ).fetchall()
            sessions = [str(row[0] or "").strip() for row in rows if str(row[0] or "").strip()]
            if len(sessions) == 1:
                return sessions[0]
        return self.LEGACY_SESSION_ID

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        alter_sql: str,
    ) -> None:
        if column_name.lower() not in self._table_columns(conn, table_name):
            conn.execute(alter_sql)

    def _migrate_profile_facts_schema(self, conn: sqlite3.Connection, legacy_session_id: str) -> None:
        columns = self._table_columns(conn, "profile_facts")
        create_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'profile_facts'"
        ).fetchone()
        create_sql = str(create_row[0] or "").lower() if create_row else ""
        needs_rebuild = "session_id" not in columns or "fact text unique" in create_sql
        if not needs_rebuild:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_profile_facts_session_fact ON profile_facts(session_id, fact)"
            )
            return

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profile_facts_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL DEFAULT '__legacy__',
                fact TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.8,
                updated_at TEXT NOT NULL,
                UNIQUE(session_id, fact)
            )
            """
        )
        if "session_id" in columns:
            conn.execute(
                """
                INSERT OR IGNORE INTO profile_facts_new (id, session_id, fact, importance, updated_at)
                SELECT id,
                       COALESCE(NULLIF(TRIM(session_id), ''), ?),
                       fact,
                       importance,
                       updated_at
                FROM profile_facts
                WHERE fact IS NOT NULL AND TRIM(fact) != ''
                """,
                (legacy_session_id,),
            )
        else:
            conn.execute(
                """
                INSERT OR IGNORE INTO profile_facts_new (id, session_id, fact, importance, updated_at)
                SELECT id, ?, fact, importance, updated_at
                FROM profile_facts
                WHERE fact IS NOT NULL AND TRIM(fact) != ''
                """,
                (legacy_session_id,),
            )
        conn.execute("DROP TABLE profile_facts")
        conn.execute("ALTER TABLE profile_facts_new RENAME TO profile_facts")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_profile_facts_session_fact ON profile_facts(session_id, fact)"
        )

    def _make_query_cache_key(self, query: str, top_k: int, session_id: str | None = None) -> str:
        canonical_query = self._canonicalize_query(query)
        session_value = str(session_id or "").strip()
        payload = json.dumps(
            {
                "q": canonical_query,
                "k": int(top_k),
                "s": session_value,
                "hybrid": bool(self.hybrid_enabled),
                "mmr": bool(self.mmr_enabled),
                "temporal": bool(self.temporal_decay_enabled),
                "dim": int(self.embedding_dim),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()

    def get_cached(self, key: str) -> list[Any] | None:
        entry = self.memory_query_cache.get(str(key or ""))
        if not entry:
            return None
        try:
            ts = float(entry.get("time", 0.0) or 0.0)
        except Exception:
            ts = 0.0
        if (time.time() - ts) > float(self.query_cache_ttl_sec):
            self.memory_query_cache.pop(str(key or ""), None)
            return None
        data = entry.get("data")
        if not isinstance(data, list):
            return None
        return list(data)

    def set_cache(self, key: str, data: list[Any]) -> None:
        self.memory_query_cache[str(key or "")] = {
            "data": list(data or []),
            "time": time.time(),
        }

    def _read_query_cache(self, query: str, top_k: int, session_id: str | None = None) -> list[str] | None:
        if not self.query_cache_enabled:
            return None
        cache_key = self._make_query_cache_key(query, top_k, session_id=session_id)
        cached = self.get_cached(cache_key)
        if cached is None:
            return None
        values = [str(v) for v in cached if str(v).strip()]
        values = self._dedupe_memory_lines(values)
        return self._apply_content_char_budget(values)[: max(1, min(int(top_k), 20))]

    def _write_query_cache(
        self,
        query: str,
        top_k: int,
        results: list[str],
        session_id: str | None = None,
    ) -> None:
        if not self.query_cache_enabled:
            return
        cache_key = self._make_query_cache_key(query, top_k, session_id=session_id)
        payload = [str(v) for v in (results or []) if str(v).strip()]
        if not payload:
            return
        self.set_cache(cache_key, payload)

    def _make_structured_query_cache_key(
        self,
        query: str,
        top_k: int,
        candidate_limit: int,
        session_id: str | None = None,
    ) -> str:
        canonical_query = self._canonicalize_query(query)
        session_value = str(session_id or "").strip()
        payload = json.dumps(
            {
                "q": canonical_query,
                "k": int(top_k),
                "c": int(candidate_limit),
                "s": session_value,
                "hybrid": bool(self.hybrid_enabled),
                "mmr": bool(self.mmr_enabled),
                "temporal": bool(self.temporal_decay_enabled),
                "dim": int(self.embedding_dim),
                "type": "structured",
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()

    def _canonicalize_query(self, query: str) -> str:
        text = " ".join(str(query or "").strip().lower().split())
        while text and text[-1] in "?.!":
            text = text[:-1].rstrip()
        return text

    def _read_structured_query_cache(
        self,
        query: str,
        top_k: int,
        candidate_limit: int,
        session_id: str | None = None,
    ) -> list[dict[str, str | float | int]] | None:
        if not self.query_cache_enabled:
            return None
        cache_key = self._make_structured_query_cache_key(
            query,
            top_k,
            candidate_limit,
            session_id=session_id,
        )
        cached = self.get_cached(cache_key)
        if cached is None:
            return None
        rows: list[dict[str, str | float | int]] = []
        for item in cached:
            if not isinstance(item, dict):
                continue
            rows.append(dict(item))
        return self._dedupe_structured_rows(rows, top_k)

    def _write_structured_query_cache(
        self,
        query: str,
        top_k: int,
        candidate_limit: int,
        rows: list[dict[str, str | float | int]],
        session_id: str | None = None,
    ) -> None:
        if not self.query_cache_enabled:
            return
        payload = [dict(item) for item in (rows or []) if isinstance(item, dict)]
        if not payload:
            return
        cache_key = self._make_structured_query_cache_key(
            query,
            top_k,
            candidate_limit,
            session_id=session_id,
        )
        self.set_cache(cache_key, payload)

    def _dedupe_structured_rows(
        self,
        rows: list[dict[str, str | float | int]],
        top_k: int,
    ) -> list[dict[str, str | float | int]]:
        seen: set[str] = set()
        result: list[dict[str, str | float | int]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role", "") or "").strip().lower()
            body = str(row.get("body", "") or "").strip()
            if not body:
                continue
            if self._is_noisy_memory_body(body):
                continue
            key = f"{role}:{body.lower()}"
            if key in seen:
                continue
            seen.add(key)
            result.append(dict(row))
            if len(result) >= max(1, min(int(top_k), 20)):
                break
        return result

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL,
                    session_id TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_ts ON conversation_messages(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_session_id ON conversation_messages(session_id, id DESC)")
            
            # FTS5 tablosu
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS conversation_messages_fts
                USING fts5(content, content='conversation_messages', content_rowid='id')
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS conv_ai AFTER INSERT ON conversation_messages BEGIN
                    INSERT INTO conversation_messages_fts(rowid, content) VALUES (new.id, new.content);
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS conv_ad AFTER DELETE ON conversation_messages BEGIN
                    INSERT INTO conversation_messages_fts(conversation_messages_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS conv_au AFTER UPDATE ON conversation_messages BEGIN
                    INSERT INTO conversation_messages_fts(conversation_messages_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                    INSERT INTO conversation_messages_fts(rowid, content) VALUES (new.id, new.content);
                END
            """)
            # Full FTS rebuild on every startup causes long initialization on large DBs.
            # Rebuild only when FTS is empty but base table has data.
            conv_count = int(
                conn.execute("SELECT COUNT(*) FROM conversation_messages").fetchone()[0] or 0
            )
            fts_count = int(
                conn.execute("SELECT COUNT(*) FROM conversation_messages_fts").fetchone()[0] or 0
            )
            if conv_count > 0 and fts_count == 0:
                conn.execute("INSERT INTO conversation_messages_fts(conversation_messages_fts) VALUES ('rebuild')")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_emotions (
                    message_id INTEGER PRIMARY KEY,
                    role TEXT NOT NULL,
                    anger REAL NOT NULL DEFAULT 0.0,
                    frustration REAL NOT NULL DEFAULT 0.0,
                    happiness REAL NOT NULL DEFAULT 0.0,
                    sadness REAL NOT NULL DEFAULT 0.0,
                    neutral REAL NOT NULL DEFAULT 1.0,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(message_id) REFERENCES conversation_messages(id) ON DELETE CASCADE
                )
            """)
            
            # Semantic memory tablosu
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT '__legacy__',
                    content TEXT,
                    importance REAL,
                    created_at TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT '__legacy__',
                    goal TEXT,
                    current_step TEXT,
                    updated_at TEXT,
                    UNIQUE(session_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS profile_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT '__legacy__',
                    fact TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.8,
                    updated_at TEXT NOT NULL,
                    UNIQUE(session_id, fact)
                )
            """)
            
            # Embedding cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_embedding_cache (
                    message_id INTEGER PRIMARY KEY,
                    dim INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT UNIQUE NOT NULL,
                    embedding TEXT
                )
            """)
            with contextlib.suppress(Exception):
                conn.execute("ALTER TABLE memory_index ADD COLUMN embedding TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_index_content ON memory_index(content)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_index_state (
                    id INTEGER PRIMARY KEY,
                    last_indexed_message_id INTEGER NOT NULL DEFAULT 0,
                    last_sync_at TEXT,
                    last_sync_mode TEXT
                )
            """)
            conn.execute("""
                INSERT OR IGNORE INTO memory_index_state (id, last_indexed_message_id, last_sync_at, last_sync_mode)
                VALUES (1, 0, NULL, NULL)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT '__legacy__',
                    owner_agent_id TEXT NOT NULL DEFAULT '',
                    task_text TEXT NOT NULL,
                    schedule_time TEXT NOT NULL DEFAULT '--:--',
                    schedule_type TEXT NOT NULL DEFAULT 'once',
                    cron_expression TEXT NOT NULL DEFAULT '',
                    next_run TEXT NOT NULL DEFAULT '',
                    last_run TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'waiting',
                    worker_name TEXT NOT NULL DEFAULT '',
                    worker_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            scheduled_task_columns = {
                str(row[1] or "").strip().lower()
                for row in conn.execute("PRAGMA table_info('scheduled_tasks')").fetchall()
            }
            scheduled_task_alters = (
                ("session_id", "ALTER TABLE scheduled_tasks ADD COLUMN session_id TEXT NOT NULL DEFAULT '__legacy__'"),
                ("owner_agent_id", "ALTER TABLE scheduled_tasks ADD COLUMN owner_agent_id TEXT NOT NULL DEFAULT ''"),
                ("schedule_time", "ALTER TABLE scheduled_tasks ADD COLUMN schedule_time TEXT NOT NULL DEFAULT '--:--'"),
                ("schedule_type", "ALTER TABLE scheduled_tasks ADD COLUMN schedule_type TEXT NOT NULL DEFAULT 'once'"),
                ("cron_expression", "ALTER TABLE scheduled_tasks ADD COLUMN cron_expression TEXT NOT NULL DEFAULT ''"),
                ("next_run", "ALTER TABLE scheduled_tasks ADD COLUMN next_run TEXT NOT NULL DEFAULT ''"),
                ("last_run", "ALTER TABLE scheduled_tasks ADD COLUMN last_run TEXT NOT NULL DEFAULT ''"),
                ("status", "ALTER TABLE scheduled_tasks ADD COLUMN status TEXT NOT NULL DEFAULT 'waiting'"),
                ("worker_name", "ALTER TABLE scheduled_tasks ADD COLUMN worker_name TEXT NOT NULL DEFAULT ''"),
                ("worker_id", "ALTER TABLE scheduled_tasks ADD COLUMN worker_id TEXT NOT NULL DEFAULT ''"),
                ("created_at", "ALTER TABLE scheduled_tasks ADD COLUMN created_at TEXT NOT NULL DEFAULT ''"),
                ("updated_at", "ALTER TABLE scheduled_tasks ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''"),
            )
            for column_name, alter_sql in scheduled_task_alters:
                if column_name not in scheduled_task_columns:
                    conn.execute(alter_sql)
            legacy_session_id = self._legacy_session_id_for_existing_rows(conn)
            self._ensure_column(
                conn,
                "semantic_memory",
                "session_id",
                "ALTER TABLE semantic_memory ADD COLUMN session_id TEXT NOT NULL DEFAULT '__legacy__'",
            )
            self._ensure_column(
                conn,
                "task_state",
                "session_id",
                "ALTER TABLE task_state ADD COLUMN session_id TEXT NOT NULL DEFAULT '__legacy__'",
            )
            with contextlib.suppress(Exception):
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_task_state_session_id ON task_state(session_id)")
            conn.execute(
                "UPDATE semantic_memory SET session_id = ? WHERE session_id IS NULL OR TRIM(session_id) = '' OR session_id = '__legacy__'",
                (legacy_session_id,),
            )
            conn.execute(
                "UPDATE task_state SET session_id = ? WHERE session_id IS NULL OR TRIM(session_id) = '' OR session_id = '__legacy__'",
                (legacy_session_id,),
            )
            conn.execute(
                "UPDATE scheduled_tasks SET session_id = ? WHERE session_id IS NULL OR TRIM(session_id) = '' OR session_id = '__legacy__'",
                (legacy_session_id,),
            )
            self._migrate_profile_facts_schema(conn, legacy_session_id)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_memory_session ON semantic_memory(session_id, importance DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_state_session ON task_state(session_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_next_run ON scheduled_tasks(session_id, status, next_run ASC)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_owner ON scheduled_tasks(owner_agent_id, status)")
            
            conn.commit()
        finally:
            conn.close()

    def _normalize_role_label(self, role: str) -> str:
        return "Assistant" if str(role or "").strip().lower() == "assistant" else "User"

    def _safe_parse_timestamp(self, timestamp: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(str(timestamp or ""))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except Exception:
            return datetime.now(UTC)

    def _remove_time_prefix(self, content: str) -> str:
        text = str(content or "").strip()
        if not text.startswith("["):
            return text
        close2 = text.find("]", text.find("]") + 2)
        if close2 > 0:
            return text[close2 + 1:].strip()
        return text

    @staticmethod
    def _emotion_phrase_strength(normalized: str, weights: dict[str, float]) -> float:
        if not normalized or not weights:
            return 0.0
        best = 0.0
        total = 0.0
        for phrase, weight in weights.items():
            phrase_text = str(phrase or "").strip()
            if not phrase_text:
                continue
            if " " in phrase_text:
                matched = phrase_text in normalized
            else:
                matched = bool(
                    re.search(
                        rf"(?<![a-z0-9]){re.escape(phrase_text)}(?![a-z0-9])",
                        normalized,
                    )
                )
            if not matched:
                continue
            value = float(weight or 0.0)
            best = max(best, value)
            total += value
        if total <= 0.0:
            return 0.0
        return min(1.0, best + (0.25 * max(0.0, total - best)))

    def _emotion_intensity_multiplier(self, normalized: str, raw_text: str) -> float:
        if not normalized:
            return 1.0
        intensifiers = self._signal_map("emotion_signals", "intensity", "intensifiers")
        softeners = self._signal_map("emotion_signals", "intensity", "softeners")
        multiplier = 1.0
        for phrase, weight in intensifiers.items():
            if phrase in normalized:
                multiplier += float(weight)
        for phrase, weight in softeners.items():
            if phrase in normalized:
                multiplier -= float(weight)
        punctuation_boost = min(0.18, 0.04 * str(raw_text or "").count("!"))
        question_softener = 0.08 if str(raw_text or "").count("?") >= 1 else 0.0
        multiplier += punctuation_boost
        multiplier -= question_softener
        return self._clamp(multiplier, 0.72, 1.45)

    def _score_emotion_signals(self, text: str, role: str = "user") -> dict[str, float]:
        normalized = self._normalize_signal_text(text)
        base = {
            "anger": 0.0,
            "frustration": 0.0,
            "happiness": 0.0,
            "sadness": 0.0,
            "neutral": 1.0,
            "confidence": 0.0,
        }
        if not normalized:
            return base

        happiness_weights = self._signal_map("emotion_signals", "weights", "happiness")
        friendly_weights = self._signal_map("emotion_signals", "weights", "friendly")
        neutral_weights = self._signal_map("emotion_signals", "weights", "neutral")
        sadness_weights = self._signal_map("emotion_signals", "weights", "sadness")
        frustration_weights = self._signal_map("emotion_signals", "weights", "frustration")
        anger_weights = self._signal_map("emotion_signals", "weights", "anger")
        insult_weights = self._signal_map("emotion_signals", "weights", "insult")

        intensity = self._emotion_intensity_multiplier(normalized, text)
        positive_seed = self._emotion_phrase_strength(normalized, happiness_weights)
        friendly_seed = self._emotion_phrase_strength(normalized, friendly_weights)
        neutral_seed = self._emotion_phrase_strength(normalized, neutral_weights)
        sadness_seed = self._emotion_phrase_strength(normalized, sadness_weights)
        frustration_seed = self._emotion_phrase_strength(normalized, frustration_weights)
        anger_seed = self._emotion_phrase_strength(normalized, anger_weights)
        insult_seed = self._emotion_phrase_strength(normalized, insult_weights)

        happiness = self._clamp((positive_seed * intensity) + (friendly_seed * 0.35), 0.0, 1.0)
        anger = self._clamp(max(anger_seed, insult_seed * 0.58) * intensity, 0.0, 1.0)
        frustration = self._clamp(max(frustration_seed, insult_seed * 0.48) * intensity, 0.0, 1.0)
        sadness = self._clamp(
            max(sadness_seed, insult_seed * 0.82, frustration_seed * 0.38, anger_seed * 0.28)
            * min(1.25, 0.92 + (intensity * 0.28)),
            0.0,
            1.0,
        )
        if happiness > 0.0:
            sadness = self._clamp(sadness - (happiness * 0.22), 0.0, 1.0)
            frustration = self._clamp(frustration - (happiness * 0.12), 0.0, 1.0)

        strongest_non_neutral = max(happiness, sadness, frustration, anger)
        neutral = self._clamp(
            max(
                neutral_seed,
                friendly_seed * 0.7,
                1.0 - (strongest_non_neutral * 1.15),
            ),
            0.0,
            1.0,
        )
        confidence = max(strongest_non_neutral, neutral_seed * 0.55, friendly_seed * 0.4, 0.35)
        if str(role or "").strip().lower() == "assistant":
            confidence = min(confidence, 0.5)

        return {
            "anger": round(anger, 4),
            "frustration": round(frustration, 4),
            "happiness": round(happiness, 4),
            "sadness": round(sadness, 4),
            "neutral": round(neutral, 4),
            "confidence": round(confidence, 4),
        }

    def _store_message_emotion_with_cursor(
        self,
        cursor: sqlite3.Cursor,
        message_id: int,
        role: str,
        content: str,
        timestamp: str,
    ) -> None:
        if not self.emotion_tracking_enabled:
            return
        target_id = int(message_id or 0)
        if target_id <= 0:
            return
        emotion = self._score_emotion_signals(content, role=role)
        cursor.execute(
            """
            INSERT OR REPLACE INTO message_emotions (
                message_id, role, anger, frustration, happiness, sadness, neutral, confidence, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                target_id,
                str(role or "").strip().lower(),
                float(emotion.get("anger", 0.0) or 0.0),
                float(emotion.get("frustration", 0.0) or 0.0),
                float(emotion.get("happiness", 0.0) or 0.0),
                float(emotion.get("sadness", 0.0) or 0.0),
                float(emotion.get("neutral", 1.0) or 1.0),
                float(emotion.get("confidence", 0.0) or 0.0),
                str(timestamp or self.now_iso()),
            ),
        )

    def get_latest_emotion_signal(
        self,
        role: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        session_value = str(session_id or "").strip()
        conn = self._connect()
        try:
            if str(role or "").strip():
                row = conn.execute(
                    """
                    SELECT me.message_id, me.role, me.anger, me.frustration, me.happiness,
                           me.sadness, me.neutral, me.confidence, me.created_at
                    FROM message_emotions me
                    JOIN conversation_messages cm ON cm.id = me.message_id
                    WHERE me.role = ?
                      AND (? = '' OR cm.session_id = ?)
                    ORDER BY me.message_id DESC
                    LIMIT 1
                    """,
                    (str(role or "").strip().lower(), session_value, session_value),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT me.message_id, me.role, me.anger, me.frustration, me.happiness,
                           me.sadness, me.neutral, me.confidence, me.created_at
                    FROM message_emotions me
                    JOIN conversation_messages cm ON cm.id = me.message_id
                    WHERE (? = '' OR cm.session_id = ?)
                    ORDER BY me.message_id DESC
                    LIMIT 1
                    """,
                    (session_value, session_value),
                ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        scores = {
            "anger": float(row[2] or 0.0),
            "frustration": float(row[3] or 0.0),
            "happiness": float(row[4] or 0.0),
            "sadness": float(row[5] or 0.0),
            "neutral": float(row[6] or 0.0),
        }
        top_label = max(scores, key=scores.get) if scores else "neutral"
        return {
            "message_id": int(row[0] or 0),
            "role": str(row[1] or ""),
            "scores": scores,
            "top_emotion": top_label,
            "top_score": float(scores.get(top_label, 0.0) or 0.0),
            "confidence": float(row[7] or 0.0),
            "created_at": str(row[8] or ""),
        }

    def analyze_prompt_signal(self, text: str) -> dict[str, Any]:
        normalized = self._normalize_signal_text(text)
        if not normalized:
            return {
                "signal": "none",
                "force_memory_search": False,
                "allow_memory_search": False,
                "suggested_query": "",
                "route": "prompt_signal",
            }

        correction_phrases = self._signal_list("prompt_memory", "prompt_signal", "correction_phrases")
        recall_phrases = self._signal_list("prompt_memory", "prompt_signal", "recall_phrases")
        memory_intent_terms = self._signal_list("prompt_memory", "prompt_signal", "memory_intent_terms")

        signal = "none"
        force_memory = False
        suggested_query = ""
        if any(phrase in normalized for phrase in correction_phrases):
            signal = "correction"
            force_memory = True
            suggested_query = normalized
        elif any(phrase in normalized for phrase in recall_phrases):
            signal = "recall"
            force_memory = True
            suggested_query = normalized
        elif any(term in normalized for term in memory_intent_terms):
            signal = "memory_intent"
            force_memory = True
            suggested_query = normalized

        return {
            "signal": signal,
            "force_memory_search": bool(force_memory),
            "allow_memory_search": bool(force_memory),
            "suggested_query": suggested_query,
            "route": "prompt_signal",
        }

    def _is_noisy_memory_body(self, text: str) -> bool:
        raw = str(text or "").strip()
        lowered = raw.lower()
        if not lowered:
            return True
        noisy = [
            "<function=",
            "<parameter=",
            "</function>",
            "<tool_result>",
            "</tool_result>",
            "[memory result]",
            "execution was cancelled",
            "waiting for new instructions",
            "memory read is complete. please continue.",
            "memory retrieval completed.",
            "from memory:",
        ]
        if any(x in lowered for x in noisy):
            return True
        _, body = self._extract_role_and_body(raw)
        if self._looks_like_tool_payload_json(body):
            return True
        return False

    def _looks_like_tool_payload_json(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw or not raw.startswith("{") or not raw.endswith("}"):
            return False
        try:
            loaded = json.loads(raw)
        except Exception:
            return False
        if not isinstance(loaded, dict):
            return False
        name = loaded.get("name")
        has_arguments = "arguments" in loaded
        if isinstance(name, str) and bool(name.strip()) and has_arguments:
            return True
        function_name = loaded.get("function")
        has_parameters = "parameters" in loaded
        if isinstance(function_name, str) and bool(function_name.strip()) and has_parameters:
            return True
        return False

    def _memory_line_dedupe_key(self, line: str) -> str:
        value = str(line or "").strip()
        if not value:
            return ""
        role, body = self._extract_role_and_body(value)
        body_value = str(body or "").strip()
        if body_value:
            return f"{str(role or '').strip().lower()}:{body_value.lower()}"
        return value.lower()

    def _dedupe_memory_lines(self, lines: list[str], top_k: int | None = None) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for line in lines:
            value = str(line or "").strip()
            if not value:
                continue
            key = self._memory_line_dedupe_key(value)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(value)
            if top_k is not None and len(deduped) >= max(1, int(top_k)):
                break
        return deduped

    def _auto_context_dedupe_key(self, line: str) -> str:
        value = str(line or "").strip()
        if not value:
            return ""
        lowered = value.lower()
        for prefix in ("[exact] ", "[recent] ", "[search] ", "[sem] "):
            if lowered.startswith(prefix):
                value = value[len(prefix):].strip()
                lowered = value.lower()
                break
        if lowered.startswith("[profile]") or lowered.startswith("[task]"):
            return lowered
        if ":" in value:
            head, tail = value.split(":", 1)
            head_norm = head.strip().lower()
            if head_norm in {"user", "assistant"}:
                return f"{head_norm}:{tail.strip().lower()}"
        return lowered

    def _apply_content_char_budget(self, rows: list[str], max_chars: int | None = None) -> list[str]:
        budget = int(max_chars or self.max_search_output_chars)
        budget = max(200, budget)
        total = 0
        selected: list[str] = []
        for row in rows:
            value = str(row or "").strip()
            if not value:
                continue
            size = len(value)
            if selected and (total + size) > budget:
                break
            selected.append(value)
            total += size
        return selected

    def _format_memory_line(self, timestamp: str, role: str, content: str) -> str:
        parsed_ts = self._safe_parse_timestamp(timestamp)
        relative = get_relative_time_label(parsed_ts)
        role_label = self._normalize_role_label(role)
        body = self._remove_time_prefix(content)
        parsed_role, parsed_body = self._extract_role_and_body(body)
        if parsed_role in {"user", "assistant"} and parsed_body:
            body = parsed_body
        return f"[{relative}] {role_label}: {body}"

    def add_conversation_messages(self, messages: list[tuple[str, str, str, str]]) -> None:
        if not messages:
            return
        rows_to_insert = []
        for timestamp, role, content, session_id in messages:
            role_lower = str(role or "").strip().lower()
            if role_lower not in {"user", "assistant"}:
                continue
            raw_body = str(content or "").strip()
            if not raw_body:
                continue
            if self._looks_like_tool_payload_json(raw_body):
                continue
            ts_value = str(timestamp or self.now_iso())
            stored_content = self._format_memory_line(ts_value, role_lower, raw_body)
            if not stored_content or self._is_noisy_memory_body(stored_content):
                continue
            rows_to_insert.append((ts_value, role_lower, stored_content, str(session_id or "")))

        if rows_to_insert:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                for ts_value, role_lower, stored_content, session_value in rows_to_insert:
                    cursor.execute(
                        """
                        INSERT INTO conversation_messages (timestamp, role, content, session_id)
                        VALUES (?, ?, ?, ?)
                        """,
                        (ts_value, role_lower, stored_content, session_value),
                    )
                    self._store_message_emotion_with_cursor(
                        cursor,
                        int(cursor.lastrowid or 0),
                        role_lower,
                        stored_content,
                        ts_value,
                    )
                self.memory_query_cache.clear()
                conn.commit()
            finally:
                conn.close()

    def recent_conversation(self, limit: int = 50, session_id: str | None = None) -> list[ConversationMessage]:
        session_value = str(session_id or "").strip()
        conn = self._connect()
        try:
            if session_value:
                rows = conn.execute(
                    """
                    SELECT id, timestamp, role, content, session_id
                    FROM conversation_messages
                    WHERE session_id = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (session_value, max(1, int(limit))),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, timestamp, role, content, session_id
                    FROM conversation_messages
                    ORDER BY id DESC LIMIT ?
                    """,
                    (max(1, int(limit)),),
                ).fetchall()
        finally:
            conn.close()
        return [ConversationMessage(*row) for row in rows]

    def get_conversation_message_by_id(self, message_id: int) -> dict[str, str | int] | None:
        target_id = int(message_id or 0)
        if target_id <= 0:
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT id, timestamp, role, content, session_id
                FROM conversation_messages
                WHERE id = ?
                LIMIT 1
                """,
                (target_id,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None

        content = str(row[3] or "")
        stored_role = str(row[2] or "").strip().lower()
        parsed_role, body = self._extract_role_and_body(content)
        role = stored_role if stored_role in {"user", "assistant"} else parsed_role
        parsed_ts = self._safe_parse_timestamp(str(row[1] or ""))
        return {
            "id": int(row[0]),
            "timestamp": str(row[1] or ""),
            "exact_date": parsed_ts.strftime("%d/%m/%Y"),
            "relative_time": get_relative_time_label(parsed_ts),
            "role": role,
            "content": content,
            "body": body,
            "session_id": str(row[4] or ""),
            "source": f"memory.db#msg-{int(row[0])}",
            "source_path": "memory.db",
            "source_line": int(row[0]),
        }

    def _tokenize(self, text: str) -> list[str]:
        raw = str(text or "").lower()
        for ch in "\"'`.,;:!?()[]{}<>/\\|+-=*#@$%^&~":
            raw = raw.replace(ch, " ")
        return [p for p in raw.split() if p.strip()]

    def _search_fts_rows(self, fts_query: str, limit: int, session_id: str | None = None) -> list[tuple]:
        session_value = str(session_id or "").strip() or None
        conn = self._connect()
        try:
            return conn.execute(
                """
                SELECT cm.id, cm.timestamp, cm.role, cm.content, bm25(conversation_messages_fts) AS fts_rank
                FROM conversation_messages_fts
                JOIN conversation_messages cm ON cm.id = conversation_messages_fts.rowid
                WHERE conversation_messages_fts MATCH ?
                  AND (? IS NULL OR cm.session_id = ?)
                ORDER BY fts_rank ASC, cm.id DESC LIMIT ?
                """,
                (fts_query, session_value, session_value, max(1, int(limit))),
            ).fetchall()
        finally:
            conn.close()

    def _extract_role_and_body(self, content: str) -> tuple[str, str]:
        text = str(content or "").strip()
        if not text:
            return "assistant", ""
        if "] " in text and text.startswith("["):
            text = text.split("] ", 1)[-1].strip()
        lowered = text.lower()
        if lowered.startswith("user:"):
            return "user", text[5:].strip()
        if lowered.startswith("assistant:"):
            return "assistant", text[10:].strip()
        return "assistant", text

    def _recency_score(self, timestamp: str) -> float:
        age_days = max(0.0, (datetime.now(UTC) - self._safe_parse_timestamp(timestamp)).total_seconds() / 86400.0)
        return 1.0 / (1.0 + (age_days / 30.0))

    def _lexical_score(self, query_tokens: list[str], candidate_text: str) -> float:
        if not query_tokens:
            return 0.0
        candidate_tokens = set(self._tokenize(candidate_text))
        if not candidate_tokens:
            return 0.0
        overlap = set(query_tokens).intersection(candidate_tokens)
        return len(overlap) / max(1, len(query_tokens))

    def _fts_rank_to_score(self, rank: float | None) -> float:
        return 1.0 / (1.0 + abs(float(rank or 0.0)))

    def _combine_scores(
        self,
        *,
        lexical: float,
        fts_score: float | None,
        vector_score: float | None,
        recency: float,
        importance: float | None = None,
        created_at: str | None = None,
    ) -> float:
        components = [
            (0.20, float(lexical)),
            (0.30, None if fts_score is None else float(fts_score)),
            (0.35, None if vector_score is None else float(vector_score)),
            (0.15, float(recency)),
        ]
        active = [(w, s) for (w, s) in components if s is not None]
        if not active:
            return 0.0
        weight_sum = sum(weight for weight, _ in active)
        if weight_sum <= 0:
            return 0.0
        score = sum(weight * value for weight, value in active)
        base_score = score / weight_sum

        importance_value = 0.5
        if importance is not None:
            try:
                importance_value = float(importance)
            except Exception:
                importance_value = 0.5
        importance_value = self.amygdala.decay(importance_value, created_at)
        final_score = self.amygdala.apply(base_score, importance_value)
        return final_score

    def _extract_semantic_source_id(self, source: str) -> int:
        token = str(source or "").strip().lower()
        prefix = "semantic_memory#"
        if not token.startswith(prefix):
            return 0
        tail = token[len(prefix):].strip()
        digits: list[str] = []
        for ch in tail:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if not digits:
            return 0
        return int("".join(digits))

    def _reinforce_selected_rows(self, selected_rows: list[Any]) -> None:
        if not selected_rows:
            return
        conn = self._connect()
        try:
            cursor = conn.cursor()
            updated = False
            for row in selected_rows:
                try:
                    if isinstance(row, dict):
                        importance = row.get("importance", 0.5)
                        text = str(row.get("content", "") or "")
                        source = str(row.get("source", "") or "")
                    else:
                        importance = getattr(row, "importance", 0.5)
                        text = str(getattr(row, "content", "") or "")
                        source = str(getattr(row, "source", "") or "")
                    semantic_id = self._extract_semantic_source_id(source)
                    if semantic_id <= 0:
                        continue

                    if self._is_warning_semantic(text):
                        continue

                    base_importance = float(importance)
                    lowered_text = text.lower()
                    semantic_bonus = 0.0
                    if "user" in lowered_text or "name" in lowered_text:
                        semantic_bonus += 0.1
                    if "goal" in lowered_text or "plan" in lowered_text:
                        semantic_bonus += 0.1
                    if "project" in lowered_text or "system" in lowered_text:
                        semantic_bonus += 0.1

                    boosted = base_importance + semantic_bonus
                    base = float(boosted)
                    if base >= 0.95:
                        continue
                    if base > 0.85:
                        new_importance = base + (1.0 - base) * 0.02
                    else:
                        new_importance = base + (1.0 - base) * 0.1
                    new_importance = min(new_importance, 1.0)
                    cursor.execute(
                        "UPDATE semantic_memory SET importance = ? WHERE id = ?",
                        (new_importance, int(semantic_id)),
                    )
                    if cursor.rowcount > 0:
                        updated = True
                except Exception:
                    logger.debug("[MEMORY] reinforce selected row update skipped", exc_info=True)
            if updated:
                conn.commit()
        except Exception:
            logger.debug("[MEMORY] reinforce selected rows failed", exc_info=True)
        finally:
            conn.close()

    def _temporal_decay_score(self, timestamp: str) -> float:
        if not self.temporal_decay_enabled:
            return 1.0
        age_days = max(0.0, (datetime.now(UTC) - self._safe_parse_timestamp(timestamp)).total_seconds() / 86400.0)
        return math.exp(-age_days / 180.0)

    def _embed_text(self, text: str) -> list[float]:
        cleaned = str(text or "").strip()
        if not cleaned:
            return []

        key = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        cached = self.memory_embedding_cache.get(key)
        if cached:
            return cached

        endpoint_base = str(self.embed_base_url or "").strip()
        endpoint = (
            f"{endpoint_base.rstrip('/')}/api/embeddings"
            if endpoint_base
            else OLLAMA_EMBED_URL
        )
        model = str(self.embed_model or "").strip() or EMBED_MODEL

        try:
            resp = requests.post(
                endpoint,
                json={"model": model, "prompt": cleaned},
                timeout=float(self.embed_timeout_sec),
            )
            resp.raise_for_status()
            vec = resp.json().get("embedding") or []
            if not isinstance(vec, list):
                vec = []
            vec = [float(v) for v in vec]
            self.memory_embedding_cache[key] = vec
            self._last_embed_provider_used = "ollama"
            self._last_embed_error = ""
            return vec
        except Exception as e:
            self._last_embed_provider_used = "fallback"
            self._last_embed_error = str(e)
            self.memory_embedding_cache[key] = []
            return []

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0

        size = min(len(a), len(b))
        dot = sum(a[i] * b[i] for i in range(size))
        na = math.sqrt(sum(a[i] * a[i] for i in range(size)))
        nb = math.sqrt(sum(b[i] * b[i] for i in range(size)))

        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def store_embedding(self, text: str, embedding: list[float]) -> None:
        content = str(text or "").strip()
        if not content:
            return
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO memory_index (content, embedding) VALUES (?, NULL)",
                (content,),
            )
            conn.execute(
                "UPDATE memory_index SET embedding = ? WHERE content = ?",
                (json.dumps(embedding), content),
            )
            conn.commit()
        finally:
            conn.close()

    def _load_embeddings_for_rows(
        self,
        rows: list[tuple],
        *,
        compute_missing: bool = True,
    ) -> dict[int, list[float]]:
        embeddings = {}
        if not rows:
            return embeddings

        conn = self._connect()
        try:
            for row in rows:
                row_id = int(row[0])
                body = self._extract_role_and_body(str(row[3] or ""))[1]
                if not body:
                    embeddings[row_id] = []
                    continue
                db_row = conn.execute(
                    "SELECT embedding FROM memory_index WHERE content = ?",
                    (body,),
                ).fetchone()
                vec: list[float]
                if db_row and db_row[0]:
                    try:
                        vec = [float(v) for v in json.loads(str(db_row[0]))]
                    except Exception:
                        if compute_missing:
                            vec = self._embed_text(body)
                            if vec:
                                self.store_embedding(body, vec)
                        else:
                            vec = []
                else:
                    if compute_missing:
                        vec = self._embed_text(body)
                        if vec:
                            self.store_embedding(body, vec)
                    else:
                        vec = []
                embeddings[row_id] = vec
        finally:
            conn.close()

        return embeddings

    def _apply_mmr_selection(self, hits: list[MemorySearchHit], embeddings: dict[int, list[float]], top_k: int) -> list[MemorySearchHit]:
        if not self.mmr_enabled or len(hits) <= top_k:
            return hits[:top_k]
        
        selected = []
        remaining = list(hits)
        
        while remaining and len(selected) < top_k:
            best_idx = -1
            best_score = -1e9
            for idx, cand in enumerate(remaining):
                relevance = cand.final_score
                diversity = 0.0
                cand_vec = embeddings.get(cand.id, [])
                if selected and cand_vec:
                    max_sim = max(self._cosine_similarity(cand_vec, embeddings.get(s.id, [])) for s in selected)
                    diversity = max_sim
                mmr = (self.mmr_lambda * relevance) - ((1.0 - self.mmr_lambda) * diversity)
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
        
        return selected

    def _search_ranked_hits(
        self,
        query: str,
        top_k: int = 8,
        candidate_limit: int = 500,
        session_id: str | None = None,
    ) -> list[MemorySearchHit]:
        return search_ranked_hits_impl(
            self,
            MemorySearchHit,
            query=query,
            top_k=top_k,
            candidate_limit=candidate_limit,
            session_id=session_id,
        )

    def search_conversation(self, query: str, top_k: int = 8, session_id: str | None = None) -> list[str]:
        return search_conversation_impl(
            self,
            MemorySearchHit,
            query=query,
            top_k=top_k,
            session_id=session_id,
        )

    def store_semantic_memory(self, content: str, importance: float, session_id: str | None = None) -> None:
        text = self._normalize_semantic_content(content)
        if not text:
            return
        if not self._is_valid_semantic_content(text):
            return
        importance_value = max(0.0, min(1.0, float(importance)))
        if self._is_warning_semantic(text):
            importance_value = min(importance_value, 0.70)
        session_value = str(session_id or "").strip()

        conn = self._connect()
        try:
            before_row = conn.execute(
                """
                SELECT id, importance
                FROM semantic_memory
                WHERE session_id = ? AND lower(content)=lower(?)
                LIMIT 1
                """,
                (session_value, text),
            ).fetchone()
            target_id: int | None = int(before_row[0]) if before_row else None
            if target_id is None:
                canonical_key = self._canonical_semantic_key(text)
                if canonical_key:
                    candidates = conn.execute(
                        """
                        SELECT id, content, importance
                        FROM semantic_memory
                        WHERE session_id = ?
                        ORDER BY id DESC LIMIT 200
                        """,
                        (session_value,),
                    ).fetchall()
                    for candidate in candidates:
                        candidate_id = int(candidate[0] or 0)
                        candidate_text = str(candidate[1] or "")
                        if candidate_id <= 0:
                            continue
                        if self._canonical_semantic_key(candidate_text) == canonical_key:
                            target_id = candidate_id
                            before_row = (candidate_id, float(candidate[2] or 0.0))
                            break
            conn.execute("""
                INSERT OR REPLACE INTO semantic_memory (id, session_id, content, importance, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (target_id, session_value, text, importance_value, self.now_iso()))
            conn.commit()
            row = conn.execute(
                """
                SELECT id, importance
                FROM semantic_memory
                WHERE session_id = ? AND lower(content)=lower(?)
                LIMIT 1
                """,
                (session_value, text),
            ).fetchone()
            if row:
                old_val = float(before_row[1] or 0.0) if before_row else 0.0
                new_val = float(row[1] or importance_value)
                self._log_memory_update_event(
                    {
                        "memory_id": int(row[0] or 0),
                        "old": old_val,
                        "new": new_val,
                        "confidence": new_val,
                        "delta": new_val - old_val,
                        "reason": "store_semantic_memory",
                        "memory_type": "semantic",
                        "content": text,
                        "signal_mode": self.signal_mode,
                    }
                )
        finally:
            conn.close()

    def _canonical_semantic_key(self, text: str) -> str:
        normalized = self._normalize_signal_text(self._normalize_semantic_content(text))
        if not normalized:
            return ""

        category_prefixes = self._signal_map("semantic_memory", "canonical_category_prefixes")

        for category, prefixes in category_prefixes.items():
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    value = normalized[len(prefix) :].strip()
                    if not value:
                        return ""
                    cleaned_value_chars: list[str] = []
                    for ch in value:
                        if ch.isalnum() or ch.isspace():
                            cleaned_value_chars.append(ch)
                        else:
                            cleaned_value_chars.append(" ")
                    cleaned_value = " ".join("".join(cleaned_value_chars).split())
                    if cleaned_value:
                        return f"{category}::{cleaned_value}"
                    return ""

        cleaned_chars: list[str] = []
        for ch in normalized:
            if ch.isalnum() or ch.isspace():
                cleaned_chars.append(ch)
            else:
                cleaned_chars.append(" ")
        return " ".join("".join(cleaned_chars).split())

    def _is_warning_semantic(self, text: str) -> bool:
        normalized = self._normalize_signal_text(text)
        if not normalized:
            return False
        warning_signals = self._signal_list("semantic_memory", "warning_signals")
        return any(signal in normalized for signal in warning_signals)

    def _resolve_semantic_feedback_target(
        self,
        row_id: int,
        text: str,
        *,
        conn: sqlite3.Connection | None = None,
        session_id: str | None = None,
    ) -> tuple[int, float, str] | None:
        owns_conn = conn is None
        db = conn or self._connect()
        session_value = str(session_id or "").strip()
        try:
            if int(row_id) > 0:
                row = db.execute(
                    """
                    SELECT id, importance, content
                    FROM semantic_memory
                    WHERE id = ? AND (? = '' OR session_id = ?)
                    LIMIT 1
                    """,
                    (int(row_id), session_value, session_value),
                ).fetchone()
                if row:
                    return int(row[0] or 0), float(row[1] or 0.5), str(row[2] or "")

            normalized_text = self._normalize_semantic_content(text)
            if not normalized_text:
                return None

            row = db.execute(
                """
                SELECT id, importance, content
                FROM semantic_memory
                WHERE (? = '' OR session_id = ?) AND lower(content) = lower(?)
                LIMIT 1
                """,
                (session_value, session_value, normalized_text),
            ).fetchone()
            if row:
                return int(row[0] or 0), float(row[1] or 0.5), str(row[2] or "")

            target_key = self._canonical_semantic_key(normalized_text)
            if not target_key:
                return None

            candidates = db.execute(
                """
                SELECT id, content, importance
                FROM semantic_memory
                WHERE (? = '' OR session_id = ?)
                ORDER BY id DESC LIMIT 500
                """,
                (session_value, session_value),
            ).fetchall()
            for candidate in candidates:
                candidate_id = int(candidate[0] or 0)
                if candidate_id <= 0:
                    continue
                candidate_text = str(candidate[1] or "")
                if self._canonical_semantic_key(candidate_text) == target_key:
                    return candidate_id, float(candidate[2] or 0.5), candidate_text
        finally:
            if owns_conn:
                db.close()
        return None

    @staticmethod
    def _normalize_success_signal(success: bool) -> float:
        return 1.0 if bool(success) else 0.0

    def _normalize_user_feedback(self, user_feedback: Any) -> float:
        if isinstance(user_feedback, (int, float)):
            numeric = max(0.0, min(float(user_feedback), 1.0))
            if numeric >= 0.75:
                return 1.0
            if numeric <= 0.25:
                return 0.0
            return 0.5

        return self._feedback_signal_from_text(user_feedback)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(float(minimum), min(float(maximum), float(value)))

    def _is_high_impact_memory(self, text: str) -> bool:
        normalized = self._normalize_signal_text(text)
        if not normalized:
            return False
        high_impact_signals = self._signal_list("semantic_memory", "high_impact_signals")
        return any(signal in normalized for signal in high_impact_signals)

    def _infer_contradiction_flag(
        self,
        *,
        contradiction: bool,
        success_signal: float,
        user_feedback_score: float,
        failure_reason: str | None = None,
    ) -> bool:
        if bool(contradiction):
            return True
        # High-priority contradiction: execution says success but user says wrong.
        reason_class = self._classify_failure_reason(failure_reason)
        return bool(
            (success_signal >= 1.0 and user_feedback_score == 0.0)
            or reason_class == "contradiction"
        )

    def _has_failure_reason_signal(self, failure_reason: str | None) -> bool:
        return bool(self._classify_failure_reason(failure_reason))

    def _compute_truth_score_light(
        self,
        *,
        success_signal: float,
        contradiction: bool,
        failure_reason: str | None,
        is_warning: bool,
    ) -> float:
        if contradiction:
            return 0.0

        reason_class = self._classify_failure_reason(failure_reason)
        if success_signal >= 1.0:
            return 0.45 if is_warning else 0.75

        if reason_class == "transient":
            return 0.45
        if reason_class == "logic":
            return 0.20
        if is_warning:
            return 0.30
        return 0.35

    def _compute_memory_confidence(
        self,
        *,
        success_signal: float,
        user_feedback_score: float,
        truth_score: float,
        consistency: float,
        vector_score: float | None = None,
    ) -> float:
        semantic_score = 0.5 if vector_score is None else self._clamp(float(vector_score), 0.0, 1.0)
        semantic_weight = 0.05 if vector_score is None else 0.10
        confidence = (
            (0.30 * self._clamp(float(user_feedback_score), 0.0, 1.0))
            + (0.25 * self._clamp(float(truth_score), 0.0, 1.0))
            + (0.20 * self._clamp(float(consistency), 0.0, 1.0))
            + (0.15 * self._clamp(float(success_signal), 0.0, 1.0))
            + (semantic_weight * semantic_score)
        )
        return self._clamp(confidence, 0.0, 1.0)

    def _feedback_window_metrics(
        self,
        memory_id: int,
        *,
        current_success: bool | None = None,
        window: int = 5,
    ) -> tuple[int, float]:
        wid = max(1, int(window))
        history = list(self.memory_feedback_history.get(int(memory_id), []))
        history = [bool(item) for item in history[-wid:]]
        if current_success is not None:
            history.append(bool(current_success))
            history = history[-wid:]
        if not history:
            return 0, 0.5
        success_count = sum(1 for item in history if item)
        consistency = float(success_count) / float(len(history))
        return int(success_count), float(consistency)

    def _record_feedback_outcome(self, memory_id: int, success: bool, window: int = 5) -> None:
        row_id = int(memory_id)
        if row_id <= 0:
            return
        wid = max(1, int(window))
        history = list(self.memory_feedback_history.get(row_id, []))
        history.append(bool(success))
        self.memory_feedback_history[row_id] = history[-wid:]

    def apply_memory_feedback(
        self,
        rows: list[Any],
        success: bool,
        *,
        reinforcement: float = 0.05,
        penalty: float = 0.07,
        max_rows: int = 10,
        failure_reason: str | None = None,
        user_feedback: Any = None,
        truth_score: float | None = None,
        suspicious_case: bool | None = None,
        contradiction: bool = False,
        session_id: str | None = None,
    ) -> int:
        session_value = self._normalize_session_id(session_id)
        user_feedback_score = self._normalize_user_feedback(user_feedback)
        if not rows:
            feedback_suffix = ""
            if user_feedback_score <= 0.0:
                feedback_suffix = " + feedback_negative"
            elif user_feedback_score >= 1.0:
                feedback_suffix = " + feedback_positive"
            self._log_memory_update_event(
                {
                    "memory_id": 0,
                    "old": 0.0,
                    "new": 0.0,
                    "confidence": 0.0,
                    "delta": 0.0,
                    "reason": f"no_update + empty_rows{feedback_suffix}",
                    "user_feedback": float(user_feedback_score),
                    "attempted_rows": 0,
                    "resolved_rows": 0,
                    "updated_rows": 0,
                }
            )
            return 0
        limit = max(1, min(int(max_rows), 10))
        base_success_weight = float(reinforcement) if reinforcement is not None else 0.22
        success_weight = self._clamp(base_success_weight, 0.10, 0.35)
        hard_penalty_factor = self._clamp(1.0 - (float(penalty) * 3.571428), 0.55, 0.92)
        reason = str(failure_reason or "").strip().lower()
        success_signal = self._normalize_success_signal(success)
        contradiction_detected = self._infer_contradiction_flag(
            contradiction=contradiction,
            success_signal=success_signal,
            user_feedback_score=user_feedback_score,
            failure_reason=reason,
        )
        reason_signal_hit = self._has_failure_reason_signal(failure_reason)
        updated = 0
        changed_events = 0
        attempted_rows = 0
        resolved_rows = 0
        skipped_warning_rows = 0
        unchanged_rows = 0
        first_resolved_id = 0
        seen_ids: set[int] = set()
        seen_keys: set[str] = set()
        conn = self._connect()
        try:
            cursor = conn.cursor()
            for row in rows[:limit]:
                attempted_rows += 1
                row_id = 0
                text = ""
                if isinstance(row, dict):
                    row_id = int(row.get("id") or 0)
                    text = self._normalize_semantic_content(str(row.get("content", "") or ""))
                else:
                    text = self._normalize_semantic_content(str(row or ""))

                canonical_key = self._canonical_semantic_key(text)
                if row_id <= 0 and not canonical_key:
                    continue
                if canonical_key:
                    if canonical_key in seen_keys:
                        continue
                    seen_keys.add(canonical_key)

                resolved = self._resolve_semantic_feedback_target(
                    int(row_id),
                    text,
                    conn=conn,
                    session_id=session_value or None,
                )
                if not resolved:
                    continue
                row_id, base_importance, resolved_text = resolved
                if row_id <= 0:
                    continue
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                resolved_rows += 1
                if first_resolved_id <= 0:
                    first_resolved_id = int(row_id)

                is_warning = self._is_warning_semantic(resolved_text or text)
                if bool(success_signal >= 1.0) and is_warning:
                    skipped_warning_rows += 1
                    continue

                is_high_impact = self._is_high_impact_memory(resolved_text or text)
                if suspicious_case is None:
                    row_suspicious_case = (
                        bool(success_signal < 1.0)
                        or contradiction_detected
                        or is_high_impact
                or reason_signal_hit
                    )
                else:
                    row_suspicious_case = bool(suspicious_case)

                if row_suspicious_case:
                    if truth_score is None:
                        row_truth_score = self._compute_truth_score_light(
                            success_signal=success_signal,
                            contradiction=contradiction_detected,
                            failure_reason=reason,
                            is_warning=is_warning,
                        )
                    else:
                        row_truth_score = self._clamp(float(truth_score), 0.0, 1.0)
                else:
                    row_truth_score = 0.5

                success_count_last_5, consistency = self._feedback_window_metrics(
                    row_id,
                    current_success=bool(success_signal >= 1.0),
                    window=5,
                )

                row_vector_score: float | None = None
                if isinstance(row, dict):
                    with contextlib.suppress(Exception):
                        row_vector_score = float(row.get("vector_score")) if row.get("vector_score") is not None else None
                confidence_score = self._compute_memory_confidence(
                    success_signal=success_signal,
                    user_feedback_score=user_feedback_score,
                    truth_score=row_truth_score,
                    consistency=consistency,
                    vector_score=row_vector_score,
                )

                delta = (
                    (success_weight * success_signal)
                    + (0.25 * (row_truth_score - 0.5))
                    + (0.25 * (user_feedback_score - 0.5))
                    + (0.15 * (consistency - 0.5))
                )
                delta *= (0.70 + (0.60 * confidence_score))
                delta_cap = self._clamp(0.06 + (0.20 * success_weight), 0.08, 0.12)
                delta = self._clamp(delta, -delta_cap, delta_cap)

                old_importance = float(base_importance)
                if old_importance > 0.9:
                    delta *= 0.3
                if old_importance < 0.2 and success_signal >= 1.0:
                    delta += 0.02

                new_importance = (old_importance * 0.98) + delta

                if row_truth_score < 0.3 or contradiction_detected:
                    new_importance *= hard_penalty_factor

                if success_count_last_5 >= 3 and consistency > 0.8:
                    new_importance += 0.05

                if is_warning:
                    new_importance = min(new_importance, 0.75)

                new_importance = self._clamp(new_importance, 0.05, 1.0)
                importance_changed = abs(float(new_importance) - float(old_importance)) > 1e-12
                reason_parts: list[str] = []
                reason_parts.append("success" if success_signal >= 1.0 else "failure")
                if user_feedback_score >= 1.0:
                    reason_parts.append("feedback_positive")
                elif user_feedback_score <= 0.0:
                    reason_parts.append("feedback_negative")
                if contradiction_detected:
                    reason_parts.append("contradiction")
                if row_truth_score < 0.3:
                    reason_parts.append("low_truth")
                normalized_reason = str(reason or "").strip().lower()
                skip_default_reason = normalized_reason in {"success", "ok", "done"}
                if normalized_reason and not skip_default_reason:
                    reason_parts.append(f"reason:{reason}")
                if is_warning:
                    reason_parts.append("warning_memory")
                event_reason = " + ".join(reason_parts) if reason_parts else "update"
                with contextlib.suppress(Exception):
                    cursor.execute(
                        "UPDATE semantic_memory SET importance = ? WHERE id = ?",
                        (float(new_importance), int(row_id)),
                    )
                    if cursor.rowcount > 0:
                        updated += 1
                        self._record_feedback_outcome(row_id, bool(success_signal >= 1.0), window=5)
                        if importance_changed:
                            changed_events += 1
                            self._log_memory_update_event(
                                {
                                    "memory_id": int(row_id),
                                    "old": float(old_importance),
                                    "new": float(new_importance),
                                    "confidence": float(confidence_score),
                                    "delta": float(new_importance - old_importance),
                                    "reason": event_reason,
                                    "truth_score": float(row_truth_score),
                                    "consistency": float(consistency),
                                    "success_signal": float(success_signal),
                                    "user_feedback": float(user_feedback_score),
                                }
                            )
                        else:
                            unchanged_rows += 1
            if updated > 0:
                conn.commit()
        finally:
            conn.close()
        if changed_events == 0:
            no_update_reason = "no_update + filtered"
            if resolved_rows <= 0:
                no_update_reason = "no_update + no_target"
            elif skipped_warning_rows > 0 and skipped_warning_rows >= resolved_rows:
                no_update_reason = "no_update + warning_skip"
            elif unchanged_rows > 0:
                no_update_reason = "no_update + unchanged_importance"
            self._log_memory_update_event(
                {
                    "memory_id": int(first_resolved_id or 0),
                    "old": 0.0,
                    "new": 0.0,
                    "confidence": 0.0,
                    "delta": 0.0,
                    "reason": no_update_reason,
                    "attempted_rows": int(attempted_rows),
                    "resolved_rows": int(resolved_rows),
                    "updated_rows": int(updated),
                    "changed_events": int(changed_events),
                    "skipped_warning_rows": int(skipped_warning_rows),
                }
            )
        return updated

    def reinforce_semantic_texts(
        self,
        contents: list[str],
        delta: float = 0.05,
        session_id: str | None = None,
    ) -> int:
        return self.apply_memory_feedback(
            rows=[{"content": str(item or "")} for item in (contents or [])],
            success=True,
            reinforcement=float(delta),
            max_rows=10,
            session_id=session_id,
        )

    def _semantic_ttl_days(self, text: str) -> int:
        lowered = str(text or "").strip().lower()
        if lowered.startswith(self.RECENT_SEMANTIC_PREFIX.lower()):
            return int(self.EPISODIC_TTL_DAYS)
        return 3650

    def _semantic_decay_factor(self, created_at: str, ttl_days: int) -> float:
        safe_ttl = max(1, int(ttl_days))
        created_ts = self._safe_parse_timestamp(str(created_at or ""))
        now = datetime.now(UTC)
        age_days = max(0.0, (now - created_ts).total_seconds() / 86400.0)
        factor = 1.0 - (age_days / float(safe_ttl))
        return max(float(self.SEMANTIC_DECAY_FLOOR), min(1.0, factor))

    def get_semantic_memory(
        self,
        limit: int = 5,
        reinforce: bool = True,
        session_id: str | None = None,
    ) -> list[str]:
        session_value = str(session_id or "").strip()
        conn = self._connect()
        try:
            fetch_limit = max(max(1, int(limit)) * 6, 30)
            rows = conn.execute("""
                SELECT id, content, importance, created_at FROM semantic_memory
                WHERE (? = '' OR session_id = ?) AND importance > 0.0 ORDER BY importance DESC LIMIT ?
            """, (session_value, session_value, fetch_limit)).fetchall()
        finally:
            conn.close()
        ranked_rows: list[tuple[float, tuple[Any, ...]]] = []
        for row in rows:
            text = self._normalize_semantic_content(str(row[1] or ""))
            ttl_days = self._semantic_ttl_days(text)
            decay_factor = self._semantic_decay_factor(str(row[3] or ""), ttl_days)
            base_importance = float(row[2] or 0.0)
            score = base_importance * decay_factor
            if self._is_warning_semantic(text):
                score *= 0.80
            ranked_rows.append((score, row))
        ranked_rows.sort(key=lambda item: item[0], reverse=True)

        values: list[str] = []
        seen: set[str] = set()
        selected_rows: list[dict[str, Any]] = []
        for scored_importance, row in ranked_rows:
            if len(values) >= max(1, int(limit)):
                break
            semantic_id = int(row[0] or 0)
            text = self._normalize_semantic_content(str(row[1] or ""))
            if not text or not self._is_valid_semantic_content(text):
                continue
            if float(scored_importance) < 0.15:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            values.append(text)
            selected_rows.append(
                {
                    "id": semantic_id,
                    "source": f"semantic_memory#{semantic_id}",
                    "content": text,
                    "importance": float(row[2] or 0.0),
                    "created_at": str(row[3] or ""),
                }
            )
        if bool(reinforce):
            with contextlib.suppress(Exception):
                self._reinforce_selected_rows(selected_rows)
        return values

    def prune_expired_semantic(self) -> int:
        conn = self._connect()
        deleted = 0
        now = datetime.now(UTC)
        try:
            rows = conn.execute("SELECT id, content, created_at FROM semantic_memory").fetchall()
            delete_ids: list[tuple[int]] = []
            for row in rows:
                row_id = int(row[0] or 0)
                text = self._normalize_semantic_content(str(row[1] or ""))
                if row_id <= 0:
                    continue
                ttl_days = self._semantic_ttl_days(text)
                if ttl_days >= 3650:
                    continue
                created_ts = self._safe_parse_timestamp(str(row[2] or ""))
                age_days = max(0.0, (now - created_ts).total_seconds() / 86400.0)
                if age_days > float(ttl_days * 2):
                    delete_ids.append((row_id,))
            if delete_ids:
                conn.executemany("DELETE FROM semantic_memory WHERE id = ?", delete_ids)
                conn.commit()
                deleted = len(delete_ids)
        finally:
            conn.close()
        return deleted

    def prune_soft_forgotten_semantic(self, *, max_age_days: int = 7, threshold: float = 0.05) -> int:
        safe_days = max(1, int(max_age_days))
        safe_threshold = max(0.0, min(float(threshold), 1.0))
        cutoff = datetime.now(UTC) - timedelta(days=safe_days)

        conn = self._connect()
        deleted = 0
        try:
            rows = conn.execute(
                "SELECT id, importance, created_at FROM semantic_memory"
            ).fetchall()
            delete_ids: list[tuple[int]] = []
            for row in rows:
                row_id = int(row[0] or 0)
                importance = float(row[1] or 0.0)
                created_at = str(row[2] or "")
                if row_id <= 0:
                    continue
                if importance >= safe_threshold:
                    continue
                created_ts = self._safe_parse_timestamp(created_at)
                if created_ts > cutoff:
                    continue
                delete_ids.append((row_id,))
            if delete_ids:
                conn.executemany("DELETE FROM semantic_memory WHERE id = ?", delete_ids)
                conn.commit()
                deleted = len(delete_ids)
        finally:
            conn.close()
        return deleted

    def _normalize_semantic_content(self, content: str) -> str:
        text = str(content or "").strip()
        if not text:
            return ""
        if text.startswith("- "):
            text = text[2:].strip()
        if text.startswith("* "):
            text = text[2:].strip()
        if text.startswith("\u2022 "):
            text = text[2:].strip()
        return text

    def _is_valid_semantic_content(self, content: str) -> bool:
        text = str(content or "").strip()
        if not text:
            return False
        lowered = self._normalize_signal_text(text)
        episodic_pattern_prefixes = (
            "[recent] action ",
            "recent action ",
        )
        if any(lowered.startswith(prefix) for prefix in episodic_pattern_prefixes):
            if " usually succeeds in similar contexts" in lowered:
                return True
            if " often fails in similar contexts" in lowered:
                return True
        blocked_exact = set(self._signal_list("semantic_memory", "validity_filters", "blocked_exact"))
        blocked_prefixes = self._signal_list("semantic_memory", "validity_filters", "blocked_prefixes")
        blocked_contains = self._signal_list("semantic_memory", "validity_filters", "blocked_contains")
        if lowered in blocked_exact:
            return False
        if any(lowered.startswith(prefix) for prefix in blocked_prefixes):
            return False
        if any(snippet in lowered for snippet in blocked_contains):
            return False
        if len(text) < 3:
            return False
        if len(text) > 160:
            return False
        return True

    def prune_semantic_memory_noise(self) -> int:
        conn = self._connect()
        deleted = 0
        try:
            rows = conn.execute("SELECT id, content FROM semantic_memory").fetchall()
            for row in rows:
                row_id = int(row[0])
                text = self._normalize_semantic_content(str(row[1] or ""))
                if self._is_valid_semantic_content(text):
                    continue
                conn.execute("DELETE FROM semantic_memory WHERE id = ?", (row_id,))
                deleted += 1
            if deleted:
                conn.commit()
        finally:
            conn.close()
        return deleted

    def prune_conversation_noise(self) -> int:
        conn = self._connect()
        deleted = 0
        try:
            rows = conn.execute("SELECT id, content FROM conversation_messages").fetchall()
            for row in rows:
                row_id = int(row[0])
                content = str(row[1] or "").strip()
                if not self._is_noisy_memory_body(content):
                    continue
                conn.execute("DELETE FROM conversation_messages WHERE id = ?", (row_id,))
                deleted += 1
            if deleted:
                conn.commit()
                self.memory_query_cache.clear()
        finally:
            conn.close()
        return deleted

    def prune_profile_facts_noise(self) -> int:
        conn = self._connect()
        deleted = 0
        try:
            rows = conn.execute("SELECT id, fact FROM profile_facts").fetchall()
            for row in rows:
                row_id = int(row[0])
                fact = str(row[1] or "").strip()
                if not fact:
                    conn.execute("DELETE FROM profile_facts WHERE id = ?", (row_id,))
                    deleted += 1
                    continue
                if ":" not in fact:
                    conn.execute("DELETE FROM profile_facts WHERE id = ?", (row_id,))
                    deleted += 1
                    continue
                if not self._is_valid_semantic_content(fact):
                    conn.execute("DELETE FROM profile_facts WHERE id = ?", (row_id,))
                    deleted += 1
                    continue
                if not self._is_valid_profile_fact(fact):
                    conn.execute("DELETE FROM profile_facts WHERE id = ?", (row_id,))
                    deleted += 1
                    continue
            if deleted:
                conn.commit()
        finally:
            conn.close()
        return deleted

    def save_task_state(self, goal: str, step: str, session_id: str | None = None) -> None:
        session_value = self._normalize_session_id(session_id)
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT goal, current_step FROM task_state WHERE session_id = ? LIMIT 1",
                (session_value,),
            ).fetchone()
            goal_text = str(goal or "").strip()
            step_text = str(step or "").strip()
            if existing:
                if not goal_text:
                    goal_text = str(existing[0] or "").strip()
                if not step_text:
                    step_text = str(existing[1] or "").strip()
            cursor = conn.execute(
                """
                UPDATE task_state
                SET goal = ?, current_step = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (goal_text, step_text, self.now_iso(), session_value),
            )
            if int(cursor.rowcount or 0) <= 0:
                conn.execute(
                    """
                    INSERT INTO task_state (session_id, goal, current_step, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_value, goal_text, step_text, self.now_iso()),
                )
            conn.commit()
            self._log_memory_update_event(
                {
                    "memory_id": 0,
                    "old": 0.0,
                    "new": 1.0,
                    "confidence": 1.0,
                    "delta": 1.0,
                    "reason": "save_task_state",
                    "memory_type": "task",
                    "goal": goal_text,
                    "step": step_text,
                    "session_id": session_value,
                    "signal_mode": self.signal_mode,
                }
            )
        finally:
            conn.close()

    def get_task_state(self, session_id: str | None = None) -> dict[str, str] | None:
        session_value = str(session_id or "").strip()
        conn = self._connect()
        try:
            if session_value:
                row = conn.execute(
                    "SELECT goal, current_step FROM task_state WHERE session_id = ? LIMIT 1",
                    (session_value,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT goal, current_step FROM task_state ORDER BY updated_at DESC, id DESC LIMIT 1"
                ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return {"goal": str(row[0] or ""), "current_step": str(row[1] or "")}

    def store_profile_fact(
        self,
        fact: str,
        importance: float = 0.9,
        session_id: str | None = None,
    ) -> None:
        self.add_profile_fact(fact=fact, importance=importance, session_id=session_id)

    def _profile_fact_key(self, fact: str) -> str:
        text = str(fact or "").strip()
        if not text:
            return ""
        return text.split(":", 1)[0].strip().lower()

    def delete_profile_fact_by_key(self, key: str, session_id: str | None = None) -> None:
        target_key = str(key or "").strip().lower()
        if not target_key:
            return
        session_value = str(session_id or "").strip()
        conn = self._connect()
        try:
            if session_value:
                rows = conn.execute(
                    "SELECT id, fact FROM profile_facts WHERE session_id = ?",
                    (session_value,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT id, fact FROM profile_facts").fetchall()
            delete_ids: list[int] = []
            for row in rows:
                row_id = int(row[0])
                fact_text = str(row[1] or "").strip()
                if not fact_text:
                    continue
                if self._profile_fact_key(fact_text) == target_key:
                    delete_ids.append(row_id)
            if delete_ids:
                conn.executemany(
                    "DELETE FROM profile_facts WHERE id = ?",
                    [(row_id,) for row_id in delete_ids],
                )
                conn.commit()
        finally:
            conn.close()

    def _insert_profile_fact(
        self,
        fact: str,
        importance: float = 0.9,
        session_id: str | None = None,
    ) -> None:
        text = str(fact or "").strip()
        if not text:
            return
        importance_value = max(0.0, min(1.0, float(importance)))
        session_value = self._normalize_session_id(session_id)
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO profile_facts (session_id, fact, importance, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_value, text, importance_value, self.now_iso()),
            )
            conn.commit()
            row_id = int(
                conn.execute(
                    "SELECT id FROM profile_facts WHERE session_id = ? AND fact = ? LIMIT 1",
                    (session_value, text),
                ).fetchone()[0]
            )
            self._log_memory_update_event(
                {
                    "memory_id": row_id,
                    "old": 0.0,
                    "new": importance_value,
                    "confidence": importance_value,
                    "delta": importance_value,
                    "reason": "store_profile_fact",
                    "memory_type": "profile",
                    "fact": text,
                    "session_id": session_value,
                    "signal_mode": self.signal_mode,
                }
            )
        finally:
            conn.close()

    def add_profile_fact(
        self,
        fact: str,
        importance: float = 0.9,
        session_id: str | None = None,
    ) -> None:
        text = str(fact or "").strip()
        if not text:
            return
        if not self._is_valid_profile_fact(text):
            return
        key = self._profile_fact_key(text)
        if not key:
            return
        self.delete_profile_fact_by_key(key, session_id=session_id)
        self._insert_profile_fact(text, importance=importance, session_id=session_id)

    def _is_valid_profile_fact(self, fact: str) -> bool:
        text = str(fact or "").strip()
        if not text or ":" not in text:
            return False
        key, value = text.split(":", 1)
        fact_key = str(key or "").strip().lower()
        fact_value = str(value or "").strip()
        if not fact_key or not fact_value:
            return False
        if "?" in fact_value:
            return False
        lowered_value = fact_value.lower().strip(" .,!?:;\"'")
        if not lowered_value:
            return False
        tokens = [token for token in fact_value.replace("-", " ").replace("'", " ").split() if token]
        lowered_tokens = [token.strip().lower() for token in tokens]
        if self._contains_question_hint(lowered_value, lowered_tokens):
            return False
        if self._contains_noise_name_phrase(lowered_value):
            return False
        if not tokens:
            return False
        if fact_key == "user_name":
            if any(ch in fact_value for ch in ("/", "\\", "|", "<", ">", "{", "}", "[", "]")):
                return False
            if not fact_value[0].isalnum():
                return False
            if len(tokens) > 4:
                return False
            for token in tokens:
                if not token or not token[0].isalnum():
                    return False
                if any(ch.isdigit() for ch in token):
                    return False
        return True

    def _contains_noise_name_phrase(self, lowered_value: str) -> bool:
        value = self._normalize_signal_text(lowered_value)
        if not value:
            return True
        blocked_values = set(self._signal_list("semantic_memory", "noise_name_filters", "blocked_exact"))
        if value in blocked_values:
            return True
        blocked_contains = self._signal_list("semantic_memory", "noise_name_filters", "blocked_contains")
        if any(noise in value for noise in blocked_contains):
            return True
        tokens = [t for t in value.replace("-", " ").replace("'", " ").split() if t]
        if not tokens:
            return True
        if len(tokens) <= 2 and any(token in blocked_values for token in tokens):
            return True
        phrases: set[str] = set(tokens)
        for n in (2, 3, 4):
            if len(tokens) < n:
                continue
            for idx in range(0, len(tokens) - n + 1):
                phrases.add(" ".join(tokens[idx : idx + n]))
        return any(phrase in blocked_values for phrase in phrases)

    def get_profile_facts(self, top_k: int = 3, session_id: str | None = None) -> list[str]:
        session_value = str(session_id or "").strip()
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT fact
                FROM profile_facts
                WHERE (? = '' OR session_id = ?)
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                (session_value, session_value, max(1, int(top_k))),
            ).fetchall()
        finally:
            conn.close()
        return [str(row[0]).strip() for row in rows if str(row[0] or "").strip()]

    def get_task_state_lines(self, session_id: str | None = None) -> list[str]:
        state = self.get_task_state(session_id=session_id)
        if not state:
            return []
        goal = str(state.get("goal", "") or "").strip()
        step = str(state.get("current_step", "") or "").strip()
        lines: list[str] = []
        if goal:
            lines.append(f"[TASK] Goal: {goal}")
        if step:
            lines.append(f"[TASK] Current step: {step}")
        return lines

    def get_recent_conversation_lines(self, session_id: str = "", limit: int = 8) -> list[str]:
        conn = self._connect()
        try:
            if str(session_id or "").strip():
                rows = conn.execute(
                    """
                    SELECT id, timestamp, role, content
                    FROM conversation_messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (str(session_id), max(1, int(limit))),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, timestamp, role, content
                    FROM conversation_messages
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (max(1, int(limit)),),
                ).fetchall()
        finally:
            conn.close()

        lines: list[str] = []
        for row in rows:
            stored_role = str(row[2] or "").strip().lower()
            parsed_role, body = self._extract_role_and_body(str(row[3] or ""))
            if not body:
                continue
            role = stored_role if stored_role in {"user", "assistant"} else parsed_role
            role_label = "USER" if role == "user" else "ASSISTANT"
            lines.append(f"[RECENT] {role_label}: {body}")
        lines.reverse()
        return lines

    def build_auto_context(
        self,
        query: str,
        session_id: str = "",
        top_k: int = 6,
    ) -> list[str]:
        return build_auto_context_impl(self, query=query, session_id=session_id, top_k=top_k)

    def _quick_score(self, text: str, query: str, order: int) -> float:
        return quick_score_impl(self, text=text, query=query, order=order)

    # Compatibility helpers for BaseAgent
    def configure_llm_extraction(self, enabled: bool, model: str | None = None) -> None:
        _ = enabled
        _ = model

    def add_scheduled_task(
        self,
        task_text: str,
        schedule_time: str,
        scheduled_for: str,
        schedule_type: str = "once",
        cron_expression: str = "",
        session_id: str | None = None,
        owner_agent_id: str | None = None,
    ) -> int:
        clean_task = str(task_text or "").strip()
        clean_schedule = str(schedule_time or "").strip()
        clean_due = str(scheduled_for or "").strip()
        clean_type = str(schedule_type or "once").strip().lower() or "once"
        clean_cron = str(cron_expression or "").strip()
        if not clean_task or not clean_schedule or not clean_due:
            return 0
        session_value = self._normalize_session_id(session_id)
        owner_value = str(owner_agent_id or "").strip()
        now_iso = self.now_iso()

        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                INSERT INTO scheduled_tasks (
                    session_id,
                    owner_agent_id,
                    task_text,
                    schedule_time,
                    schedule_type,
                    cron_expression,
                    next_run,
                    last_run,
                    status,
                    worker_name,
                    worker_id,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, '', 'waiting', '', '', ?, ?)
                """,
                (
                    session_value,
                    owner_value,
                    clean_task,
                    clean_schedule,
                    clean_type,
                    clean_cron,
                    clean_due,
                    now_iso,
                    now_iso,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)
        finally:
            conn.close()

    def delete_all_scheduled_tasks(
        self,
        session_id: str | None = None,
        owner_agent_id: str | None = None,
    ) -> int:
        session_value = str(session_id or "").strip()
        owner_value = str(owner_agent_id or "").strip()
        conn = self._connect()
        try:
            if session_value:
                deleted = int(
                    conn.execute(
                        """
                        SELECT COUNT(*) FROM scheduled_tasks
                        WHERE session_id = ? AND (? = '' OR owner_agent_id = ?)
                        """,
                        (session_value, owner_value, owner_value),
                    ).fetchone()[0]
                    or 0
                )
                conn.execute(
                    """
                    DELETE FROM scheduled_tasks
                    WHERE session_id = ? AND (? = '' OR owner_agent_id = ?)
                    """,
                    (session_value, owner_value, owner_value),
                )
            else:
                deleted = int(conn.execute("SELECT COUNT(*) FROM scheduled_tasks").fetchone()[0] or 0)
                conn.execute("DELETE FROM scheduled_tasks")
            conn.commit()
            return deleted
        finally:
            conn.close()

    def get_scheduled_tasks(
        self,
        limit: int = 5,
        session_id: str | None = None,
        owner_agent_id: str | None = None,
    ) -> list[Any]:
        safe_limit = max(1, min(int(limit), 100))
        session_value = str(session_id or "").strip()
        owner_value = str(owner_agent_id or "").strip()
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, task_text, schedule_time, next_run, status, created_at,
                       COALESCE(schedule_type, 'once'), COALESCE(cron_expression, ''),
                       COALESCE(last_run, ''), COALESCE(worker_name, ''), COALESCE(worker_id, ''),
                       COALESCE(session_id, ''), COALESCE(owner_agent_id, '')
                FROM scheduled_tasks
                WHERE (? = '' OR session_id = ?)
                  AND (? = '' OR owner_agent_id = ?)
                ORDER BY next_run ASC, id ASC
                LIMIT ?
                """,
                (session_value, session_value, owner_value, owner_value, safe_limit),
            ).fetchall()
        finally:
            conn.close()

        return [
            ScheduledTask(
                id=int(row[0] or 0),
                task_text=str(row[1] or ""),
                schedule_time=str(row[2] or ""),
                scheduled_for=str(row[3] or ""),
                status=str(row[4] or "waiting"),
                created_at=str(row[5] or ""),
                schedule_type=str(row[6] or "once"),
                cron_expression=str(row[7] or ""),
                dispatched_at=str(row[8] or ""),
                worker_name=str(row[9] or ""),
                worker_id=str(row[10] or ""),
                session_id=str(row[11] or ""),
                owner_agent_id=str(row[12] or ""),
            )
            for row in rows
        ]

    def get_scheduled_task_by_id(
        self,
        task_id: int,
        session_id: str | None = None,
        owner_agent_id: str | None = None,
    ) -> Any | None:
        row_id = max(0, int(task_id))
        if row_id <= 0:
            return None
        session_value = str(session_id or "").strip()
        owner_value = str(owner_agent_id or "").strip()
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT id, task_text, schedule_time, next_run, status, created_at,
                       COALESCE(schedule_type, 'once'), COALESCE(cron_expression, ''),
                       COALESCE(last_run, ''), COALESCE(worker_name, ''), COALESCE(worker_id, ''),
                       COALESCE(session_id, ''), COALESCE(owner_agent_id, '')
                FROM scheduled_tasks
                WHERE id = ?
                  AND (? = '' OR session_id = ?)
                  AND (? = '' OR owner_agent_id = ?)
                LIMIT 1
                """,
                (row_id, session_value, session_value, owner_value, owner_value),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return ScheduledTask(
            id=int(row[0] or 0),
            task_text=str(row[1] or ""),
            schedule_time=str(row[2] or ""),
            scheduled_for=str(row[3] or ""),
            status=str(row[4] or "waiting"),
            created_at=str(row[5] or ""),
            schedule_type=str(row[6] or "once"),
            cron_expression=str(row[7] or ""),
            dispatched_at=str(row[8] or ""),
            worker_name=str(row[9] or ""),
            worker_id=str(row[10] or ""),
            session_id=str(row[11] or ""),
            owner_agent_id=str(row[12] or ""),
        )

    def delete_scheduled_task_by_id(
        self,
        task_id: int,
        session_id: str | None = None,
        owner_agent_id: str | None = None,
    ) -> int:
        row_id = max(0, int(task_id))
        if row_id <= 0:
            return 0
        session_value = str(session_id or "").strip()
        owner_value = str(owner_agent_id or "").strip()
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                DELETE FROM scheduled_tasks
                WHERE id = ?
                  AND (? = '' OR session_id = ?)
                  AND (? = '' OR owner_agent_id = ?)
                """,
                (row_id, session_value, session_value, owner_value, owner_value),
            )
            conn.commit()
            return int(cursor.rowcount or 0)
        finally:
            conn.close()

    def get_due_tasks(
        self,
        now_iso: str | None = None,
        limit: int = 5,
        session_id: str | None = None,
        owner_agent_id: str | None = None,
    ) -> list[Any]:
        due_now = str(now_iso or self.now_iso()).strip() or self.now_iso()
        safe_limit = max(1, min(int(limit), 100))
        session_value = str(session_id or "").strip()
        owner_value = str(owner_agent_id or "").strip()
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, task_text, schedule_time, next_run, status, created_at,
                       COALESCE(schedule_type, 'once'), COALESCE(cron_expression, ''),
                       COALESCE(last_run, ''), COALESCE(worker_name, ''), COALESCE(worker_id, ''),
                       COALESCE(session_id, ''), COALESCE(owner_agent_id, '')
                FROM scheduled_tasks
                WHERE LOWER(status) = 'waiting' AND next_run != '' AND next_run <= ?
                  AND (? = '' OR session_id = ?)
                  AND (? = '' OR owner_agent_id = ?)
                ORDER BY next_run ASC, id ASC
                LIMIT ?
                """,
                (due_now, session_value, session_value, owner_value, owner_value, safe_limit),
            ).fetchall()
        finally:
            conn.close()

        return [
            ScheduledTask(
                id=int(row[0] or 0),
                task_text=str(row[1] or ""),
                schedule_time=str(row[2] or ""),
                scheduled_for=str(row[3] or ""),
                status=str(row[4] or "waiting"),
                created_at=str(row[5] or ""),
                schedule_type=str(row[6] or "once"),
                cron_expression=str(row[7] or ""),
                dispatched_at=str(row[8] or ""),
                worker_name=str(row[9] or ""),
                worker_id=str(row[10] or ""),
                session_id=str(row[11] or ""),
                owner_agent_id=str(row[12] or ""),
            )
            for row in rows
        ]

    def mark_task_running(
        self,
        task_id: int,
        worker_name: str | None = None,
        worker_id: str | None = None,
        session_id: str | None = None,
        owner_agent_id: str | None = None,
    ) -> bool:
        row_id = max(0, int(task_id))
        if row_id <= 0:
            return False
        session_value = str(session_id or "").strip()
        owner_value = str(owner_agent_id or "").strip()
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                UPDATE scheduled_tasks
                SET status = 'running',
                    worker_name = ?,
                    worker_id = ?,
                    updated_at = ?
                WHERE id = ?
                  AND (? = '' OR session_id = ?)
                  AND (? = '' OR owner_agent_id = ?)
                """,
                (
                    str(worker_name or "").strip(),
                    str(worker_id or "").strip(),
                    self.now_iso(),
                    row_id,
                    session_value,
                    session_value,
                    owner_value,
                    owner_value,
                ),
            )
            conn.commit()
            return int(cursor.rowcount or 0) > 0
        finally:
            conn.close()

    def claim_due_task(
        self,
        task_id: int,
        *,
        now_iso: str | None = None,
        worker_name: str | None = None,
        worker_id: str | None = None,
        session_id: str | None = None,
        owner_agent_id: str | None = None,
    ) -> bool:
        row_id = max(0, int(task_id))
        if row_id <= 0:
            return False
        due_now = str(now_iso or self.now_iso()).strip() or self.now_iso()
        session_value = str(session_id or "").strip()
        owner_value = str(owner_agent_id or "").strip()
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                UPDATE scheduled_tasks
                SET status = 'running',
                    worker_name = ?,
                    worker_id = ?,
                    updated_at = ?
                WHERE id = ?
                  AND LOWER(status) = 'waiting'
                  AND next_run != ''
                  AND next_run <= ?
                  AND (? = '' OR session_id = ?)
                  AND (? = '' OR owner_agent_id = ?)
                """,
                (
                    str(worker_name or "").strip(),
                    str(worker_id or "").strip(),
                    self.now_iso(),
                    row_id,
                    due_now,
                    session_value,
                    session_value,
                    owner_value,
                    owner_value,
                ),
            )
            conn.commit()
            return int(cursor.rowcount or 0) > 0
        finally:
            conn.close()

    def release_task_claim(
        self,
        task_id: int,
        *,
        session_id: str | None = None,
        owner_agent_id: str | None = None,
    ) -> bool:
        row_id = max(0, int(task_id))
        if row_id <= 0:
            return False
        session_value = str(session_id or "").strip()
        owner_value = str(owner_agent_id or "").strip()
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                UPDATE scheduled_tasks
                SET status = 'waiting',
                    worker_name = '',
                    worker_id = '',
                    updated_at = ?
                WHERE id = ?
                  AND LOWER(status) = 'running'
                  AND (? = '' OR session_id = ?)
                  AND (? = '' OR owner_agent_id = ?)
                """,
                (self.now_iso(), row_id, session_value, session_value, owner_value, owner_value),
            )
            conn.commit()
            return int(cursor.rowcount or 0) > 0
        finally:
            conn.close()

    def mark_task_completed(
        self,
        task_id: int,
        next_run: str | None = None,
        session_id: str | None = None,
        owner_agent_id: str | None = None,
    ) -> None:
        row_id = max(0, int(task_id))
        if row_id <= 0:
            return
        now_iso = self.now_iso()
        next_due = str(next_run or "").strip()
        session_value = str(session_id or "").strip()
        owner_value = str(owner_agent_id or "").strip()
        conn = self._connect()
        try:
            if next_due:
                conn.execute(
                    """
                    UPDATE scheduled_tasks
                    SET status = 'waiting',
                        next_run = ?,
                        last_run = ?,
                        worker_name = '',
                        worker_id = '',
                        updated_at = ?
                    WHERE id = ?
                      AND (? = '' OR session_id = ?)
                      AND (? = '' OR owner_agent_id = ?)
                    """,
                    (next_due, now_iso, now_iso, row_id, session_value, session_value, owner_value, owner_value),
                )
            else:
                conn.execute(
                    """
                    UPDATE scheduled_tasks
                    SET status = 'completed',
                        last_run = ?,
                        worker_name = '',
                        worker_id = '',
                        updated_at = ?
                    WHERE id = ?
                      AND (? = '' OR session_id = ?)
                      AND (? = '' OR owner_agent_id = ?)
                    """,
                    (now_iso, now_iso, row_id, session_value, session_value, owner_value, owner_value),
                )
            conn.commit()
        finally:
            conn.close()

    # Compatibility helpers for MemoryIndexManager
    def clear_query_cache(self) -> None:
        self.memory_query_cache.clear()

    def sync_indices(self, mode: str = "manual", limit: int = 300) -> int:
        safe_limit = max(1, int(limit))
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(last_indexed_message_id, 0) FROM memory_index_state WHERE id = 1"
            ).fetchone()
            last_indexed = int(row[0] or 0) if row else 0
            rows = conn.execute(
                """
                SELECT id, timestamp, role, content
                FROM conversation_messages
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (last_indexed, safe_limit),
            ).fetchall()
            if not rows:
                conn.execute(
                    "UPDATE memory_index_state SET last_sync_at = ?, last_sync_mode = ? WHERE id = 1",
                    (self.now_iso(), str(mode or "manual")),
                )
                conn.commit()
                self._last_sync_run = datetime.now(UTC)
                return 0

            self._load_embeddings_for_rows(rows)
            latest_id = max(int(r[0]) for r in rows)
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_index_state (id, last_indexed_message_id, last_sync_at, last_sync_mode)
                VALUES (1, ?, ?, ?)
                """,
                (latest_id, self.now_iso(), str(mode or "manual")),
            )
            conn.commit()
            self._last_sync_run = datetime.now(UTC)
            return len(rows)
        finally:
            conn.close()

    def maybe_sync(self, trigger: str = "interval", limit: int = 250) -> int:
        mode = str(self.sync_mode or "off").lower()
        if mode == "off":
            return 0
        if mode == "interval":
            elapsed = (datetime.now(UTC) - self._last_sync_run).total_seconds()
            if elapsed < float(self.sync_interval_sec):
                return 0
        return self.sync_indices(mode=str(trigger or "interval"), limit=limit)

    def reindex_embeddings(self, limit: int = 5000) -> int:
        safe_limit = max(1, int(limit))
        conn = self._connect()
        try:
            conn.execute("DELETE FROM memory_embedding_cache")
            conn.execute("DELETE FROM memory_index")
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_index_state (id, last_indexed_message_id, last_sync_at, last_sync_mode)
                VALUES (1, 0, ?, ?)
                """,
                (self.now_iso(), "reindex_reset"),
            )
            conn.commit()
        finally:
            conn.close()
        self.memory_embedding_cache.clear()

        total = 0
        while total < safe_limit:
            indexed = self.sync_indices(mode="reindex", limit=min(500, safe_limit - total))
            if indexed <= 0:
                break
            total += indexed
        return total

    def memory_health(self) -> dict[str, int | float | str | bool]:
        conn = self._connect()
        try:
            total_messages = int(conn.execute("SELECT COUNT(*) FROM conversation_messages").fetchone()[0])
            total_embeddings = int(
                conn.execute(
                    "SELECT COUNT(*) FROM memory_index WHERE embedding IS NOT NULL AND embedding != ''"
                ).fetchone()[0]
            )
            total_query_cache = int(len(self.memory_query_cache))
            total_profile_facts = int(conn.execute("SELECT COUNT(*) FROM profile_facts").fetchone()[0])
            state_row = conn.execute(
                "SELECT last_indexed_message_id, last_sync_at, last_sync_mode FROM memory_index_state WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()
        last_indexed = int(state_row[0] or 0) if state_row else 0
        pending = max(0, total_messages - last_indexed)
        coverage = (float(total_embeddings) / float(total_messages)) if total_messages > 0 else 0.0
        return {
            "hybrid_enabled": bool(self.hybrid_enabled),
            "mmr_enabled": bool(self.mmr_enabled),
            "temporal_decay_enabled": bool(self.temporal_decay_enabled),
            "vector_available": bool(self._vector_available),
            "embed_provider": str(self.embed_provider),
            "embed_provider_last_used": str(self._last_embed_provider_used),
            "embed_provider_last_error": str(self._last_embed_error),
            "sync_mode": str(self.sync_mode),
            "signal_mode": str(self.signal_mode),
            "prompt_memory_trigger_enabled": bool(self.prompt_memory_trigger_enabled),
            "emotion_tracking_enabled": bool(self.emotion_tracking_enabled),
            "overflow_policy": str(self.overflow_policy),
            "total_messages": total_messages,
            "total_embeddings": total_embeddings,
            "embedding_coverage": round(min(1.0, coverage), 4),
            "query_cache_entries": total_query_cache,
            "profile_facts": total_profile_facts,
            "pending_index_messages": pending,
            "last_indexed_message_id": last_indexed,
            "last_sync_at": str(state_row[1] or "") if state_row else "",
            "last_sync_mode": str(state_row[2] or "") if state_row else "",
        }

    def memory_signal_snapshot(self) -> dict[str, Any]:
        latest_event: dict[str, Any] = {}
        tail_events = self._read_jsonl_tail(self.memory_events_jsonl_path, limit=1)
        if tail_events:
            latest_event = dict(tail_events[-1])

        latest_semantic: dict[str, Any] = {}
        latest_profile: dict[str, Any] = {}
        latest_task: dict[str, Any] = {}
        conn = self._connect()
        try:
            semantic_row = conn.execute(
                "SELECT id, content, importance, created_at FROM semantic_memory ORDER BY id DESC LIMIT 1"
            ).fetchone()
            profile_row = conn.execute(
                "SELECT id, fact, importance, updated_at FROM profile_facts ORDER BY updated_at DESC, id DESC LIMIT 1"
            ).fetchone()
            task_row = conn.execute(
                "SELECT id, goal, current_step, updated_at FROM task_state ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if semantic_row:
            latest_semantic = {
                "id": int(semantic_row[0] or 0),
                "content": str(semantic_row[1] or ""),
                "importance": float(semantic_row[2] or 0.0),
                "created_at": str(semantic_row[3] or ""),
            }
        if profile_row:
            latest_profile = {
                "id": int(profile_row[0] or 0),
                "fact": str(profile_row[1] or ""),
                "importance": float(profile_row[2] or 0.0),
                "updated_at": str(profile_row[3] or ""),
            }
        if task_row:
            latest_task = {
                "id": int(task_row[0] or 0),
                "goal": str(task_row[1] or ""),
                "step": str(task_row[2] or ""),
                "updated_at": str(task_row[3] or ""),
            }

        def _event_is_present(event: dict[str, Any]) -> bool:
            event_type = str(event.get("memory_type", "") or "").strip().lower()
            event_id = int(event.get("memory_id", 0) or 0)
            if event_type == "semantic":
                return event_id > 0 and int(latest_semantic.get("id", 0) or 0) == event_id
            if event_type == "profile":
                return event_id > 0 and int(latest_profile.get("id", 0) or 0) == event_id
            if event_type == "task":
                return bool(latest_task)
            return bool(event)

        latest_db_event: dict[str, Any] = {}
        db_candidates: list[dict[str, Any]] = []
        if latest_semantic:
            db_candidates.append(
                {
                    "ts": str(latest_semantic.get("created_at", "") or ""),
                    "memory_type": "semantic",
                    "memory_id": int(latest_semantic.get("id", 0) or 0),
                    "new": float(latest_semantic.get("importance", 0.0) or 0.0),
                    "confidence": float(latest_semantic.get("importance", 0.0) or 0.0),
                    "delta": 0.0,
                    "reason": "store_semantic_memory",
                }
            )
        if latest_profile:
            db_candidates.append(
                {
                    "ts": str(latest_profile.get("updated_at", "") or ""),
                    "memory_type": "profile",
                    "memory_id": int(latest_profile.get("id", 0) or 0),
                    "new": float(latest_profile.get("importance", 0.0) or 0.0),
                    "confidence": float(latest_profile.get("importance", 0.0) or 0.0),
                    "delta": 0.0,
                    "reason": "store_profile_fact",
                }
            )
        if latest_task:
            db_candidates.append(
                {
                    "ts": str(latest_task.get("updated_at", "") or ""),
                    "memory_type": "task",
                    "memory_id": int(latest_task.get("id", 0) or 0),
                    "new": 1.0,
                    "confidence": 1.0,
                    "delta": 0.0,
                    "reason": "save_task_state",
                }
            )
        if db_candidates:
            latest_db_event = max(
                db_candidates,
                key=lambda item: self._safe_parse_timestamp(str(item.get("ts", "") or "")).timestamp(),
            )

        event_ts = self._safe_parse_timestamp(str(latest_event.get("ts", "") or "")).timestamp() if latest_event else 0.0
        db_ts = self._safe_parse_timestamp(str(latest_db_event.get("ts", "") or "")).timestamp() if latest_db_event else 0.0
        if latest_db_event and (not latest_event or not _event_is_present(latest_event) or db_ts > event_ts):
            latest_event = dict(latest_db_event)

        return {
            "signal_mode": str(self.signal_mode),
            "latest_event": latest_event,
            "latest_semantic": latest_semantic,
            "latest_profile": latest_profile,
            "latest_task": latest_task,
            "latest_emotion": self.get_latest_emotion_signal(role="user") or {},
            "health": self.memory_health(),
        }

    def _to_structured_hit(
        self,
        hit: MemorySearchHit,
        citation_index: int,
    ) -> dict[str, str | float | int]:
        parsed_ts = self._safe_parse_timestamp(hit.timestamp)
        relative = get_relative_time_label(parsed_ts)
        exact_date = parsed_ts.strftime("%d/%m/%Y")
        source_path = "memory.db"
        source_line = int(hit.id)
        source = f"{source_path}#msg-{source_line}"
        return {
            "citation": f"C{citation_index}",
            "id": int(hit.id),
            "timestamp": str(hit.timestamp or ""),
            "exact_date": exact_date,
            "relative_time": relative,
            "role": str(hit.role or "").strip().lower(),
            "content": str(hit.content or ""),
            "body": str(hit.body or ""),
            "source": source,
            "source_path": source_path,
            "source_line": source_line,
            "score": float(hit.final_score),
            "bm25_score": float(hit.fts_score),
            "vector_score": float(hit.vector_score),
            "lexical_score": float(hit.lexical_score),
            "recency_score": float(hit.recency_score),
            "temporal_score": float(hit.temporal_score),
        }

    def search_conversation_structured(
        self,
        query: str,
        top_k: int = 8,
        candidate_limit: int = 500,
        session_id: str | None = None,
    ) -> list[dict[str, str | float | int]]:
        return search_conversation_structured_impl(
            self,
            MemorySearchHit,
            query=query,
            top_k=top_k,
            candidate_limit=candidate_limit,
            session_id=session_id,
        )

