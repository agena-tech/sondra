from datetime import UTC, datetime

from sondra.memory.persistent_memory import PersistentMemoryStore


def _add_message(store: PersistentMemoryStore, role: str, text: str, session_id: str = "s1") -> None:
    store.add_conversation_messages(
        [
            (
                datetime.now(UTC).isoformat(),
                role,
                text,
                session_id,
            )
        ]
    )


def test_search_conversation_structured_includes_citation_and_source(tmp_path) -> None:
    store = PersistentMemoryStore(base_dir=str(tmp_path))
    _add_message(store, "user", "Benim adim Anezatra")
    _add_message(store, "assistant", "Merhaba Anezatra")

    rows = store.search_conversation_structured("adim", top_k=5, candidate_limit=100)
    assert rows
    first = rows[0]
    assert first["citation"] == "C1"
    assert str(first["source"]).startswith("memory.db#msg-")
    assert int(first["id"]) > 0


def test_get_conversation_message_by_id_returns_row_metadata(tmp_path) -> None:
    store = PersistentMemoryStore(base_dir=str(tmp_path))
    _add_message(store, "user", "Hobim satranctir")

    latest = store.recent_conversation(limit=1)[0]
    row = store.get_conversation_message_by_id(int(latest.id))

    assert row is not None
    assert int(row["id"]) == int(latest.id)
    assert str(row["role"]) in {"user", "assistant"}
    assert str(row["source"]).startswith("memory.db#msg-")
    assert int(row["source_line"]) == int(latest.id)


def test_structured_search_uses_query_cache(tmp_path) -> None:
    store = PersistentMemoryStore(base_dir=str(tmp_path))
    _add_message(store, "user", "JWT token kullaniliyor")
    _add_message(store, "assistant", "Tamam, JWT not edildi")

    rows1 = store.search_conversation_structured("jwt token", top_k=3, candidate_limit=150)
    cache_size_before = len(store.memory_query_cache)
    rows2 = store.search_conversation_structured("jwt token", top_k=3, candidate_limit=150)
    cache_size_after = len(store.memory_query_cache)

    assert rows1 == rows2
    assert cache_size_before > 0
    assert cache_size_after == cache_size_before
