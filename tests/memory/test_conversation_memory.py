from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from sondra.agents.base_agent import BaseAgent
from sondra.agents.state import AgentState
from sondra.memory.conversation_memory import memory_search
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


def test_memory_search_prefers_recent_user_correction_for_hobby(tmp_path) -> None:
    store = PersistentMemoryStore(base_dir=str(tmp_path))
    _add_message(store, "user", "Hobim satranç.")
    store.store_semantic_memory("User's hobby is satranç", importance=0.9)
    _add_message(store, "user", "Hayır, hobim satranç değil, piyano.")

    agent = SimpleNamespace(
        memory_store=store,
        state=AgentState(agent_name="Root Agent", parent_id=None),
        llm_config=SimpleNamespace(scan_mode="general"),
        _last_memory_hits=[],
    )
    agent.state.update_context("conversation_session_id", "s1")
    agent._is_general_root_agent = lambda: True
    agent._memory_health_snapshot = lambda: {"total_messages": 2}
    agent._latest_user_message_raw = lambda: "Hobim neydi?"

    rows = memory_search(agent, "Hobim neydi?")

    assert rows
    assert any("piyano" in line.lower() for line in rows[:2])


def test_last_emotion_snapshot_roundtrip_is_boot_only(tmp_path) -> None:
    def build_agent() -> BaseAgent:
        agent = BaseAgent.__new__(BaseAgent)
        agent.llm_config = SimpleNamespace(scan_mode="general")
        agent.state = AgentState(agent_name="Root Agent", parent_id=None)
        return agent

    snapshot_path = Path(tmp_path) / "last_emotion.json"

    writer = build_agent()
    writer.state.update_context("emotion_happiness", 24.0)
    writer.state.update_context("emotion_sadness", 11.0)
    writer.state.update_context("emotion_stress", 39.0)
    writer.state.update_context("emotion_neutral", 67.0)
    writer.state.update_context("emotion_confidence", 82.0)
    writer.state.update_context("emotion_curiosity", 58.0)
    writer.state.update_context("emotion_tone", "stabilizing")
    writer.state.update_context("emotion_signal_category", "critical")
    writer.state.update_context("emotion_signal_strength", 0.83)
    writer._last_emotion_store_path = lambda: snapshot_path

    assert writer.persist_last_emotion_snapshot()
    assert snapshot_path.exists()

    reader = build_agent()
    reader._last_emotion_store_path = lambda: snapshot_path
    reader._restore_last_emotion_snapshot()

    assert reader.state.context["emotion_happiness"] == 24.0
    assert reader.state.context["emotion_stress"] == 39.0
    assert reader.state.context["boot_last_emotion_pending"] is True

    lines = reader._consume_last_emotion_boot_lines()

    assert lines
    assert any("startup carryover" in line.lower() for line in lines)
    assert reader.state.context["boot_last_emotion_pending"] is False
