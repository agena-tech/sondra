from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from typing import Any

from .signal_catalog import normalize_signal_text


logger = logging.getLogger(__name__)


def _normalize_memory_query(text: str) -> str:
    return normalize_signal_text(text)

def _memory_query_intent(query: str) -> str:
    normalized = _normalize_memory_query(query)
    if not normalized:
        return "general"
    if any(
        phrase in normalized
        for phrase in (
            "my name",
            "what is my name",
            "what was my name",
            "benim adim",
            "adim ne",
            "adim neydi",
            "ismim",
            "kimim",
        )
    ):
        return "name"
    if any(phrase in normalized for phrase in ("my hobby", "hobim", "hobby")):
        return "hobby"
    if any(
        phrase in normalized
        for phrase in ("my interest", "my interests", "ilgi alanim", "ilgileniyorum", "interested in")
    ):
        return "interest"
    if any(phrase in normalized for phrase in ("my goal", "hedefim", "goal")):
        return "goal"
    if any(
        phrase in normalized
        for phrase in (
            "current step",
            "su anki adimim",
            "su an ne yapiyorum",
            "su anda ne yapiyorum",
            "suanda ne yapiyorum",
            "what is my step",
        )
    ):
        return "step"
    return "general"


def _line_matches_intent(text: str, intent: str) -> bool:
    normalized = _normalize_memory_query(text)
    if not normalized:
        return False
    if intent == "name":
        if any(phrase in normalized for phrase in ("su anki adimim", "current step")):
            return False
        return any(
            phrase in normalized
            for phrase in ("user_name:", "user name is", "users name is", "my name is", "call me", "benim adim", "adim ")
        )
    if intent == "hobby":
        return any(phrase in normalized for phrase in ("hobby", "hobim"))
    if intent == "interest":
        return any(phrase in normalized for phrase in ("interested in", "interest", "ilgi alanim", "ilgileniyorum"))
    if intent == "goal":
        return any(phrase in normalized for phrase in ("goal", "hedefim"))
    if intent == "step":
        return any(
            phrase in normalized
            for phrase in ("current step", "su anki adimim", "su anda", "suanda", "i am currently")
        )
    return True


def _render_profile_fact_line(fact: str) -> str:
    raw = str(fact or "").strip()
    lowered = raw.lower()
    if lowered.startswith("user_name:"):
        value = raw.split(":", 1)[1].strip(" .")
        return f"[PROFILE] USER NAME: {value}"
    return f"[PROFILE] {raw}"


def _render_semantic_line(text: str) -> str:
    raw = str(text or "").strip()
    normalized = _normalize_memory_query(raw)
    if normalized.startswith("users hobby is "):
        value = raw.split(" is ", 1)[-1].strip(" .")
        return f"[SEM] USER HOBBY: {value}"
    if normalized.startswith("user is interested in "):
        value = raw.split(" in ", 1)[-1].strip(" .")
        return f"[SEM] USER INTEREST: {value}"
    if normalized.startswith("users goal is "):
        value = raw.split(" is ", 1)[-1].strip(" .")
        return f"[SEM] USER GOAL: {value}"
    if normalized.startswith("user name is ") or normalized.startswith("users name is "):
        value = raw.split(" is ", 1)[-1].strip(" .")
        return f"[SEM] USER NAME: {value}"
    return f"[SEM] USER FACT: {raw}"


def _render_task_state_lines(agent: Any, intent: str) -> list[str]:
    session_id = str(agent.state.context.get("conversation_session_id", agent.state.agent_id) or "").strip()
    state = agent.memory_store.get_task_state(session_id=session_id) if agent.memory_store else None
    if not state:
        return []
    goal = str(state.get("goal", "") or "").strip()
    step = str(state.get("current_step", "") or "").strip()
    lines: list[str] = []
    if intent == "goal" and goal:
        lines.append(f"[TASK] USER GOAL: {goal}")
    if intent == "step" and step:
        lines.append(f"[TASK] USER CURRENT STEP: {step}")
    return lines


def _render_recent_hit_line(hit: dict[str, Any]) -> str:
    role_raw = str(hit.get("role", "") or "").strip().lower()
    role = "USER" if role_raw == "user" else "ASSISTANT"
    body = str(hit.get("body", "") or "").strip()
    relative = str(hit.get("relative_time", "") or "").strip().upper()
    citation = str(hit.get("citation", "") or "").strip()
    source = str(hit.get("source", "") or "").strip()
    line = f"{role}: {body}"
    if relative:
        line += f" ({relative})"
    if citation:
        line = f"[{citation}] {line}"
    if source:
        line += f" | Source: {source}"
    return line


def fallback_memory_rows_from_recent(
    agent: Any,
    limit: int = 8,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    if not agent.memory_store:
        return []
    rows: list[dict[str, Any]] = []
    recent = agent.memory_store.recent_conversation(limit=max(1, limit), session_id=session_id)
    for msg in recent:
        raw = str(msg.content or "").strip()
        if not raw:
            continue
        lowered = raw.lower()
        if "<function=" in lowered or "<parameter=" in lowered or "</function>" in lowered:
            continue
        if "execution was cancelled" in lowered or "waiting for new instructions" in lowered:
            continue
        role_value = str(msg.role or "").strip().lower()
        role = "user" if role_value == "user" else "assistant"
        body = raw
        if role == "user" and body.lower().startswith("user:"):
            body = body[5:].strip()
        if role == "assistant" and body.lower().startswith("assistant:"):
            body = body[10:].strip()
        if not body:
            continue
        source = f"memory.db#msg-{int(msg.id)}"
        rows.append(
            {
                "citation": f"C{len(rows) + 1}",
                "id": int(msg.id),
                "timestamp": str(msg.timestamp or ""),
                "relative_time": "",
                "role": role,
                "content": raw,
                "body": body,
                "source": source,
                "source_path": "memory.db",
                "source_line": int(msg.id),
                "score": 0.0,
                "bm25_score": 0.0,
                "vector_score": 0.0,
                "lexical_score": 0.0,
                "recency_score": 0.0,
                "temporal_score": 0.0,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def search_memory_rows(agent: Any, query: str, limit: int = 8) -> list[dict[str, Any]]:
    if not agent.memory_store:
        return []
    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        cleaned_query = "recent user statements"
    current_turn = str(agent.state.context.get("last_user_turn_raw", "") or "").strip()
    with contextlib.suppress(Exception):
        sync_mode = str(getattr(agent.memory_store, "sync_mode", "off") or "off").strip().lower()
        should_sync_on_search = sync_mode == "on_search"
        if (
            agent.memory_index_manager
            and should_sync_on_search
            and current_turn
            and agent._memory_sync_turn_key != current_turn
        ):
            agent.memory_index_manager.maybe_sync(trigger="on_search", limit=20)
            agent._memory_sync_turn_key = current_turn
    session_id = str(agent.state.context.get("conversation_session_id", agent.state.agent_id) or "").strip()
    try:
        candidate_limit = max(80, min(300, max(1, limit) * 30))
        rows = agent.memory_store.search_conversation_structured(
            cleaned_query,
            top_k=max(1, limit),
            candidate_limit=candidate_limit,
            session_id=session_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Memory search failed, using recent fallback: %s", exc)
        rows = []
    if not rows:
        rows = fallback_memory_rows_from_recent(agent, limit=max(1, limit), session_id=session_id)
    rows = sorted(
        rows,
        key=lambda row: 0 if str(row.get("role", "")).strip().lower() == "user" else 1,
    )
    for idx, row in enumerate(rows, start=1):
        row["citation"] = f"C{idx}"
    return rows[: max(1, limit)]


def build_memory_status_lines(
    *,
    status: str,
    reason: str,
    action: str,
    provider: str = "",
    detail: str = "",
) -> list[str]:
    lines = [
        "[MEMORY STATUS]",
        f"status: {str(status or '').strip()}",
        f"reason: {str(reason or '').strip()}",
    ]
    provider_value = str(provider or "").strip()
    detail_value = str(detail or "").strip()
    if provider_value and detail_value:
        lines.append(f"provider_detail: {provider_value} | {detail_value}")
    elif provider_value:
        lines.append(f"provider: {provider_value}")
    elif detail_value:
        lines.append(f"detail: {detail_value}")
    lines.append(f"action: {str(action or '').strip()}")
    return lines


def extract_memory_source_id(token: str) -> int:
    value = str(token or "").strip().lower()
    if not value:
        return 0
    marker = "msg-"
    pos = value.find(marker)
    if pos < 0:
        return 0
    tail = value[pos + len(marker) :].strip()
    digits: list[str] = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return 0
    return int("".join(digits))


def memory_search(agent: Any, query: str) -> list[str]:
    if not agent._is_general_root_agent():
        return []
    if not agent.memory_store:
        return build_memory_status_lines(
            status="unavailable",
            reason="memory_store_not_initialized",
            action="answer normally and say memory is unavailable",
        )
    health = agent._memory_health_snapshot()
    total_messages = int(health.get("total_messages", 0) or 0)
    if total_messages <= 0:
        return build_memory_status_lines(
            status="unavailable",
            reason="no_persisted_messages",
            action="ask user for details and continue",
        )
    provider_error = str(health.get("embed_provider_last_error", "") or "").strip()
    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        cleaned_query = agent._latest_user_message_raw().strip()
    if not cleaned_query:
        cleaned_query = "recent user statements"
    intent = _memory_query_intent(cleaned_query)

    hits = search_memory_rows(agent, cleaned_query, limit=8)
    agent._last_memory_hits = list(hits)

    rendered: list[str] = []
    seen: set[str] = set()
    if intent == "name":
        session_id = str(agent.state.context.get("conversation_session_id", agent.state.agent_id) or "").strip()
        for fact in agent.memory_store.get_profile_facts(top_k=6, session_id=session_id):
            if not _line_matches_intent(fact, intent):
                continue
            profile_line = _render_profile_fact_line(fact)
            profile_key = profile_line.lower()
            if profile_key in seen:
                continue
            seen.add(profile_key)
            rendered.append(profile_line)
            if len(rendered) >= 6:
                return rendered

    for task_line in _render_task_state_lines(agent, intent):
        task_key = task_line.lower()
        if task_key in seen:
            continue
        seen.add(task_key)
        rendered.append(task_line)
        if len(rendered) >= 6:
            return rendered

    if intent != "general":
        for hit in hits:
            if str(hit.get("role", "") or "").strip().lower() != "user":
                continue
            body = str(hit.get("body", "") or "").strip()
            if not body or not _line_matches_intent(body, intent):
                continue
            line = _render_recent_hit_line(hit)
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            rendered.append(line)
            if len(rendered) >= 6:
                return rendered

    session_id = str(agent.state.context.get("conversation_session_id", agent.state.agent_id) or "").strip()
    semantic_items = agent.memory_store.get_semantic_memory(limit=12, reinforce=False, session_id=session_id)
    for item in semantic_items:
        semantic_text = str(item or "").strip()
        if not semantic_text:
            continue
        semantic_lower = semantic_text.lower()
        if "<function=" in semantic_lower or "<parameter=" in semantic_lower:
            continue
        if intent != "general" and not _line_matches_intent(semantic_text, intent):
            continue
        semantic_line = _render_semantic_line(semantic_text)
        semantic_key = semantic_line.lower()
        if semantic_key in seen:
            continue
        seen.add(semantic_key)
        rendered.append(semantic_line)
        if len(rendered) >= 6:
            return rendered

    for hit in hits:
        body = str(hit.get("body", "") or "").strip()
        if not body:
            continue
        if intent != "general" and not _line_matches_intent(body, intent):
            continue
        line = _render_recent_hit_line(hit)
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        rendered.append(line)
        if len(rendered) >= 6:
            break
    if not rendered:
        if provider_error:
            return build_memory_status_lines(
                status="warning",
                reason="embedding_provider_error",
                provider=str(health.get("embed_provider", "") or "").strip(),
                detail=provider_error,
                action="answer cautiously and ask user to re-confirm key details",
            )
        return [
            "[MEMORY STATUS]",
            "status: empty",
            "reason: no_matching_result",
            "action: answer briefly and ask user to restate details if needed",
        ]
    return rendered


def memory_get(
    agent: Any,
    citation: str = "",
    id: str = "",
    source: str = "",
    query: str = "",
    index: str = "",
) -> str:
    if not agent._is_general_root_agent():
        return "Memory is unavailable."
    if not agent.memory_store:
        return "\n".join(
            build_memory_status_lines(
                status="unavailable",
                reason="memory_store_not_initialized",
                action="run without memory for this answer",
            )
        )

    token = str(citation or "").strip()
    id_token = str(id or "").strip()
    source_token = str(source or "").strip()
    query_token = str(query or "").strip()
    index_token = str(index or "").strip()
    target_id = 0
    if token:
        upper = token.upper()
        if upper.startswith("C"):
            idx_raw = upper[1:].strip()
            if idx_raw.isdigit():
                idx = int(idx_raw)
                if 1 <= idx <= len(agent._last_memory_hits):
                    target_id = int(agent._last_memory_hits[idx - 1].get("id", 0) or 0)
        elif token.isdigit():
            target_id = int(token)
        else:
            target_id = extract_memory_source_id(token)

    if target_id <= 0 and id_token:
        if id_token.isdigit():
            target_id = int(id_token)
        else:
            target_id = extract_memory_source_id(id_token)
    if target_id <= 0 and source_token:
        target_id = extract_memory_source_id(source_token)
    if target_id <= 0 and index_token.isdigit():
        idx_value = int(index_token)
        if 1 <= idx_value <= len(agent._last_memory_hits):
            target_id = int(agent._last_memory_hits[idx_value - 1].get("id", 0) or 0)
    if target_id <= 0 and query_token:
        query_hits = search_memory_rows(agent, query_token, limit=1)
        if query_hits:
            target_id = int(query_hits[0].get("id", 0) or 0)

    if target_id <= 0 and agent._last_memory_hits:
        target_id = int(agent._last_memory_hits[0].get("id", 0) or 0)
    if target_id <= 0:
        return "No memory citation is available. Run memory_search first."

    entry = agent.memory_store.get_conversation_message_by_id(target_id)
    if not entry:
        return f"No memory entry found for citation '{token or str(target_id)}'."

    role_raw = str(entry.get("role", "") or "").strip().lower()
    role = "USER" if role_raw == "user" else "ASSISTANT"
    body = str(entry.get("body", "") or "").strip()
    source_value = str(entry.get("source", "") or f"memory.db#msg-{target_id}")
    relative = str(entry.get("relative_time", "") or "").strip().upper()
    exact_date = str(entry.get("exact_date", "") or "").strip()

    result_lines = [f"{role}: {body}", f"Source: {source_value}"]
    if exact_date:
        result_lines.append(f"Date: {exact_date}")
    if relative:
        result_lines.append(f"Relative: {relative}")
    return "\n".join(result_lines)


def persist_general_messages_to_disk(agent: Any) -> None:
    if not agent.memory_store or not agent._is_general_root_agent():
        return
    session_id = str(agent.state.context.get("conversation_session_id", agent.state.agent_id))
    start_idx = int(agent.state.context.get("memory_last_persisted_index", 0))
    messages = agent.state.messages
    if start_idx >= len(messages):
        return

    rows: list[tuple[str, str, str, str]] = []
    for msg in messages[start_idx:]:
        role = str(msg.get("role", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        content = agent._strip_internal_metadata_blocks(str(msg.get("content", ""))).strip()
        if not content:
            continue
        if content.startswith("Tool Results:"):
            continue
        if "<tool_result>" in content or "<inter_agent_message>" in content:
            continue
        lowered_content = content.lower()
        if "<function=" in lowered_content or "<parameter=" in lowered_content or "</function>" in lowered_content:
            continue
        if "execution was cancelled" in lowered_content or "waiting for new instructions" in lowered_content:
            continue
        if agent._is_internal_metadata_text(content):
            continue
        if agent._is_internal_control_user_message(content):
            continue
        if content.startswith("<relevant_past_conversation>"):
            continue
        if agent._is_memory_context_text(content):
            continue
        if content in {
            agent.PERSISTENT_MEMORY_READING_MESSAGE,
            agent.PERSISTENT_MEMORY_DELETING_MESSAGE,
            agent.PERSISTENT_MEMORY_RESET_MESSAGE,
        }:
            continue
        # Legacy status variants (kept only as cleanup filter; not emitted anymore).
        if (
            "reading from disk" in lowered_content
            or "llm reading persistent memory" in lowered_content
            or "llm is trying to remember" in lowered_content
        ):
            continue
        if content.startswith("Initializing ..."):
            continue

        timestamp = str(msg.get("timestamp") or datetime.now(UTC).isoformat())
        rows.append((timestamp, role, content, session_id))

    if not rows:
        agent.state.update_context("memory_last_persisted_index", len(messages))
        return

    with contextlib.suppress(Exception):
        agent.memory_store.add_conversation_messages(rows)
        agent.state.update_context("memory_last_persisted_index", len(messages))


