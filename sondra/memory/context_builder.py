from __future__ import annotations

import re
from typing import Any


def _has_lexical_overlap(a: str, b: str) -> bool:
    a_tokens = {token for token in re.findall(r"\w+", str(a or "").lower()) if token}
    b_tokens = {token for token in re.findall(r"\w+", str(b or "").lower()) if token}
    return bool(a_tokens.intersection(b_tokens))


def build_auto_context(
    store: Any,
    query: str,
    session_id: str = "",
    top_k: int = 6,
) -> list[str]:
    """Build deterministic controller-side memory context for the current user turn."""
    text = str(query or "").strip()
    if not text:
        return []

    profile_cap = 5
    task_cap = 5
    semantic_cap = 5
    recent_cap = 5
    exact_cap = 5
    search_cap = 5

    profile = [f"[PROFILE] {item}" for item in store.get_profile_facts(top_k=profile_cap, session_id=session_id)]
    task_lines = [str(item or "").strip() for item in store.get_task_state_lines(session_id=session_id)[:task_cap]]
    semantic_mem: list[str] = []
    for item in store.get_semantic_memory(limit=semantic_cap, reinforce=False, session_id=session_id):
        semantic_text = str(item or "").strip()
        if not semantic_text:
            continue
        if store._is_warning_semantic(semantic_text):
            continue
        semantic_mem.append(f"[SEM] USER FACT: {semantic_text}")
    recent = store.get_recent_conversation_lines(session_id=session_id, limit=max(8, recent_cap))
    recent = recent[:recent_cap]
    semantic_rows = store.search_conversation_structured(
        query=text,
        top_k=4,
        candidate_limit=120,
        session_id=session_id,
    )

    exact_rows = [str(row.get("content", "") or "").strip() for row in semantic_rows]
    exact: list[str] = []
    for row in exact_rows:
        raw = str(row or "").strip()
        if not raw:
            continue
        role, body = store._extract_role_and_body(raw)
        if not body:
            continue
        if role != "user":
            continue
        if store._is_noisy_memory_body(body):
            continue
        role_label = "USER" if role == "user" else "ASSISTANT"
        exact.append(f"[EXACT] {role_label}: {body}")
        if len(exact) >= exact_cap:
            break
    semantic: list[str] = []
    for row in semantic_rows:
        body = str(row.get("body", "") or "").strip()
        role = str(row.get("role", "") or "").strip().lower()
        if not body:
            continue
        if role != "user":
            continue
        if store._is_noisy_memory_body(body):
            continue
        role_label = "USER" if role == "user" else "ASSISTANT"
        semantic.append(f"[SEARCH] {role_label}: {body}")
        if len(semantic) >= search_cap:
            break

    merged: list[str] = []
    seen: set[str] = set()
    blocked_snippets = (
        "### assistant",
        "resume execution",
        "tool descriptions",
        "system prompts",
        "runtime tools currently available",
        "<tool",
    )
    for bucket in [profile, task_lines, semantic_mem, recent, exact, semantic]:
        for item in bucket:
            value = str(item or "").strip()
            if not value:
                continue
            lowered_value = value.lower()
            if any(snippet in lowered_value for snippet in blocked_snippets):
                continue
            if value.startswith("[SEM] USER FACT:") and store._is_warning_semantic(value):
                continue
            source = "user" if ("user:" in lowered_value or value.startswith("[SEM] USER FACT:")) else "assistant"
            if source != "user" and not (
                value.startswith("[PROFILE]") or value.startswith("[TASK]")
            ):
                continue
            key = store._auto_context_dedupe_key(value)
            if key in seen:
                continue
            seen.add(key)
            merged.append(value)

    scored: list[tuple[float, str]] = []
    for idx, text_value in enumerate(merged):
        score = quick_score(store, text_value, text, idx)
        scored.append((score, text_value))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    context = [item for _, item in scored]
    seen_bodies: set[str] = set()
    clean: list[str] = []
    for item in context:
        lowered_item = str(item or "").lower()
        if lowered_item.startswith("[recent]") and not _has_lexical_overlap(text, lowered_item):
            continue
        body = item.split("USER:")[-1].strip()
        if body in seen_bodies:
            continue
        clean.append(item)
        seen_bodies.add(body)
        if len(clean) >= max(1, int(top_k)):
            break
    return clean


def quick_score(store: Any, text: str, query: str, order: int) -> float:
    """Fast deterministic scorer for auto context line ordering."""
    content = str(text or "").strip()
    query_text = str(query or "").strip()
    if not content:
        return 0.0

    lowered = content.lower()
    query_lower = query_text.lower()
    query_tokens = store._tokenize(query_text)
    overlap = store._lexical_score(query_tokens, content)

    score = overlap
    if query_lower and query_lower in lowered:
        score += 0.40
    if lowered.startswith("[profile]"):
        score += 0.30
    if lowered.startswith("[task]"):
        score += 0.45
    if lowered.startswith("[exact]"):
        score += 0.35
    if lowered.startswith("[recent]"):
        score += 0.20
    if lowered.startswith("[search]"):
        score += 0.20
    if lowered.startswith("[sem]"):
        score += 0.15
    if "user:" in lowered:
        score += 0.05
    score += max(0.0, (1000.0 - float(order)) / 1000000.0)
    return score
