from __future__ import annotations

import re
from typing import Any

from .signal_catalog import normalize_signal_text


def extract_profile_fact(agent: Any, text: str) -> None:
    if not agent.memory_store or not agent._is_general_root_agent():
        return
    raw = str(text or "").strip()
    if not raw or "?" in raw:
        return

    normalized = normalize_signal_text(raw)
    if not normalized:
        return

    def _clean_value(value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            return ""
        separators = [".", ",", "!", "?", ";", ":", "/", "\\", "|", "<", ">"]
        stop_idx = len(cleaned)
        for sep in separators:
            idx = cleaned.find(sep)
            if idx >= 0 and idx < stop_idx:
                stop_idx = idx
        cleaned = cleaned[:stop_idx].strip(" \"'")
        if not cleaned:
            return ""
        words = [w for w in cleaned.split() if w]
        if not words:
            return ""

        # Remove lowercase connective particles without damaging proper names.
        while len(words) > 1:
            head = str(words[0] or "").strip()
            if head and head == head.lower() and head.lower() in leading_particles:
                words = words[1:]
                continue
            break
        if not words:
            return ""
        return " ".join(words[:5]).strip()

    blocked_values = {
        "name",
        "my name",
        "what is my name",
        "who am i",
        "adim",
        "adim ne",
        "ismim",
        "kimim",
        "neydi",
    }
    leading_particles = {"da", "de", "is", "also"}

    def _looks_like_name(value: str) -> bool:
        candidate = str(value or "").strip()
        if not candidate:
            return False
        if any(ch in candidate for ch in ("/", "\\", "|", "<", ">", "{", "}", "[", "]")):
            return False
        if not candidate[0].isalnum():
            return False
        normalized_candidate = normalize_signal_text(candidate)
        if not normalized_candidate:
            return False
        if normalized_candidate in blocked_values:
            return False
        if "?" in candidate:
            return False
        tokens = [t for t in candidate.replace("-", " ").replace("'", " ").split() if t]
        if not tokens or len(tokens) > 4:
            return False
        for token in tokens:
            token_norm = normalize_signal_text(token.strip().lower())
            if token_norm in blocked_values:
                return False
            if not token or not token[0].isalnum():
                return False
            if any(ch.isdigit() for ch in token):
                return False
        return True

    raw_patterns = [
        re.compile(r"\bmy name is\s+(.+)", re.IGNORECASE),
        re.compile(r"\bcall me\s+(.+)", re.IGNORECASE),
        re.compile(r"\bbenim ad[ıi]m\s+(.+)", re.IGNORECASE),
        re.compile(r"\bad[ıi]m\s+(.+)", re.IGNORECASE),
    ]
    for pattern in raw_patterns:
        match = pattern.search(raw)
        if not match:
            continue
        name_value = _clean_value(match.group(1))
        if not name_value or len(name_value) < 2 or not _looks_like_name(name_value):
            continue
        fact = f"user_name: {name_value}"
        session_id = str(agent.state.context.get("conversation_session_id", agent.state.agent_id) or "").strip()
        agent.memory_store.store_profile_fact(fact, importance=1.0, session_id=session_id)
        return

    normalized_patterns = [
        re.compile(r"\bmy name is\s+(.+)", re.IGNORECASE),
        re.compile(r"\bcall me\s+(.+)", re.IGNORECASE),
        re.compile(r"\bbenim adim\s+(.+)", re.IGNORECASE),
        re.compile(r"\badim\s+(.+)", re.IGNORECASE),
    ]
    for pattern in normalized_patterns:
        match = pattern.search(normalized)
        if not match:
            continue
        name_value = _clean_value(match.group(1))
        if not name_value or len(name_value) < 2 or not _looks_like_name(name_value):
            continue
        fact = f"user_name: {name_value}"
        session_id = str(agent.state.context.get("conversation_session_id", agent.state.agent_id) or "").strip()
        agent.memory_store.store_profile_fact(fact, importance=1.0, session_id=session_id)
        return
