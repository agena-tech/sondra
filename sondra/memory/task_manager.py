from __future__ import annotations

import re
from typing import Any

from .signal_catalog import normalize_signal_text


def _fallback_task_state(text: str) -> dict[str, str] | None:
    body = str(text or "").strip()
    if not body or "?" in body:
        return None

    normalized = normalize_signal_text(body)
    if not normalized:
        return None

    goal = ""
    step = ""
    raw_patterns = [
        (re.compile(r"\bmy goal is\s+(.+)", re.IGNORECASE), "goal"),
        (re.compile(r"\bhedefim\s+(.+)", re.IGNORECASE), "goal"),
        (re.compile(r"\bcurrent step is\s+(.+)", re.IGNORECASE), "step"),
        (re.compile(r"\bi am currently\s+(.+)", re.IGNORECASE), "step"),
        (re.compile(r"\bşu anki ad[ıi]m[ıi]m\s+(.+)", re.IGNORECASE), "step"),
        (re.compile(r"\bşu an\s+(.+?)\s+yapıyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bşu anda\s+(.+?)\s+yapıyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bşuanda\s+(.+?)\s+yapıyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bşu an\s+(.+?)\s+üzerinde çalışıyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bşu anda\s+(.+?)\s+üzerinde çalışıyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bşuanda\s+(.+?)\s+üzerinde çalışıyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bşu an\s+(.+?)\s+kontrol ediyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bşu anda\s+(.+?)\s+kontrol ediyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bşuanda\s+(.+?)\s+kontrol ediyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bşu an\s+(.+?)\s+inceliyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bşu anda\s+(.+?)\s+inceliyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bşuanda\s+(.+?)\s+inceliyorum\b", re.IGNORECASE), "step"),
    ]
    normalized_patterns = [
        (re.compile(r"\bmy goal is\s+(.+)", re.IGNORECASE), "goal"),
        (re.compile(r"\bhedefim\s+(.+)", re.IGNORECASE), "goal"),
        (re.compile(r"\bcurrent step is\s+(.+)", re.IGNORECASE), "step"),
        (re.compile(r"\bi am currently\s+(.+)", re.IGNORECASE), "step"),
        (re.compile(r"\bsu anki adimim\s+(.+)", re.IGNORECASE), "step"),
        (re.compile(r"\bsu an\s+(.+?)\s+yapiyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bsu anda\s+(.+?)\s+yapiyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bsuanda\s+(.+?)\s+yapiyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bsu an\s+(.+?)\s+uzerinde calisiyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bsu anda\s+(.+?)\s+uzerinde calisiyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bsuanda\s+(.+?)\s+uzerinde calisiyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bsu an\s+(.+?)\s+kontrol ediyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bsu anda\s+(.+?)\s+kontrol ediyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bsuanda\s+(.+?)\s+kontrol ediyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bsu an\s+(.+?)\s+inceliyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bsu anda\s+(.+?)\s+inceliyorum\b", re.IGNORECASE), "step"),
        (re.compile(r"\bsuanda\s+(.+?)\s+inceliyorum\b", re.IGNORECASE), "step"),
    ]
    for pattern, field in [*raw_patterns, *normalized_patterns]:
        source_text = body if (pattern, field) in raw_patterns else normalized
        match = pattern.search(source_text)
        if not match:
            continue
        value = str(match.group(1) or "").strip().strip(" .,!?:;\"'")
        if not value:
            continue
        if field == "goal" and not goal:
            goal = value[:160]
        if field == "step" and not step:
            step = value[:160]
    if not goal and not step:
        return None
    return {"goal": goal, "step": step}


async def update_task_state(agent: Any, text: str) -> dict[str, str] | None:
    body = str(text or "").strip()
    if not body:
        return None

    fallback = _fallback_task_state(body)
    if fallback:
        return fallback

    prompt = f"""
What is the user's goal and current step?

Text:
{body}

Return JSON:
goal, step
"""
    try:
        content = await agent._run_memory_llm_prompt(prompt, timeout=8)
    except Exception:  # noqa: BLE001
        return _fallback_task_state(body)

    raw = str(content or "").strip()
    if not raw:
        return None

    loaded = agent._load_json_object(raw)
    if loaded is None:
        return _fallback_task_state(body)

    goal = str(loaded.get("goal", "") or "").strip()
    step = str(loaded.get("step", "") or "").strip()
    if not goal and not step:
        return _fallback_task_state(body)
    return {"goal": goal, "step": step}
