from __future__ import annotations

import re
from typing import Any

from .signal_catalog import normalize_signal_text


def compute_initial_importance(text: str) -> float:
    text = str(text or "").lower()
    score = 0.4

    if any(k in text for k in ["my name", "i am", "i'm", "call me"]):
        score += 0.4

    if any(k in text for k in ["goal", "i want", "i plan", "i will"]):
        score += 0.3

    if any(k in text for k in ["i like", "i prefer", "i love", "i hate"]):
        score += 0.2

    if any(k in text for k in ["project", "system", "code", "working on"]):
        score += 0.2

    return min(score, 1.0)


def _normalize_signal_text(text: str) -> str:
    return normalize_signal_text(text)

def _clean_fact_value(value: str) -> str:
    cleaned = str(value or "").strip().strip(" .,!?:;\"'")
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:120].strip()


def _fallback_extract_semantic(text: str) -> list[str]:
    raw = str(text or "").strip()
    normalized = _normalize_signal_text(raw)
    normalized_clean = normalized.rstrip(" .,!?:;\"'")
    if not raw or not normalized or "?" in raw:
        return []

    patterns: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\bmy name is\s+(.+)", re.IGNORECASE), "User's name is {value}"),
        (re.compile(r"\bcall me\s+(.+)", re.IGNORECASE), "User's name is {value}"),
        (re.compile(r"\bbenim ad[ıi]m\s+(.+)", re.IGNORECASE), "User's name is {value}"),
        (re.compile(r"\bad[ıi]m\s+(.+)", re.IGNORECASE), "User's name is {value}"),
        (re.compile(r"\bmy hobby is\s+(.+)", re.IGNORECASE), "User's hobby is {value}"),
        (re.compile(r"\bhobim\s+(.+)", re.IGNORECASE), "User's hobby is {value}"),
        (re.compile(r"\bmy interest is\s+(.+)", re.IGNORECASE), "User is interested in {value}"),
        (re.compile(r"\bmy interests are\s+(.+)", re.IGNORECASE), "User is interested in {value}"),
        (re.compile(r"\bilgi alan[ıi]m\s+(.+)", re.IGNORECASE), "User is interested in {value}"),
        (re.compile(r"\bilgileniyorum(?:\s+|:\s*)(.+)", re.IGNORECASE), "User is interested in {value}"),
        (re.compile(r"\b(.+?)ten\s+ho[sş]laniyorum\b", re.IGNORECASE), "User likes {value}"),
        (re.compile(r"\bi am interested in\s+(.+)", re.IGNORECASE), "User is interested in {value}"),
        (re.compile(r"\bi like\s+(.+)", re.IGNORECASE), "User likes {value}"),
        (re.compile(r"\bi prefer\s+(.+)", re.IGNORECASE), "User prefers {value}"),
        (re.compile(r"\bseviyorum\s+(.+)", re.IGNORECASE), "User likes {value}"),
        (re.compile(r"\btercih ederim\s+(.+)", re.IGNORECASE), "User prefers {value}"),
        (re.compile(r"\bmy goal is\s+(.+)", re.IGNORECASE), "User's goal is {value}"),
        (re.compile(r"\bhedefim\s+(.+)", re.IGNORECASE), "User's goal is {value}"),
    ]

    items: list[str] = []
    seen: set[str] = set()
    for pattern, template in patterns:
        match = pattern.search(raw)
        if not match:
            continue
        value = _clean_fact_value(match.group(1))
        if not value:
            continue
        item = template.format(value=value)
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)

    if items:
        return items

    normalized_suffix_patterns = (
        (" ilgim var", "User is interested in {value}"),
        (" ilgimi cekiyor", "User is interested in {value}"),
    )
    for suffix, template in normalized_suffix_patterns:
        if not normalized_clean.endswith(suffix):
            continue
        value = _clean_fact_value(normalized_clean[: -len(suffix)])
        if value:
            return [template.format(value=value)]

    generic_first_person = (
        normalized.startswith("i like ")
        or normalized.startswith("i prefer ")
        or normalized.startswith("i am interested in ")
        or normalized.startswith("ilgileniyorum ")
        or normalized.startswith("ilgi alanim ")
        or normalized.endswith(" ilgim var")
        or normalized.endswith(" ilgimi cekiyor")
        or normalized.startswith("hobim ")
    )
    if generic_first_person:
        value = _clean_fact_value(raw.split(" ", 1)[1] if " " in raw else "")
        if value:
            return [f"User fact: {value}"]
    return []


async def extract_semantic(agent: Any, text: str) -> list[str]:
    body = str(text or "").strip()
    if not body:
        return []
    fallback_items = _fallback_extract_semantic(body)
    if fallback_items:
        return fallback_items
    prompt = f"""
Extract ONLY long-term user facts.

Rules:
- No temporary info
- No repetition
- Short bullet list
- Output each line as a neutral USER fact (e.g., "User likes cybersecurity").
- Do NOT describe assistant identity, role, or specialization.
- Do NOT infer capabilities from user interests.

Text:
{body}
"""
    try:
        content = await agent._run_memory_llm_prompt(prompt, timeout=8)
    except Exception:  # noqa: BLE001
        return _fallback_extract_semantic(body)

    items: list[str] = []
    seen: set[str] = set()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        elif line.startswith("* "):
            line = line[2:].strip()
        elif line.startswith("\u2022 "):
            line = line[2:].strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        if not is_valid_semantic_item(agent, line):
            continue
        seen.add(key)
        items.append(line)
    if items:
        return items
    return _fallback_extract_semantic(body)


def is_valid_semantic_item(agent: Any, item: str) -> bool:
    text = str(item or "").strip()
    if not text:
        return False
    lowered = text.lower()
    normalized = _normalize_signal_text(lowered)
    if normalized in {"-", "none", "n/a", "not provided", "no facts"}:
        return False
    if any(
        normalized.startswith(prefix)
        for prefix in ("no long term user facts", "based on the provided text", "reason:", "rules:", "text:")
    ):
        return False
    if any(snippet in normalized for snippet in ("<function=", "<parameter=", "</function>", "no long term")):
        return False
    if len(text) < 3:
        return False
    if len(text) > 160:
        return False
    return True


