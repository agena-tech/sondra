from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sondra.memory.signal_catalog import normalize_signal_text
from sondra.utils.resource_paths import get_sondra_resource_path


@dataclass(frozen=True)
class AutoIntentResult:
    scan_mode: str
    scan_level: str
    instruction: str
    targets: list[str]


def _load_intent_signals() -> dict[str, Any]:
    path = get_sondra_resource_path("memory", "intent_signals", "intent_signals.json")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _normalized_markers(data: dict[str, Any], *path: str) -> list[str]:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return []
        current = current.get(key, [])
    if not isinstance(current, list):
        return []
    return [normalize_signal_text(str(item or "")) for item in current if str(item or "").strip()]


def _has_marker(normalized_text: str, markers: list[str]) -> bool:
    return any(marker and marker in normalized_text for marker in markers)


def _strip_request_preamble(text: str) -> str:
    cleaned = str(text or "").strip().strip("\"'")
    if not cleaned:
        return ""
    patterns = (
        r"^\s*senden\s+(?:şunu|sunu)\s+istiyorum\s*[:;,\-–—]?\s*",
        r"^\s*(?:şunu|sunu)\s+istiyorum\s*[:;,\-–—]?\s*",
        r"^\s*senden\s+istediğim\s*[:;,\-–—]?\s*",
        r"^\s*senden\s+istedigim\s*[:;,\-–—]?\s*",
        r"^\s*benim\s+isteğim\s*[:;,\-–—]?\s*",
        r"^\s*benim\s+istegim\s*[:;,\-–—]?\s*",
        r"^\s*lütfen\s+",
        r"^\s*lutfen\s+",
        r"^\s*please\s+",
    )
    for pattern in patterns:
        updated = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        if updated != cleaned:
            cleaned = updated.strip()
            break
    return cleaned or str(text or "").strip()


def _clean_target(value: str) -> str:
    return str(value or "").strip().strip("\"'`<>()[]{}").rstrip(".,;:!?")


def _extract_targets(text: str) -> list[str]:
    raw = str(text or "")
    matches: list[str] = []
    url_pattern = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
    domain_pattern = re.compile(
        r"(?<!@)\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}(?:/[^\s\"'<>]*)?",
        re.IGNORECASE,
    )
    url_spans: list[tuple[int, int]] = []
    for match in url_pattern.finditer(raw):
        target = _clean_target(match.group(0))
        if target and target not in matches:
            matches.append(target)
            url_spans.append(match.span())

    for match in domain_pattern.finditer(raw):
        start, end = match.span()
        if any(url_start <= start and end <= url_end for url_start, url_end in url_spans):
            continue
        target = _clean_target(match.group(0))
        if target and target not in matches:
            matches.append(target)
    return matches


def resolve_auto_intent(request: str) -> AutoIntentResult:
    signals = _load_intent_signals()
    instruction = _strip_request_preamble(request)
    normalized = normalize_signal_text(instruction)

    adb_markers = _normalized_markers(signals, "modes", "adb")
    pentest_markers = _normalized_markers(signals, "modes", "pentest")
    osint_markers = _normalized_markers(signals, "modes", "osint")
    quick_markers = _normalized_markers(signals, "levels", "quick")
    deep_markers = _normalized_markers(signals, "levels", "deep")

    scan_mode = "general"
    targets: list[str] = []
    if _has_marker(normalized, adb_markers):
        scan_mode = "adb"
    elif _has_marker(normalized, pentest_markers):
        scan_mode = "pentest"
        targets = _extract_targets(instruction)
    elif _has_marker(normalized, osint_markers):
        scan_mode = "osint"

    scan_level = "standard"
    if _has_marker(normalized, quick_markers):
        scan_level = "quick"
    elif _has_marker(normalized, deep_markers):
        scan_level = "deep"

    return AutoIntentResult(
        scan_mode=scan_mode,
        scan_level=scan_level,
        instruction=instruction,
        targets=targets,
    )


def build_auto_execute_command(args: Any) -> str:
    parts = ["poetry", "run", "sondra", "-m", args.scan_mode, "-l", args.scan_level]
    if getattr(args, "non_interactive", False):
        parts.append("-n")
    for target in list(getattr(args, "target", []) or []):
        parts.extend(["--target", str(target)])
    if getattr(args, "instruction", None):
        parts.extend(["--instruction", str(args.instruction)])
    if getattr(args, "config", None):
        parts.extend(["--config", str(args.config)])
    if getattr(args, "subagents", None) is not None:
        parts.extend(["--subagents", str(args.subagents)])
    if getattr(args, "retry", None) is not None:
        parts.extend(["--retry", str(args.retry)])
    if getattr(args, "voice_speech", False):
        parts.append("--voice-speech")
    return " ".join(shlex.quote(part) for part in parts)
