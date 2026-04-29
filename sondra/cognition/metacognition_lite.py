from __future__ import annotations

from collections import deque


class MetacognitionLite:
    """Small self-model for runtime self-evaluation (no persistence)."""

    def __init__(
        self,
        *,
        abilities: list[str] | None = None,
        confidence_history_size: int = 30,
        outcome_history_size: int = 50,
        max_failure_patterns: int = 20,
    ) -> None:
        self.abilities = list(abilities or ["memory", "tool_usage"])
        self._confidence_history: deque[float] = deque(maxlen=max(5, int(confidence_history_size)))
        self._recent_outcomes: deque[bool] = deque(maxlen=max(5, int(outcome_history_size)))
        self._failure_patterns: dict[str, int] = {}
        self._max_failure_patterns = max(3, int(max_failure_patterns))

    def record_confidence(self, value: float) -> None:
        try:
            normalized = float(value)
        except Exception:
            normalized = 0.0
        normalized = max(0.0, min(1.0, normalized))
        self._confidence_history.append(normalized)

    def record_event(self, action: str, success: bool) -> None:
        action_key = str(action or "").strip().lower()
        if not action_key:
            return
        outcome = bool(success)
        self._recent_outcomes.append(outcome)

        if not outcome:
            self._failure_patterns[action_key] = int(self._failure_patterns.get(action_key, 0)) + 1
        elif action_key in self._failure_patterns:
            next_count = int(self._failure_patterns.get(action_key, 0)) - 1
            if next_count <= 0:
                self._failure_patterns.pop(action_key, None)
            else:
                self._failure_patterns[action_key] = next_count

        if len(self._failure_patterns) > self._max_failure_patterns:
            top_items = sorted(
                self._failure_patterns.items(),
                key=lambda item: item[1],
                reverse=True,
            )[: self._max_failure_patterns]
            self._failure_patterns = dict(top_items)

    def recent_failure_ratio(self, limit: int = 10) -> float:
        safe_limit = max(1, int(limit))
        if not self._recent_outcomes:
            return 0.0
        outcomes = list(self._recent_outcomes)[-safe_limit:]
        if not outcomes:
            return 0.0
        failures = sum(1 for value in outcomes if not bool(value))
        return float(failures) / float(len(outcomes))

    def dominant_failure_pattern(self) -> tuple[str, int]:
        if not self._failure_patterns:
            return ("", 0)
        action, count = max(self._failure_patterns.items(), key=lambda item: item[1])
        return (str(action), int(count))

    def snapshot(self) -> dict:
        action, count = self.dominant_failure_pattern()
        return {
            "abilities": list(self.abilities),
            "confidence_history": list(self._confidence_history),
            "failure_patterns": sorted(
                self._failure_patterns.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5],
            "dominant_failure_action": action,
            "dominant_failure_count": int(count),
        }

