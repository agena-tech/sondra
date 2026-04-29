from collections import Counter
from datetime import UTC, datetime


class EpisodicMemory:
    def __init__(self, max_events: int = 1000):
        self.events: list[dict] = []
        self.max_events = int(max_events)

    def add_event(self, action: str, result: str, success: bool, metadata: dict | None = None):
        event = {
            "action": str(action or ""),
            "result": str(result or ""),
            "success": bool(success),
            "metadata": dict(metadata or {}),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self.events.append(event)

        # prevent memory overflow
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

    def recent(self, limit: int = 5):
        return self.events[-max(1, int(limit)) :]

    def search(self, keyword: str):
        needle = str(keyword or "").lower()
        return [
            e
            for e in self.events
            if needle in str(e.get("action", "")).lower() or needle in str(e.get("result", "")).lower()
        ]

    def to_semantic_candidates(self, min_repeats: int = 2):
        action_counter = Counter()
        failure_counter = Counter()
        window = min(50, len(self.events))
        recent_events = self.events[-window:]

        for event in recent_events:
            action = str(event.get("action", "") or "").strip()
            success = bool(event.get("success", False))
            if not action:
                continue
            key = (action, success)
            action_counter[key] += 1
            if not success:
                failure_counter[action] += 1

        candidates: list[dict] = []

        # repeated success patterns
        for (action, success), count in action_counter.items():
            if success and count >= max(max(1, int(min_repeats)), 3):
                candidates.append(
                    {
                        "type": "success_pattern",
                        "text": f"[RECENT] Action '{action}' usually succeeds in similar contexts",
                        "importance": 0.6 + min(0.3, count * 0.05),
                    }
                )

        # repeated failure patterns
        for action, count in failure_counter.items():
            if count >= max(1, int(min_repeats)):
                candidates.append(
                    {
                        "type": "failure_pattern",
                        "text": f"[RECENT] Action '{action}' often fails in similar contexts",
                        "importance": 0.7 + min(0.3, count * 0.05),
                    }
                )

        return candidates
