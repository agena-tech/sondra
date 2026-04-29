from datetime import UTC, datetime


class Amygdala:
    def __init__(self, decay_rate=0.97, reinforce_step=0.1, min_importance=0.05):
        self.decay_rate = decay_rate
        self.reinforce_step = reinforce_step
        self.min_importance = min_importance

    def decay(self, importance, created_at):
        if not created_at:
            return importance
        try:
            created = datetime.fromisoformat(created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            else:
                created = created.astimezone(UTC)
            now = datetime.now(UTC)
            hours = (now - created).total_seconds() / 3600.0
            value = importance * (self.decay_rate ** hours)
            return max(value, self.min_importance)
        except Exception:
            return importance

    def reinforce(self, importance):
        try:
            return min(importance + self.reinforce_step, 1.0)
        except Exception:
            return importance

    def apply(self, base_score, importance):
        return base_score + (importance * 0.4)
