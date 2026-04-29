class DecisionEngine:
    def __init__(self):
        pass

    def decide(self, state: dict) -> dict:
        """
        state = {
            "confidence": float,
            "stress": float,
            "curiosity": float,
            "memory_hits": int,
            "query_complexity": float
        }
        """

        decision = {
            "use_tool": False,
            "depth": "medium",
            "explore": False,
            "strategy": "normal",
            "temperature": 0.5,
            "max_depth": None,
            "confidence": float(state.get("confidence", 0.0)),
            "mood_state": str(state.get("mood_state", "calm") or "calm").strip().lower(),
            "tone_state": str(state.get("tone_state", "balanced") or "balanced").strip().lower(),
        }

        # Tool usage decision
        if float(state.get("confidence", 0.0)) < 0.4:
            decision["use_tool"] = True
        failure_ratio = float(state.get("recent_failure_ratio", 0.0))
        decision["use_tool"] = bool(decision.get("use_tool", False) or (failure_ratio > 0.5))
        self_failure_ratio = float(state.get("self_failure_ratio", 0.0))
        if self_failure_ratio > 0.6:
            decision["use_tool"] = True
        if self_failure_ratio > 0.75:
            decision["explore"] = False

        # Depth control
        if float(state.get("curiosity", 0.0)) > 0.7:
            decision["depth"] = "deep"
        elif float(state.get("stress", 0.0)) > 0.6:
            decision["depth"] = "shallow"

        # Exploration trigger
        if int(state.get("memory_hits", 0)) < 2:
            decision["explore"] = True

        strategy = str(state.get("strategy", "normal") or "normal").strip().lower()
        if strategy not in {"normal", "fallback", "panic", "explore"}:
            strategy = "normal"
        decision["strategy"] = strategy
        planned = str(state.get("planned_action", "") or "").strip().lower()
        raw_scores = state.get("action_scores", {})
        scores = raw_scores if isinstance(raw_scores, dict) else {}
        safe_scores: dict[str, float] = {}
        for key, value in scores.items():
            with_safety = float(value) if isinstance(value, (int, float)) else 0.0
            safe_scores[str(key)] = max(0.0, min(with_safety, 1.0))
        planning_confidence = max(safe_scores.values()) if safe_scores else 0.5
        decision["planning_confidence"] = max(0.0, min(float(planning_confidence), 1.0))
        mood_state = str(state.get("mood_state", "calm") or "calm").strip().lower()
        if mood_state not in {"calm", "stressed", "confident"}:
            mood_state = "calm"
        decision["mood_state"] = mood_state
        tone_state = str(state.get("tone_state", "balanced") or "balanced").strip().lower()
        if tone_state not in {"balanced", "steady", "warm", "empathetic", "stabilizing"}:
            tone_state = "balanced"
        decision["tone_state"] = tone_state

        if strategy == "panic":
            decision["use_tool"] = True
            decision["explore"] = False
            decision["max_depth"] = 1
            decision["confidence"] = max(
                0.0,
                min(float(decision.get("confidence", 0.0)) * 0.7, 1.0),
            )
        elif strategy == "fallback":
            decision["use_tool"] = True
            decision["explore"] = False
            decision["temperature"] = 0.2
        elif strategy == "explore":
            decision["temperature"] = 0.8

        # Planning bias (soft): never overrides panic/fallback hard safety.
        if strategy not in {"panic", "fallback"}:
            memory_search_score = float(safe_scores.get("memory_search", 0.0))
            direct_answer_score = float(safe_scores.get("direct_answer", 0.0))
            if planned == "memory_search" and memory_search_score > 0.6:
                decision["use_tool"] = True
            elif planned == "direct_answer" and direct_answer_score > 0.6:
                if failure_ratio < 0.5 and float(state.get("confidence", 0.0)) >= 0.4:
                    decision["use_tool"] = False

        # Mood is a low-impact global signal; strategy remains the stronger controller.
        if strategy == "normal":
            if mood_state == "stressed":
                decision["confidence"] = max(
                    0.0,
                    min(float(decision.get("confidence", 0.0)) * 0.9, 1.0),
                )
                decision["explore"] = False
            elif mood_state == "confident":
                decision["confidence"] = max(
                    0.0,
                    min(float(decision.get("confidence", 0.0)) * 1.05, 1.0),
                )

        if self_failure_ratio > 0.0:
            decision["confidence"] = max(
                0.0,
                min(
                    float(decision.get("confidence", 0.0))
                    * (1.0 - min(0.2, self_failure_ratio * 0.2)),
                    1.0,
                ),
            )

        return decision
