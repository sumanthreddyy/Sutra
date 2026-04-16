"""Prediction engine — anticipates user needs based on patterns and context.

Uses pattern data + current context to generate predictions about what
the user might need, ask, or want next.
"""

import logging
from datetime import datetime
from typing import Any

from providers.base import Message

logger = logging.getLogger(__name__)

PREDICTION_PROMPT = """You are Sutra's intuition module. Based on behavioral patterns and current context,
predict what the user might need or want next.

Rules:
- Only predict if you have genuine signal. Don't make things up.
- Predictions should be actionable ("you might want to...") not vague ("you seem busy")
- Return 1-3 predictions, each one sentence. Or "NONE" if no strong signal.

Behavioral patterns:
{patterns}

Current context:
- Time: {time}
- Day: {day}
- Active goals: {goals}
- Last topic: {last_topic}
- Anomalies: {anomalies}

Generate predictions:"""


class PredictionEngine:
    """Generates predictions about user needs."""

    def __init__(self, pattern_detector: Any, goal_tracker: Any):
        self._patterns = pattern_detector
        self._goals = goal_tracker
        self._prediction_cache: str = ""
        self._cache_time: float = 0.0
        self._cache_ttl: float = 300.0  # 5 min cache

    async def predict(
        self,
        router: Any,
        last_messages: list[Message] | None = None,
    ) -> list[str]:
        """Generate predictions based on current context."""
        import time

        # Cache check
        if (time.time() - self._cache_time) < self._cache_ttl and self._prediction_cache:
            return self._parse_predictions(self._prediction_cache)

        # Gather context
        pattern_summary = self._patterns.get_summary(days=7)
        anomalies = self._patterns.detect_anomalies(days=7)
        goals_summary = self._goals.get_active_summary() if self._goals else ""

        last_topic = "none"
        if last_messages:
            for msg in reversed(last_messages):
                if msg.role == "user":
                    last_topic = msg.content[:100]
                    break

        now = datetime.now()
        prompt = PREDICTION_PROMPT.format(
            patterns=pattern_summary,
            time=now.strftime("%H:%M"),
            day=now.strftime("%A"),
            goals=goals_summary or "None set",
            last_topic=last_topic,
            anomalies=", ".join(anomalies) if anomalies else "None",
        )

        response = await router.complete(
            messages=[
                Message(role="system", content=prompt),
                Message(role="user", content="What might I need next?"),
            ],
            task_type="classification",
        )

        self._prediction_cache = response.content
        self._cache_time = time.time()

        return self._parse_predictions(response.content)

    def _parse_predictions(self, text: str) -> list[str]:
        """Parse prediction text into a list of predictions."""
        if text.strip().upper() == "NONE":
            return []
        lines = [l.strip().lstrip("- ").lstrip("123456789.").strip()
                 for l in text.strip().split("\n") if l.strip()]
        return [l for l in lines if l and l.upper() != "NONE"]

    async def get_prefetch_suggestions(self, router: Any) -> list[str]:
        """Suggest information to pre-fetch based on predicted needs."""
        predictions = await self.predict(router)
        if not predictions:
            return []

        # For now, just return predictions as suggestions
        # In future: actually pre-fetch web searches, files, etc.
        return [f"Pre-fetch suggestion: {p}" for p in predictions]
