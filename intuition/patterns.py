"""Pattern detector — analyzes conversation history for long-term behavioral patterns.

Looks for:
- Time-of-day usage patterns
- Topic frequency and trends
- Mood/energy indicators
- Work habit patterns
"""

import json
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PatternDetector:
    """Analyzes transcripts to find behavioral patterns."""

    def __init__(self, transcript_dir: str):
        self._transcript_dir = Path(transcript_dir)

    def analyze(self, days: int = 30) -> dict[str, Any]:
        """Analyze transcripts from the last N days."""
        transcripts = self._load_recent(days)
        if not transcripts:
            return {"error": "No transcripts found", "patterns": []}

        return {
            "usage_by_hour": self._hourly_usage(transcripts),
            "usage_by_day": self._daily_usage(transcripts),
            "top_topics": self._topic_frequency(transcripts),
            "session_lengths": self._session_lengths(transcripts),
            "activity_trend": self._activity_trend(transcripts),
            "total_sessions": len(set(t.get("_session", "") for t in transcripts)),
            "total_turns": len(transcripts),
            "avg_message_length": self._avg_length(transcripts),
        }

    def detect_anomalies(self, days: int = 7) -> list[str]:
        """Detect unusual patterns compared to baseline."""
        recent = self.analyze(days=days)
        baseline = self.analyze(days=30)
        anomalies = []

        if not recent.get("usage_by_hour") or not baseline.get("usage_by_hour"):
            return anomalies

        # Check if working at unusual hours
        recent_hours = recent["usage_by_hour"]
        baseline_hours = baseline["usage_by_hour"]
        for hour, count in recent_hours.items():
            baseline_count = baseline_hours.get(hour, 0)
            if count > 0 and baseline_count == 0:
                anomalies.append(f"Unusual activity at {hour}:00 (not in your usual pattern)")

        # Check activity level changes
        recent_total = recent.get("total_turns", 0)
        baseline_total = baseline.get("total_turns", 0)
        if baseline_total > 0:
            ratio = recent_total / (baseline_total * days / 30)
            if ratio > 2.0:
                anomalies.append("Activity is 2x higher than usual this week")
            elif ratio < 0.3:
                anomalies.append("Activity is much lower than usual this week")

        return anomalies

    def get_summary(self, days: int = 7) -> str:
        """Get a human-readable pattern summary."""
        data = self.analyze(days)
        if "error" in data:
            return "Not enough data for pattern analysis yet."

        lines = [f"Pattern analysis (last {days} days):"]
        lines.append(f"- {data['total_sessions']} sessions, {data['total_turns']} total messages")
        lines.append(f"- Average message length: {data['avg_message_length']} chars")

        # Peak hours
        hours = data.get("usage_by_hour", {})
        if hours:
            peak = max(hours, key=hours.get)
            lines.append(f"- Peak activity hour: {peak}:00")

        # Peak days
        days_data = data.get("usage_by_day", {})
        if days_data:
            peak_day = max(days_data, key=days_data.get)
            lines.append(f"- Most active day: {peak_day}")

        # Top topics
        topics = data.get("top_topics", [])
        if topics:
            top3 = topics[:3]
            lines.append(f"- Top topics: {', '.join(t[0] for t in top3)}")

        # Anomalies
        anomalies = self.detect_anomalies(min(days, 7))
        if anomalies:
            lines.append("\nAnomalies detected:")
            for a in anomalies:
                lines.append(f"  - {a}")

        return "\n".join(lines)

    def _load_recent(self, days: int) -> list[dict]:
        """Load transcript turns from the last N days."""
        if not self._transcript_dir.exists():
            return []

        cutoff = time.time() - (days * 86400)
        turns = []

        for f in sorted(self._transcript_dir.glob("*.jsonl")):
            session_id = f.stem
            try:
                for line in f.read_text(encoding="utf-8").strip().split("\n"):
                    if not line.strip():
                        continue
                    turn = json.loads(line)
                    ts = turn.get("timestamp", "")
                    if ts:
                        try:
                            dt = datetime.fromisoformat(ts)
                            if dt.timestamp() >= cutoff:
                                turn["_session"] = session_id
                                turn["_dt"] = dt
                                turns.append(turn)
                        except ValueError:
                            pass
            except Exception as e:
                logger.warning(f"Failed to parse {f}: {e}")

        return turns

    def _hourly_usage(self, turns: list[dict]) -> dict[int, int]:
        counter: dict[int, int] = defaultdict(int)
        for t in turns:
            dt = t.get("_dt")
            if dt:
                counter[dt.hour] += 1
        return dict(sorted(counter.items()))

    def _daily_usage(self, turns: list[dict]) -> dict[str, int]:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        counter: dict[str, int] = defaultdict(int)
        for t in turns:
            dt = t.get("_dt")
            if dt:
                counter[days[dt.weekday()]] += 1
        return dict(counter)

    def _topic_frequency(self, turns: list[dict]) -> list[tuple[str, int]]:
        """Extract crude topic indicators from user messages."""
        words = Counter()
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "i", "you",
                       "we", "it", "to", "of", "in", "for", "on", "with", "that",
                       "this", "and", "or", "but", "not", "my", "me", "do", "can",
                       "what", "how", "why", "when", "be", "have", "has", "had",
                       "will", "would", "could", "should", "just", "so", "if",
                       "about", "from", "at", "by", "as", "up", "out", "there"}
        for t in turns:
            user_text = t.get("user", "")
            for word in user_text.lower().split():
                word = word.strip(".,!?\"'()[]{}:;")
                if len(word) > 2 and word not in stop_words:
                    words[word] += 1
        return words.most_common(10)

    def _session_lengths(self, turns: list[dict]) -> dict[str, int]:
        sessions: dict[str, int] = defaultdict(int)
        for t in turns:
            sessions[t.get("_session", "")] += 1
        if not sessions:
            return {"avg": 0, "max": 0, "min": 0}
        lengths = list(sessions.values())
        return {
            "avg": sum(lengths) // len(lengths),
            "max": max(lengths),
            "min": min(lengths),
        }

    def _activity_trend(self, turns: list[dict]) -> str:
        """Is activity trending up, down, or stable?"""
        if len(turns) < 10:
            return "insufficient_data"
        mid = len(turns) // 2
        first_half = len(turns[:mid])
        second_half = len(turns[mid:])
        ratio = second_half / first_half if first_half > 0 else 1.0
        if ratio > 1.3:
            return "increasing"
        elif ratio < 0.7:
            return "decreasing"
        return "stable"

    def _avg_length(self, turns: list[dict]) -> int:
        lengths = [len(t.get("user", "")) for t in turns]
        return sum(lengths) // len(lengths) if lengths else 0
