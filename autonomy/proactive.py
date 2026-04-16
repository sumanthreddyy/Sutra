"""Proactive engine — generates suggestions and summaries without being asked.

This is Sutra's "voice" — the Vishuddha chakra.
Instead of only responding, Sutra can initiate:
- Daily summaries of activity
- Gentle nudges based on inactivity or patterns
- Contextual suggestions based on recent work
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from providers.base import Message

logger = logging.getLogger(__name__)

DAILY_SUMMARY_PROMPT = """You are Sutra, generating a daily summary for your user.

Review today's conversation transcripts and memory changes, then write a brief,
warm summary (3-5 bullet points) covering:
- Key topics discussed
- Memories saved or updated
- Anything left unfinished
- A gentle suggestion for tomorrow

Be concise and personal. You know this user well.

Today's date: {date}

Transcripts from today:
{transcripts}

Current memory index:
{memory_index}
"""

PROACTIVE_PROMPT = """You are Sutra, generating a proactive suggestion for your user.

Context:
- User has been idle for {idle_minutes} minutes
- Current time: {time}
- Last topic discussed: {last_topic}
- Recent memories: {recent_memories}

Generate ONE short, helpful suggestion or reminder. Be warm but not annoying.
Only suggest if you have something genuinely useful. If not, respond with "NONE".
Keep it under 2 sentences."""


class ProactiveEngine:
    """Generates suggestions and summaries proactively."""

    def __init__(self, transcript_dir: str, memory_dir: Any):
        self._transcript_dir = Path(transcript_dir)
        self._memory_dir = memory_dir
        self._last_suggestion_time = 0.0
        self._suggestion_cooldown = 1800  # 30 min between suggestions

    async def generate_daily_summary(self, router: Any) -> str:
        """Generate a daily summary of activity."""
        today = datetime.now().strftime("%Y%m%d")
        transcripts = self._load_today_transcripts(today)

        if not transcripts:
            return "No activity to summarize today."

        memory_index = self._memory_dir.read_index() or "Empty"

        prompt = DAILY_SUMMARY_PROMPT.format(
            date=datetime.now().strftime("%Y-%m-%d"),
            transcripts=transcripts[:8000],  # Cap transcript size
            memory_index=memory_index[:3000],
        )

        response = await router.complete(
            messages=[
                Message(role="system", content=prompt),
                Message(role="user", content="Generate today's summary."),
            ],
            task_type="summarization",
        )
        return response.content

    async def maybe_suggest(
        self,
        router: Any,
        idle_seconds: float,
        last_messages: list[Message],
    ) -> str | None:
        """Generate a proactive suggestion if conditions are met."""
        # Cooldown check
        if (time.time() - self._last_suggestion_time) < self._suggestion_cooldown:
            return None

        # Only suggest after 10+ minutes idle
        if idle_seconds < 600:
            return None

        idle_minutes = int(idle_seconds / 60)
        last_topic = "unknown"
        if last_messages:
            for msg in reversed(last_messages):
                if msg.role == "user":
                    last_topic = msg.content[:100]
                    break

        recent = self._memory_dir.list_memories()[-5:]
        recent_str = ", ".join(m.name for m in recent) if recent else "none"

        prompt = PROACTIVE_PROMPT.format(
            idle_minutes=idle_minutes,
            time=datetime.now().strftime("%H:%M"),
            last_topic=last_topic,
            recent_memories=recent_str,
        )

        response = await router.complete(
            messages=[
                Message(role="system", content=prompt),
                Message(role="user", content="Generate a suggestion if appropriate."),
            ],
            task_type="classification",  # Use cheap model
        )

        text = response.content.strip()
        if text.upper() == "NONE" or not text:
            return None

        self._last_suggestion_time = time.time()
        return text

    def _load_today_transcripts(self, date_prefix: str) -> str:
        """Load transcript files from today."""
        if not self._transcript_dir.exists():
            return ""

        lines = []
        for f in sorted(self._transcript_dir.glob(f"{date_prefix}*.jsonl")):
            try:
                content = f.read_text(encoding="utf-8")
                lines.append(content)
            except Exception:
                pass
        return "\n".join(lines)
