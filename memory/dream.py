"""Dream System — Background memory consolidation engine.

- Three-gate trigger: time + sessions + lock
- Four-phase consolidation: Orient → Gather → Consolidate → Prune
- Runs as a background task with read-only access to project files
- Memory-only write access

v0.7: NON-LOSSY — dreams consolidate but NEVER delete.
  - Originals are archived (frontmatter: archived: true), not deleted
  - Consolidated summaries are a fast-access layer on top of raw data
  - Raw transcripts are always preserved as source of truth
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from memory.lock import ConsolidationLock
from memory.memdir import MemoryDir, ENTRYPOINT_NAME, MAX_INDEX_LINES

logger = logging.getLogger(__name__)


def build_consolidation_prompt(memory_root: str, transcript_dir: str, extra: str = "") -> str:
    """Build the dream consolidation prompt — 4-phase system.

    """
    return f"""# Dream: Memory Consolidation

You are performing a dream — a reflective pass over your memory files.
Synthesize what you've learned recently into durable, well-organized memories
so that future sessions can orient quickly.

Memory directory: `{memory_root}`
This directory already exists — write to it directly.

Session transcripts: `{transcript_dir}` (JSONL files — grep narrowly, don't read whole files)

---

## Phase 1 — Orient

- List the memory directory to see what already exists
- Read `{ENTRYPOINT_NAME}` to understand the current index
- Skim existing topic files so you improve them rather than creating duplicates

## Phase 2 — Gather recent signal

Look for new information worth persisting. Sources in priority order:

1. **Session transcripts** — scan for user preferences, corrections, project context
2. **Existing memories that drifted** — facts that contradict what you see now
3. **Patterns across sessions** — recurring themes or preferences

Don't exhaustively read transcripts. Look only for things you suspect matter.

## Phase 3 — Consolidate

For each thing worth remembering, write or update a memory file.
Use frontmatter format:

```markdown
---
name: {{memory name}}
description: {{one-line description}}
type: {{user, feedback, project, reference}}
---

{{memory content}}
```

Focus on:
- Merging new signal into existing topic files rather than creating duplicates
- Converting relative dates ("yesterday", "last week") to absolute dates
- If new info contradicts an old memory, update the EXISTING file — do NOT delete it

CRITICAL: NEVER delete any memory file. If a memory is outdated or superseded:
- Add `archived: true` to its frontmatter instead of deleting
- Create a new consolidated file with the corrected information
- The archived file stays on disk as a historical record

## Phase 4 — Index and archive

Update `{ENTRYPOINT_NAME}` so it stays under {MAX_INDEX_LINES} lines AND under ~25KB.
It's an **index**, not a dump — each entry should be one line under ~150 characters:
`- [Title](file.md) — one-line hook`. Never write memory content directly into it.

- Move stale/superseded entries to `MEMORY_ARCHIVE.md` (create if it doesn't exist)
- Add pointers to newly important or consolidated memories
- Both files are searchable — the index is for quick access, the archive is for completeness

---

Return a brief summary of what you consolidated, updated, or pruned.
If nothing changed (memories are already tight), say so.{f'''

## Additional context

{extra}''' if extra else ''}"""


class DreamEngine:
    """Background memory consolidation — Sutra's "sleep" cycle."""

    def __init__(
        self,
        memory_dir: MemoryDir,
        transcript_dir: str,
        min_hours: int = 24,
        min_sessions: int = 5,
    ):
        self.memory_dir = memory_dir
        self.transcript_dir = transcript_dir
        self.min_hours = min_hours
        self.min_sessions = min_sessions
        self.lock = ConsolidationLock(str(memory_dir.base_dir))
        self._last_scan_at = 0.0
        self._scan_interval_ms = 10 * 60 * 1000  # 10 minutes

    def _count_sessions_since(self, since_ms: float) -> int:
        """Count transcript files modified since the given time."""
        transcript_path = Path(self.transcript_dir)
        if not transcript_path.exists():
            return 0

        count = 0
        for f in transcript_path.glob("*.jsonl"):
            try:
                if f.stat().st_mtime * 1000 > since_ms:
                    count += 1
            except OSError:
                continue
        return count

    def should_dream(self) -> bool:
        """Three-gate check: time + sessions + lock availability.

        Gate order (cheapest first):
        1. Time: hours since lastConsolidatedAt >= min_hours
        2. Sessions: transcript count since last consolidation >= min_sessions
        3. Lock: no other process mid-consolidation
        """
        # Gate 1: Time
        last_at = self.lock.read_last_consolidated_at()
        hours_since = (time.time() * 1000 - last_at) / 3_600_000
        if hours_since < self.min_hours:
            return False

        # Scan throttle
        since_scan_ms = time.time() * 1000 - self._last_scan_at
        if since_scan_ms < self._scan_interval_ms:
            return False
        self._last_scan_at = time.time() * 1000

        # Gate 2: Sessions
        session_count = self._count_sessions_since(last_at)
        if session_count < self.min_sessions:
            logger.debug(f"Dream skip — {session_count} sessions, need {self.min_sessions}")
            return False

        return True

    async def run_dream(self, router) -> dict:
        """Execute the dream consolidation.

        Returns a summary of what changed.
        """
        from providers.base import Message

        # Gate 3: Lock
        prior_mtime = self.lock.try_acquire()
        if prior_mtime is None:
            logger.debug("Dream skip — lock held by another process")
            return {"status": "skipped", "reason": "locked"}

        last_at = self.lock.read_last_consolidated_at()
        session_count = self._count_sessions_since(last_at)

        logger.info(
            f"Dream firing — {((time.time() * 1000 - last_at) / 3_600_000):.1f}h "
            f"since last, {session_count} sessions to review"
        )

        try:
            # Build the consolidation prompt
            extra = f"Sessions since last consolidation: {session_count}"
            prompt = build_consolidation_prompt(
                str(self.memory_dir.base_dir),
                self.transcript_dir,
                extra,
            )

            # Run consolidation via LLM (using the "dream" task route — typically local/free)
            response = await router.complete(
                messages=[Message(role="user", content=prompt)],
                task_type="dream",
                temperature=0.3,
                max_tokens=4096,
            )

            # Record successful consolidation
            self.lock.record_consolidation()

            logger.info("Dream completed successfully")
            return {
                "status": "completed",
                "summary": response.content,
                "usage": response.usage,
            }

        except Exception as e:
            logger.error(f"Dream failed: {e}")
            self.lock.rollback(prior_mtime)
            return {"status": "failed", "error": str(e)}

    def get_status(self) -> dict:
        """Get dream system status."""
        last_at = self.lock.read_last_consolidated_at()
        hours_since = (time.time() * 1000 - last_at) / 3_600_000 if last_at > 0 else -1
        session_count = self._count_sessions_since(last_at)

        return {
            "last_consolidation": datetime.fromtimestamp(last_at / 1000).isoformat() if last_at > 0 else "never",
            "hours_since": round(hours_since, 1),
            "sessions_since": session_count,
            "min_hours": self.min_hours,
            "min_sessions": self.min_sessions,
            "ready": self.should_dream(),
        }
