"""Consolidation lock — prevents concurrent dream runs.

- Lock file whose mtime IS lastConsolidatedAt
- Body contains holder's PID
- Stale lock detection via PID liveness + timeout
"""

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

LOCK_FILE = ".consolidate-lock"
HOLDER_STALE_MS = 60 * 60 * 1000  # 1 hour


class ConsolidationLock:
    def __init__(self, memory_dir: str):
        self.lock_path = Path(memory_dir) / LOCK_FILE

    def read_last_consolidated_at(self) -> float:
        """mtime of lock file = lastConsolidatedAt. 0 if absent."""
        try:
            return self.lock_path.stat().st_mtime * 1000  # ms
        except FileNotFoundError:
            return 0

    def _is_pid_alive(self, pid: int) -> bool:
        """Check if a process is running."""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def try_acquire(self) -> float | None:
        """Acquire lock. Returns prior mtime (for rollback) or None if blocked."""
        mtime_ms: float | None = None
        holder_pid: int | None = None

        try:
            stat = self.lock_path.stat()
            mtime_ms = stat.st_mtime * 1000
            raw = self.lock_path.read_text().strip()
            holder_pid = int(raw) if raw.isdigit() else None
        except FileNotFoundError:
            pass  # No prior lock

        if mtime_ms is not None:
            age_ms = time.time() * 1000 - mtime_ms
            if age_ms < HOLDER_STALE_MS:
                if holder_pid is not None and self._is_pid_alive(holder_pid):
                    logger.debug(f"Lock held by live PID {holder_pid} ({age_ms/1000:.0f}s ago)")
                    return None
                # Dead PID — reclaim

        # Write our PID
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_text(str(os.getpid()))

        # Verify we won the race
        try:
            verify = self.lock_path.read_text().strip()
            if int(verify) != os.getpid():
                return None
        except (FileNotFoundError, ValueError):
            return None

        return mtime_ms if mtime_ms is not None else 0

    def rollback(self, prior_mtime: float) -> None:
        """Rewind mtime to pre-acquire state after a failed dream."""
        try:
            if prior_mtime == 0:
                self.lock_path.unlink(missing_ok=True)
                return
            self.lock_path.write_text("")
            t = prior_mtime / 1000  # seconds
            os.utime(self.lock_path, (t, t))
        except OSError as e:
            logger.warning(f"Rollback failed: {e}")

    def record_consolidation(self) -> None:
        """Stamp current time as last consolidation."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_text(str(os.getpid()))
