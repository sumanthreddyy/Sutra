"""Event watcher — monitors file system and git for changes.

Fires events into the scheduler when things happen:
- File modified/created/deleted in watched directories
- Git commits, branch changes
- Idle time detection
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class FileChange:
    path: str
    event: str  # "modified", "created", "deleted"
    timestamp: float


class FileWatcher:
    """Polls directories for file changes (no external deps needed)."""

    def __init__(self, watch_dirs: list[str], poll_interval: float = 5.0):
        self._dirs = [Path(d) for d in watch_dirs]
        self._interval = poll_interval
        self._snapshots: dict[str, float] = {}
        self._running = False
        self._callbacks: list[Callable[[FileChange], Any]] = []
        self._task: asyncio.Task | None = None

    def on_change(self, callback: Callable[[FileChange], Any]) -> None:
        self._callbacks.append(callback)

    def _take_snapshot(self) -> dict[str, float]:
        """Get current mtime for all files in watched dirs."""
        snap = {}
        for d in self._dirs:
            if not d.exists():
                continue
            for f in d.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    try:
                        snap[str(f)] = f.stat().st_mtime
                    except OSError:
                        pass
        return snap

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._snapshots = self._take_snapshot()
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"File watcher started ({len(self._dirs)} dirs, {len(self._snapshots)} files)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._interval)
            new_snap = self._take_snapshot()
            changes = self._diff(self._snapshots, new_snap)
            self._snapshots = new_snap
            for change in changes:
                for cb in self._callbacks:
                    try:
                        result = cb(change)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"File watcher callback error: {e}")

    def _diff(self, old: dict[str, float], new: dict[str, float]) -> list[FileChange]:
        changes = []
        now = time.time()
        for path, mtime in new.items():
            if path not in old:
                changes.append(FileChange(path, "created", now))
            elif mtime != old[path]:
                changes.append(FileChange(path, "modified", now))
        for path in old:
            if path not in new:
                changes.append(FileChange(path, "deleted", now))
        return changes


class IdleDetector:
    """Tracks user idle time based on last interaction."""

    def __init__(self):
        self._last_active = time.time()
        self._idle_callbacks: list[tuple[float, Callable]] = []  # (seconds, callback)

    def touch(self) -> None:
        """Call this on every user interaction."""
        self._last_active = time.time()

    @property
    def idle_seconds(self) -> float:
        return time.time() - self._last_active

    def on_idle(self, seconds: float, callback: Callable) -> None:
        self._idle_callbacks.append((seconds, callback))

    async def check(self) -> None:
        """Check and fire idle callbacks."""
        idle = self.idle_seconds
        for threshold, cb in self._idle_callbacks:
            if idle >= threshold:
                try:
                    result = cb()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Idle callback error: {e}")
