"""Scheduler — cron-style task scheduling for Sutra.

Runs background tasks on intervals:
- Periodic checks (file changes, git status, etc.)
- Daily summaries
- Custom user-defined schedules
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class ScheduleType(str, Enum):
    INTERVAL = "interval"    # Every N seconds
    DAILY = "daily"          # Once per day at HH:MM
    ON_EVENT = "on_event"    # Triggered by an event


@dataclass
class ScheduledTask:
    """A task that runs on a schedule."""
    name: str
    description: str
    schedule_type: ScheduleType
    handler: Callable[..., Coroutine]
    interval_seconds: int = 0       # For INTERVAL type
    daily_time: str = "09:00"       # For DAILY type (HH:MM)
    event_name: str = ""            # For ON_EVENT type
    enabled: bool = True
    last_run: float = 0.0
    run_count: int = 0

    @property
    def is_due(self) -> bool:
        if not self.enabled:
            return False
        if self.schedule_type == ScheduleType.INTERVAL:
            return (time.time() - self.last_run) >= self.interval_seconds
        if self.schedule_type == ScheduleType.DAILY:
            from datetime import datetime
            now = datetime.now()
            h, m = self.daily_time.split(":")
            target = now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
            was_today = self.last_run > target.timestamp()
            return now.timestamp() >= target.timestamp() and not was_today
        return False


class Scheduler:
    """Background task scheduler."""

    def __init__(self):
        self._tasks: dict[str, ScheduledTask] = {}
        self._running = False
        self._loop_task: asyncio.Task | None = None

    def register(self, task: ScheduledTask) -> None:
        self._tasks[task.name] = task
        logger.info(f"Registered scheduled task: {task.name} ({task.schedule_type.value})")

    def unregister(self, name: str) -> None:
        self._tasks.pop(name, None)

    def enable(self, name: str) -> None:
        if name in self._tasks:
            self._tasks[name].enabled = True

    def disable(self, name: str) -> None:
        if name in self._tasks:
            self._tasks[name].enabled = False

    async def start(self) -> None:
        """Start the scheduler background loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop — checks every 30 seconds for due tasks."""
        while self._running:
            for task in list(self._tasks.values()):
                if task.is_due:
                    asyncio.create_task(self._execute(task))
            await asyncio.sleep(30)

    async def _execute(self, task: ScheduledTask) -> None:
        """Execute a scheduled task."""
        try:
            logger.info(f"Executing scheduled task: {task.name}")
            task.last_run = time.time()
            task.run_count += 1
            await task.handler()
        except Exception as e:
            logger.error(f"Scheduled task {task.name} failed: {e}")

    def fire_event(self, event_name: str) -> None:
        """Fire an event — triggers all ON_EVENT tasks matching this event."""
        for task in self._tasks.values():
            if (
                task.schedule_type == ScheduleType.ON_EVENT
                and task.event_name == event_name
                and task.enabled
            ):
                asyncio.create_task(self._execute(task))

    def get_status(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "type": t.schedule_type.value,
                "enabled": t.enabled,
                "run_count": t.run_count,
                "last_run": time.strftime("%H:%M:%S", time.localtime(t.last_run)) if t.last_run else "never",
            }
            for t in self._tasks.values()
        ]
