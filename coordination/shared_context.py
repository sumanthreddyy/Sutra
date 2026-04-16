"""Shared context — thread-safe state shared between coordinator and sub-agents.

Provides:
- Shared key-value store for passing data between agents
- Task result collection
- Progress tracking
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubTask:
    """A single sub-task assigned to a worker agent."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    assigned_at: float = 0.0
    completed_at: float = 0.0
    dependencies: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if self.completed_at and self.assigned_at:
            return self.completed_at - self.assigned_at
        return 0.0


@dataclass
class CoordinationPlan:
    """A decomposed task plan with sub-tasks and execution order."""
    original_task: str
    subtasks: list[SubTask] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def is_complete(self) -> bool:
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for t in self.subtasks
        )

    @property
    def progress(self) -> str:
        done = sum(1 for t in self.subtasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.subtasks if t.status == TaskStatus.FAILED)
        total = len(self.subtasks)
        return f"{done}/{total} done, {failed} failed"

    def get_ready_tasks(self) -> list[SubTask]:
        """Get tasks whose dependencies are all met."""
        completed_ids = {
            t.id for t in self.subtasks if t.status == TaskStatus.COMPLETED
        }
        ready = []
        for t in self.subtasks:
            if t.status != TaskStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in t.dependencies):
                ready.append(t)
        return ready


class SharedContext:
    """Thread-safe shared state for multi-agent coordination."""

    def __init__(self):
        self._store: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._store[key] = value

    async def get(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._store.get(key, default)

    async def get_all(self) -> dict[str, Any]:
        async with self._lock:
            return dict(self._store)

    async def append(self, key: str, value: Any) -> None:
        """Append value to a list stored at key."""
        async with self._lock:
            if key not in self._store:
                self._store[key] = []
            self._store[key].append(value)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
