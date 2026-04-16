"""Coordination package — multi-agent orchestration for Sutra (v0.4 Anahata)."""

from coordination.coordinator import Coordinator
from coordination.decomposer import decompose_task
from coordination.shared_context import (
    CoordinationPlan,
    SharedContext,
    SubTask,
    TaskStatus,
)
from coordination.worker import Worker

__all__ = [
    "Coordinator",
    "Worker",
    "SharedContext",
    "CoordinationPlan",
    "SubTask",
    "TaskStatus",
    "decompose_task",
]
