"""Coordinator — orchestrates multi-agent task execution.

The coordinator:
1. Receives a complex task
2. Decomposes it into sub-tasks (via LLM)
3. Executes sub-tasks in dependency order, parallelizing where possible
4. Collects results and synthesizes a final answer
"""

import asyncio
import logging
import time
from typing import Any

from coordination.decomposer import decompose_task
from coordination.shared_context import (
    CoordinationPlan,
    SharedContext,
    SubTask,
    TaskStatus,
)
from coordination.worker import Worker
from providers.base import Message
from tools.base import ToolRegistry

logger = logging.getLogger(__name__)

MAX_PARALLEL_WORKERS = 4

SYNTHESIZE_PROMPT = """You are synthesizing results from multiple sub-agents that worked on parts of a larger task.

Original task: {task}

Sub-task results:
{results}

Provide a unified, coherent response that combines all results. Be concise but complete."""


class Coordinator:
    """Orchestrates decomposition, parallel execution, and synthesis."""

    def __init__(self, router: Any, tools: ToolRegistry):
        self.router = router
        self.tools = tools
        self.shared_ctx = SharedContext()
        self._current_plan: CoordinationPlan | None = None

    async def coordinate(
        self,
        task: str,
        on_progress: Any = None,
    ) -> str:
        """Decompose, execute, and synthesize a complex task.

        Args:
            task: The complex task description
            on_progress: Optional callback(plan) called after each sub-task completes
        """
        start = time.time()
        await self.shared_ctx.clear()

        # Step 1: Decompose
        logger.info(f"Coordinating: {task[:80]}...")
        plan = await decompose_task(task, self.router)
        self._current_plan = plan

        if len(plan.subtasks) == 1:
            # Not worth coordinating — single task
            worker = Worker(self.router, self.tools, self.shared_ctx)
            result = await worker.execute(plan.subtasks[0])
            return result.result

        # Step 2: Execute in dependency waves
        logger.info(f"Executing {len(plan.subtasks)} sub-tasks...")
        while not plan.is_complete:
            ready = plan.get_ready_tasks()
            if not ready:
                # Check for deadlock — tasks pending but nothing ready
                pending = [
                    t for t in plan.subtasks if t.status == TaskStatus.PENDING
                ]
                if pending:
                    logger.error("Deadlock: pending tasks with unmet dependencies")
                    for t in pending:
                        t.status = TaskStatus.FAILED
                        t.error = "Deadlock: unmet dependencies"
                break

            # Run ready tasks in parallel (up to limit)
            batch = ready[:MAX_PARALLEL_WORKERS]
            workers = [
                Worker(self.router, self.tools, self.shared_ctx)
                for _ in batch
            ]
            tasks = [
                w.execute(subtask)
                for w, subtask in zip(workers, batch)
            ]
            await asyncio.gather(*tasks)

            if on_progress:
                on_progress(plan)

        # Step 3: Synthesize
        result = await self._synthesize(plan)

        elapsed = time.time() - start
        logger.info(
            f"Coordination complete: {plan.progress} in {elapsed:.1f}s"
        )

        self._current_plan = None
        return result

    async def _synthesize(self, plan: CoordinationPlan) -> str:
        """Combine sub-task results into a final response."""
        results_text = ""
        for t in plan.subtasks:
            status = "completed" if t.status == TaskStatus.COMPLETED else "FAILED"
            content = t.result if t.status == TaskStatus.COMPLETED else t.error
            results_text += f"\n[{t.id}] ({status}) {t.description}\n{content}\n"

        prompt = SYNTHESIZE_PROMPT.format(
            task=plan.original_task,
            results=results_text,
        )

        response = await self.router.complete(
            messages=[
                Message(role="system", content=prompt),
                Message(role="user", content="Synthesize the results into a final answer."),
            ],
            task_type="default",
        )
        return response.content

    def get_status(self) -> dict:
        """Get current coordination status."""
        if not self._current_plan:
            return {"active": False}
        plan = self._current_plan
        return {
            "active": True,
            "task": plan.original_task[:80],
            "progress": plan.progress,
            "subtasks": [
                {
                    "id": t.id,
                    "description": t.description[:60],
                    "status": t.status.value,
                    "duration": f"{t.duration:.1f}s" if t.duration else "-",
                }
                for t in plan.subtasks
            ],
        }
