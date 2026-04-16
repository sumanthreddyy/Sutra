"""Worker agent — a lightweight sub-agent that executes a single sub-task.

Workers share the same tool registry as the main agent but have their own
conversation context scoped to their specific sub-task.
"""

import json
import logging
import time
from typing import Any

from coordination.shared_context import SharedContext, SubTask, TaskStatus
from providers.base import LLMResponse, Message
from tools.base import ToolRegistry

logger = logging.getLogger(__name__)

MAX_WORKER_ROUNDS = 8  # Tool-call rounds per worker

WORKER_SYSTEM_PROMPT = """You are a focused sub-agent working on ONE specific task.
You have access to tools. Complete your task and respond with a clear, concise result.
Do NOT go beyond your assigned task. Stay focused.

If you need information from the shared context, it will be provided.
When done, summarize what you accomplished in 2-3 sentences."""


class Worker:
    """A sub-agent that executes a single sub-task."""

    def __init__(
        self,
        router: Any,
        tools: ToolRegistry,
        shared_ctx: SharedContext,
    ):
        self.router = router
        self.tools = tools
        self.shared_ctx = shared_ctx

    async def execute(self, subtask: SubTask) -> SubTask:
        """Execute a sub-task and update its status."""
        subtask.status = TaskStatus.RUNNING
        subtask.assigned_at = time.time()

        logger.info(f"Worker starting: [{subtask.id}] {subtask.description}")

        try:
            result = await self._run(subtask)
            subtask.status = TaskStatus.COMPLETED
            subtask.result = result
            # Store result in shared context for dependent tasks
            await self.shared_ctx.set(f"result:{subtask.id}", result)
        except Exception as e:
            subtask.status = TaskStatus.FAILED
            subtask.error = str(e)
            logger.error(f"Worker [{subtask.id}] failed: {e}")

        subtask.completed_at = time.time()
        logger.info(
            f"Worker [{subtask.id}] {subtask.status.value} "
            f"({subtask.duration:.1f}s)"
        )
        return subtask

    async def _run(self, subtask: SubTask) -> str:
        """Execute the sub-task with tool-call loop."""
        # Gather dependency results from shared context
        dep_context = ""
        if subtask.dependencies:
            dep_parts = []
            for dep_id in subtask.dependencies:
                dep_result = await self.shared_ctx.get(f"result:{dep_id}", "")
                if dep_result:
                    dep_parts.append(f"Result from {dep_id}: {dep_result}")
            if dep_parts:
                dep_context = "\n\nContext from previous tasks:\n" + "\n".join(dep_parts)

        messages = [
            Message(role="system", content=WORKER_SYSTEM_PROMPT),
            Message(
                role="user",
                content=f"Task: {subtask.description}{dep_context}",
            ),
        ]

        tool_defs = self.tools.get_tool_defs()

        response: LLMResponse | None = None
        for _ in range(MAX_WORKER_ROUNDS):
            response = await self.router.complete(
                messages=messages,
                task_type="default",
                tools=tool_defs if tool_defs else None,
            )

            if not response.tool_calls:
                break

            # Handle tool calls
            messages.append(Message(
                role="assistant",
                content=response.content or "",
                tool_calls=response.tool_calls,
            ))

            for tc in response.tool_calls:
                tool_name = tc.get("name", "")
                tool_id = tc.get("id", "")
                raw_args = tc.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}

                result = await self.tools.execute(tool_name, raw_args)
                messages.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=tool_id,
                    name=tool_name,
                ))

        return response.content if response else "No response generated."
