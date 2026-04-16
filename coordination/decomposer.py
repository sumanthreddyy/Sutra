"""Task decomposer — breaks complex requests into parallelizable sub-tasks using an LLM."""

import json
import logging
from typing import Any

from coordination.shared_context import CoordinationPlan, SubTask
from providers.base import Message

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """You are a task planner. Break this complex task into smaller, concrete sub-tasks.

Rules:
- Each sub-task should be completable by a single agent with access to tools (shell, files, web search, memory)
- Identify which sub-tasks can run in PARALLEL (no dependencies) vs SEQUENTIAL (depends on another)
- Keep sub-tasks focused — one clear action each
- Use 2-6 sub-tasks. Don't over-decompose simple tasks.

Respond with ONLY valid JSON (no markdown, no explanation):
{
  "subtasks": [
    {
      "id": "t1",
      "description": "Clear description of what to do",
      "dependencies": []
    },
    {
      "id": "t2",
      "description": "This depends on t1's result",
      "dependencies": ["t1"]
    }
  ]
}

Task to decompose:
"""


async def decompose_task(task: str, router: Any) -> CoordinationPlan:
    """Use LLM to decompose a complex task into sub-tasks."""
    messages = [
        Message(role="system", content=DECOMPOSE_PROMPT),
        Message(role="user", content=task),
    ]

    response = await router.complete(
        messages=messages,
        task_type="classification",  # Use cheap/local model for planning
        temperature=0.3,
    )

    # Parse JSON response
    try:
        raw = response.content.strip()
        # Handle markdown code fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning(f"Failed to parse decomposition, using single task: {e}")
        # Fallback: treat the whole thing as one task
        return CoordinationPlan(
            original_task=task,
            subtasks=[SubTask(id="t1", description=task)],
        )

    subtasks = []
    for item in data.get("subtasks", []):
        subtasks.append(SubTask(
            id=item.get("id", f"t{len(subtasks) + 1}"),
            description=item.get("description", ""),
            dependencies=item.get("dependencies", []),
        ))

    if not subtasks:
        subtasks = [SubTask(id="t1", description=task)]

    plan = CoordinationPlan(original_task=task, subtasks=subtasks)
    logger.info(f"Decomposed task into {len(subtasks)} sub-tasks")
    return plan
