"""Goal tracker — set objectives and track progress across sessions.

Goals persist as memory files with special frontmatter.
Sutra can check progress, remind about deadlines, and celebrate completions.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from memory.memdir import MemoryDir
from memory.types import MemoryFile, MemoryType

logger = logging.getLogger(__name__)

GOALS_PREFIX = "goal_"


@dataclass
class Goal:
    id: str
    title: str
    description: str
    deadline: str  # ISO date or "" for no deadline
    status: str  # "active", "completed", "abandoned"
    progress: int  # 0-100
    created_at: str
    milestones: list[str]

    def to_memory(self) -> MemoryFile:
        milestone_text = "\n".join(f"- {m}" for m in self.milestones) if self.milestones else "No milestones set."
        content = f"""## {self.title}

{self.description}

**Status:** {self.status}
**Progress:** {self.progress}%
**Deadline:** {self.deadline or 'None'}
**Created:** {self.created_at}

### Milestones
{milestone_text}
"""
        return MemoryFile(
            path=f"{GOALS_PREFIX}{self.id}.md",
            name=self.title,
            description=f"Goal ({self.status}, {self.progress}%): {self.description[:60]}",
            type=MemoryType.PROJECT,
            content=content,
        )

    @classmethod
    def from_memory(cls, mem: MemoryFile) -> "Goal":
        """Parse a goal from a memory file (best-effort)."""
        content = mem.content
        lines = content.split("\n")

        goal_id = mem.path.replace(GOALS_PREFIX, "").replace(".md", "")
        title = mem.name
        description = ""
        status = "active"
        progress = 0
        deadline = ""
        created_at = ""
        milestones = []

        in_milestones = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("**Status:**"):
                status = stripped.split("**Status:**")[1].strip()
            elif stripped.startswith("**Progress:**"):
                try:
                    progress = int(stripped.split("**Progress:**")[1].strip().replace("%", ""))
                except ValueError:
                    pass
            elif stripped.startswith("**Deadline:**"):
                deadline = stripped.split("**Deadline:**")[1].strip()
                if deadline == "None":
                    deadline = ""
            elif stripped.startswith("**Created:**"):
                created_at = stripped.split("**Created:**")[1].strip()
            elif stripped == "### Milestones":
                in_milestones = True
            elif in_milestones and stripped.startswith("- "):
                milestones.append(stripped[2:])
            elif not stripped.startswith("#") and not stripped.startswith("**") and stripped and not in_milestones:
                description += stripped + " "

        return cls(
            id=goal_id,
            title=title,
            description=description.strip() or mem.description,
            deadline=deadline,
            status=status,
            progress=progress,
            created_at=created_at or datetime.now().isoformat()[:10],
            milestones=milestones,
        )


class GoalTracker:
    """Track goals across sessions via the memory system."""

    def __init__(self, memory_dir: MemoryDir):
        self._mem = memory_dir

    def create_goal(
        self,
        title: str,
        description: str = "",
        deadline: str = "",
        milestones: list[str] | None = None,
    ) -> Goal:
        goal_id = title.lower().replace(" ", "_")[:30]
        goal = Goal(
            id=goal_id,
            title=title,
            description=description,
            deadline=deadline,
            status="active",
            progress=0,
            created_at=datetime.now().isoformat()[:10],
            milestones=milestones or [],
        )
        self._mem.save_memory(goal.to_memory())
        logger.info(f"Created goal: {title}")
        return goal

    def update_progress(self, goal_id: str, progress: int, note: str = "") -> Goal | None:
        goal = self._get_goal(goal_id)
        if not goal:
            return None
        goal.progress = min(100, max(0, progress))
        if progress >= 100:
            goal.status = "completed"
        if note:
            goal.milestones.append(f"[{datetime.now().isoformat()[:10]}] {note}")
        self._mem.save_memory(goal.to_memory())
        return goal

    def abandon_goal(self, goal_id: str, reason: str = "") -> Goal | None:
        goal = self._get_goal(goal_id)
        if not goal:
            return None
        goal.status = "abandoned"
        if reason:
            goal.milestones.append(f"[{datetime.now().isoformat()[:10]}] Abandoned: {reason}")
        self._mem.save_memory(goal.to_memory())
        return goal

    def list_goals(self, status: str = "") -> list[Goal]:
        goals = []
        for mem in self._mem.list_memories():
            if mem.path.startswith(GOALS_PREFIX):
                goal = Goal.from_memory(mem)
                if not status or goal.status == status:
                    goals.append(goal)
        return goals

    def get_active_summary(self) -> str:
        """Get a summary of active goals for prompt injection."""
        active = self.list_goals(status="active")
        if not active:
            return ""
        lines = ["## Active Goals"]
        for g in active:
            deadline_str = f" (due: {g.deadline})" if g.deadline else ""
            lines.append(f"- **{g.title}** — {g.progress}%{deadline_str}")
        return "\n".join(lines)

    def _get_goal(self, goal_id: str) -> Goal | None:
        filename = f"{GOALS_PREFIX}{goal_id}.md"
        mem = self._mem.read_memory(filename)
        if not mem:
            return None
        return Goal.from_memory(mem)
