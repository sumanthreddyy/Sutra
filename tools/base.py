"""Tool base class and registry.

Tools give Sutra the ability to act on the world:
- Read/write files
- Execute shell commands
- Search the web
- Read/write memories
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from providers.base import ToolDef

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Base class for all Sutra tools."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        ...

    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool and return a string result."""
        try:
            return await self._run(**kwargs)
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}")
            return f"Error: {e}"

    @abstractmethod
    async def _run(self, **kwargs: Any) -> str: ...

    def to_tool_def(self) -> ToolDef:
        """Convert to LLM-compatible tool definition."""
        return ToolDef(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_tool_defs(self) -> list[ToolDef]:
        return [t.to_tool_def() for t in self._tools.values()]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"
        return await tool.execute(**arguments)
