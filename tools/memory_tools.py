"""Memory tools — let the LLM directly interact with the memory system."""

import logging
from typing import Any

from memory.memdir import MemoryDir
from memory.types import MemoryFile, MemoryType
from tools.base import Tool

logger = logging.getLogger(__name__)


class MemoryReadTool(Tool):
    def __init__(self, memory_dir: MemoryDir):
        self._mem = memory_dir

    @property
    def name(self) -> str:
        return "memory_read"

    @property
    def description(self) -> str:
        return "Read a specific memory file by filename, or pass 'index' to read MEMORY.md."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Memory filename (e.g., 'user_role.md') or 'index' for MEMORY.md"},
            },
            "required": ["filename"],
        }

    async def _run(self, filename: str, **kwargs: Any) -> str:
        if filename == "index":
            content = self._mem.read_index()
            return content or "MEMORY.md is empty."

        mem = self._mem.read_memory(filename)
        if not mem:
            return f"Memory file not found: {filename}"
        return f"---\nname: {mem.name}\ndescription: {mem.description}\ntype: {mem.type.value}\n---\n\n{mem.content}"


class MemoryWriteTool(Tool):
    def __init__(self, memory_dir: MemoryDir):
        self._mem = memory_dir

    @property
    def name(self) -> str:
        return "memory_write"

    @property
    def description(self) -> str:
        return "Save a memory to the persistent memory system. Types: user, feedback, project, reference."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename like 'user_role.md' or 'feedback_testing.md'"},
                "name": {"type": "string", "description": "Short title for the memory"},
                "description": {"type": "string", "description": "One-line description (used for relevance matching)"},
                "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"], "description": "Memory type"},
                "content": {"type": "string", "description": "Memory content to save"},
            },
            "required": ["filename", "name", "description", "type", "content"],
        }

    async def _run(self, filename: str, name: str, description: str, type: str, content: str, **kwargs: Any) -> str:
        try:
            mem_type = MemoryType(type)
        except ValueError:
            return f"Error: Invalid memory type '{type}'. Use: user, feedback, project, reference"

        memory = MemoryFile(
            path=filename,
            name=name,
            description=description,
            type=mem_type,
            content=content,
        )
        path = self._mem.save_memory(memory)
        return f"Saved memory: {name} -> {path}"


class MemorySearchTool(Tool):
    def __init__(self, memory_dir: MemoryDir):
        self._mem = memory_dir

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return "Search memories by meaning (semantic) or keyword. Returns matching memories ranked by relevance."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term or natural language query"},
                "n_results": {"type": "integer", "description": "Max results to return (default 5)"},
            },
            "required": ["query"],
        }

    async def _run(self, query: str, n_results: int = 5, **kwargs: Any) -> str:
        # Try semantic search first (falls back to keyword internally)
        results = self._mem.semantic_search(query, n_results=n_results)
        if not results:
            return f"No memories matching '{query}'."

        lines = []
        for r in results:
            score_str = f"score: {r['score']}"
            if r.get("semantic_similarity"):
                score_str += f", similarity: {r['semantic_similarity']}"
            lines.append(
                f"- **{r['name']}** ({r['type']}) — {r['description']}\n"
                f"  File: {r['path']} | {score_str}"
            )
        return "\n".join(lines)
