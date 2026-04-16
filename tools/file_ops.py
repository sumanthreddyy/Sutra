"""File operations tool — read, write, list, search files."""

import logging
import os
from pathlib import Path
from typing import Any

from tools.base import Tool

logger = logging.getLogger(__name__)


class FileReadTool(Tool):
    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return "Read the contents of a file. Specify a line range for large files."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "start_line": {"type": "integer", "description": "Start line (1-based, optional)"},
                "end_line": {"type": "integer", "description": "End line (1-based, optional)"},
            },
            "required": ["path"],
        }

    async def _run(self, path: str, start_line: int | None = None, end_line: int | None = None, **kwargs: Any) -> str:
        p = Path(path).resolve()
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"

        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: Cannot read binary file: {path}"

        if start_line or end_line:
            lines = content.split("\n")
            s = (start_line or 1) - 1
            e = end_line or len(lines)
            content = "\n".join(lines[s:e])

        if len(content) > 50000:
            content = content[:50000] + "\n... (truncated at 50KB — use start_line/end_line)"

        return content


class FileWriteTool(Tool):
    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write to"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        }

    async def _run(self, path: str, content: str, **kwargs: Any) -> str:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {path}"


class ListDirTool(Tool):
    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List the contents of a directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list"},
            },
            "required": ["path"],
        }

    async def _run(self, path: str, **kwargs: Any) -> str:
        p = Path(path).resolve()
        if not p.exists():
            return f"Error: Directory not found: {path}"
        if not p.is_dir():
            return f"Error: Not a directory: {path}"

        entries = []
        for item in sorted(p.iterdir()):
            suffix = "/" if item.is_dir() else ""
            size = ""
            if item.is_file():
                size = f" ({item.stat().st_size} bytes)"
            entries.append(f"  {item.name}{suffix}{size}")

        return f"{path}/\n" + "\n".join(entries) if entries else f"{path}/ (empty)"
