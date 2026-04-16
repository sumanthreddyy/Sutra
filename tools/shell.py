"""Shell command execution tool.

Restricted to a configurable allowlist of read-only commands.
"""

import asyncio
import logging
import shlex
from typing import Any

from tools.base import Tool

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED = ["ls", "find", "grep", "cat", "head", "tail", "wc", "git", "stat", "echo", "pwd", "date"]


class ShellTool(Tool):
    def __init__(self, allowed_commands: list[str] | None = None):
        self._allowed = set(allowed_commands or DEFAULT_ALLOWED)

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return f"Execute a shell command. Allowed commands: {', '.join(sorted(self._allowed))}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                }
            },
            "required": ["command"],
        }

    async def _run(self, command: str, **kwargs: Any) -> str:
        # Parse and validate the command
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return f"Error: Invalid command syntax: {e}"

        if not parts:
            return "Error: Empty command"

        base_cmd = parts[0]

        # Block dangerous patterns
        dangerous = [";", "&&", "||", "|", ">", ">>", "<", "$(", "`", "rm ", "del ", "sudo"]
        for d in dangerous:
            if d in command:
                return f"Error: '{d}' is not allowed in commands"

        if base_cmd not in self._allowed:
            return f"Error: '{base_cmd}' is not in the allowed commands list: {', '.join(sorted(self._allowed))}"

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            output = ""
            if stdout:
                output += stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\n[stderr] " + stderr.decode("utf-8", errors="replace")

            # Truncate large outputs
            if len(output) > 10000:
                output = output[:10000] + "\n... (output truncated at 10KB)"

            return output.strip() or "(no output)"

        except asyncio.TimeoutError:
            return "Error: Command timed out after 30 seconds"
        except Exception as e:
            return f"Error: {e}"
