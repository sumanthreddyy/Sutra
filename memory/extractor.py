"""Auto-extract memories from conversations.

- Runs as a background pass after each conversation turn
- Analyzes recent messages for memory-worthy information
- Only fires when the main agent didn't already write memories
- Uses the 4-type taxonomy to categorize extractions
"""

import json
import logging
from pathlib import Path

from memory.memdir import MemoryDir
from memory.types import MemoryFile, MemoryType, build_memory_type_descriptions

logger = logging.getLogger(__name__)


def build_extraction_prompt(
    new_message_count: int,
    existing_memories: str,
    memory_dir: str,
) -> str:
    """Build the extraction prompt for analyzing recent conversation."""
    manifest = ""
    if existing_memories:
        manifest = f"""

## Existing memory files

{existing_memories}

Check this list before writing — update an existing file rather than creating a duplicate."""

    return f"""You are the memory extraction subagent. Analyze the most recent ~{new_message_count} messages
and extract any information worth persisting to memory.

Memory directory: `{memory_dir}`

You should output a JSON array of memories to save. Each memory should have:
- "filename": a descriptive filename like "user_role.md" or "feedback_testing.md"
- "name": short title
- "description": one-line description (used for relevance matching in future)
- "type": one of "user", "feedback", "project", "reference"
- "content": the memory content (for feedback/project types, include Why: and How to apply: lines)

{build_memory_type_descriptions()}
{manifest}

If the user explicitly asked you to remember something, include it.
If there's nothing worth saving, return an empty array: []

IMPORTANT: Only output valid JSON. No markdown, no explanation. Just the array."""


class MemoryExtractor:
    """Extracts memories from conversation turns automatically."""

    def __init__(self, memory_dir: MemoryDir, min_new_messages: int = 3):
        self.memory_dir = memory_dir
        self.min_new_messages = min_new_messages
        self._messages_since_last_extract = 0

    def should_extract(self) -> bool:
        """Check if we have enough new messages to warrant extraction."""
        return self._messages_since_last_extract >= self.min_new_messages

    def tick(self) -> None:
        """Called after each message exchange."""
        self._messages_since_last_extract += 1

    async def extract(self, recent_messages: list, router) -> list[MemoryFile]:
        """Run memory extraction on recent messages.

        Returns list of memories that were saved.
        """
        from providers.base import Message

        if not self.should_extract():
            return []

        logger.debug(f"Running extraction on {self._messages_since_last_extract} new messages")

        # Build conversation context for extraction
        conversation_text = ""
        for msg in recent_messages:
            role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            conversation_text += f"\n[{role}]: {content}\n"

        existing_manifest = self.memory_dir.get_existing_manifest()

        prompt = build_extraction_prompt(
            new_message_count=len(recent_messages),
            existing_memories=existing_manifest,
            memory_dir=str(self.memory_dir.base_dir),
        )

        # Combine conversation + extraction prompt
        messages = [
            Message(role="system", content="You extract structured memories from conversations."),
            Message(role="user", content=f"Conversation to analyze:\n{conversation_text}\n\n{prompt}"),
        ]

        try:
            response = await router.complete(
                messages=messages,
                task_type="extraction",
                temperature=0.2,
                max_tokens=2048,
            )

            # Parse the JSON response
            content = response.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            memories_data = json.loads(content)
            if not isinstance(memories_data, list):
                return []

            saved = []
            for mem_data in memories_data:
                try:
                    mem_type = MemoryType(mem_data.get("type", "user"))
                except ValueError:
                    mem_type = MemoryType.USER

                memory = MemoryFile(
                    path=mem_data.get("filename", "unknown.md"),
                    name=mem_data.get("name", ""),
                    description=mem_data.get("description", ""),
                    type=mem_type,
                    content=mem_data.get("content", ""),
                )
                self.memory_dir.save_memory(memory)
                saved.append(memory)

            self._messages_since_last_extract = 0
            logger.info(f"Extracted {len(saved)} memories")
            return saved

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Memory extraction parse error: {e}")
            return []
        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")
            return []
