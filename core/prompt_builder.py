"""Dynamic system prompt builder.

- Static sections (cacheable)
- Dynamic sections (user/session-specific)
- Memory content injected from MEMORY.md
- Tool descriptions appended based on available tools
"""

from datetime import datetime

from memory.memdir import MemoryDir
from memory.types import build_memory_type_descriptions


CORE_IDENTITY = """You are Sutra, a personal AI assistant with persistent memory.
You remember conversations across sessions and learn from every interaction.

You have a file-based memory system that persists across conversations.
When you learn something worth remembering, save it to memory.
When you recall something relevant, use it to inform your response."""


def build_system_prompt(
    memory_dir: MemoryDir,
    user_name: str = "User",
    extra_context: str = "",
    relevant_query: str = "",
) -> str:
    """Build the full system prompt with all sections.

    Args:
        relevant_query: If provided, inject semantically relevant memories for this query.
    """
    sections = [CORE_IDENTITY]

    # Current date/time
    sections.append(f"\nCurrent date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    sections.append(f"User: {user_name}")

    # Memory system instructions
    sections.append(f"""
# Persistent Memory

You have a persistent, file-based memory system at `{memory_dir.base_dir}`.
This directory already exists — write to it directly.

You should build up this memory system over time so that future conversations
have a complete picture of who the user is, how they'd like to collaborate,
what behaviors to avoid or repeat, and the context behind their work.

If the user explicitly asks you to remember something, save it immediately.
If they ask you to forget something, find and remove the relevant entry.

{build_memory_type_descriptions()}

## How to save memories

Write each memory to its own file (e.g., `user_role.md`, `feedback_testing.md`)
using this frontmatter format:

```markdown
---
name: {{memory name}}
description: {{one-line description}}
type: {{user, feedback, project, reference}}
---

{{memory content}}
```

- Organize memory semantically by topic, not chronologically
- Update or remove memories that turn out to be wrong or outdated
- Do not write duplicate memories — check existing ones first

## When to access memories
- When memories seem relevant, or the user references prior-conversation work
- You MUST access memory when the user explicitly asks you to recall or remember
- Memory records can become stale. Verify memory against current state before acting on it.

## Before recommending from memory

A memory that names a specific fact is a claim about what was true *when written*.
Before recommending based on it, verify it's still accurate.""")

    # Load current MEMORY.md index
    index_content = memory_dir.read_index()
    if index_content:
        sections.append(f"\n## MEMORY.md\n\n{index_content}")
    else:
        sections.append(
            "\n## MEMORY.md\n\nYour MEMORY.md is currently empty. "
            "When you save new memories, they will appear here."
        )

    # Inject semantically relevant memories for this conversation turn
    if relevant_query:
        relevant = memory_dir.semantic_search(relevant_query, n_results=5, min_score=0.3)
        if relevant:
            sections.append("\n## Relevant Memories\n\nThese memories were retrieved as potentially relevant to the current conversation:\n")
            for r in relevant:
                sections.append(
                    f"### {r['name']} ({r['type']}) — score: {r['score']}\n"
                    f"*{r['description']}*\n\n"
                    f"{r['content'].split(chr(10), 2)[-1].strip()}\n"
                )

    # Extra context
    if extra_context:
        sections.append(f"\n{extra_context}")

    return "\n".join(sections)
