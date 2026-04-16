"""Memory type taxonomy — 4-type system.

Memories are constrained to four types capturing context NOT derivable
from the current project state:
  - user: who the user is, their preferences, role, knowledge
  - feedback: guidance on how to work (corrections + confirmations)
  - project: ongoing work context not in code/git
  - reference: pointers to external resources
"""

from dataclasses import dataclass
from enum import Enum


class MemoryType(str, Enum):
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


@dataclass
class MemoryFile:
    """A single memory file with frontmatter metadata."""
    path: str
    name: str
    description: str
    type: MemoryType
    content: str
    archived: bool = False
    source: str = "extracted"  # extracted | consolidated | manual

    def to_frontmatter(self) -> str:
        name = self.name.replace('"', '\\"')
        desc = self.description.replace('"', '\\"')
        lines = [
            "---",
            f'name: "{name}"',
            f'description: "{desc}"',
            f"type: {self.type.value}",
        ]
        if self.archived:
            lines.append("archived: true")
        if self.source != "extracted":
            lines.append(f"source: {self.source}")
        lines.append("---")
        lines.append("")
        lines.append(self.content)
        return "\n".join(lines)

    @classmethod
    def from_file(cls, path: str, raw_content: str) -> "MemoryFile":
        """Parse a memory file with YAML frontmatter."""
        import yaml
        if not raw_content.startswith("---"):
            # No frontmatter — treat as legacy
            return cls(
                path=path,
                name=path.rsplit("/", 1)[-1].replace(".md", ""),
                description="",
                type=MemoryType.USER,
                content=raw_content.strip(),
            )

        parts = raw_content.split("---", 2)
        if len(parts) < 3:
            return cls(path=path, name="", description="", type=MemoryType.USER, content=raw_content)

        meta = yaml.safe_load(parts[1]) or {}
        body = parts[2].strip()

        mem_type = MemoryType.USER
        raw_type = meta.get("type", "")
        try:
            mem_type = MemoryType(raw_type)
        except ValueError:
            pass

        return cls(
            path=path,
            name=meta.get("name", ""),
            description=meta.get("description", ""),
            type=mem_type,
            content=body,
            archived=bool(meta.get("archived", False)),
            source=meta.get("source", "extracted"),
        )


# What NOT to save — enforced in extraction prompt
WHAT_NOT_TO_SAVE = [
    "Code patterns, conventions, architecture, file paths, or project structure — derivable from reading the project.",
    "Git history, recent changes, or who-changed-what — git log/blame are authoritative.",
    "Debugging solutions or fix recipes — the fix is in the code; the commit message has context.",
    "Ephemeral task details: in-progress work, temporary state, current conversation context.",
]


def build_memory_type_descriptions() -> str:
    """Build the memory type descriptions for inclusion in prompts."""
    return """## Types of memory

<types>
<type>
    <name>user</name>
    <description>Information about the user's role, goals, preferences, and knowledge.
    Great user memories help tailor future behavior to the user specifically.</description>
    <when_to_save>When you learn details about the user's role, preferences, or knowledge</when_to_save>
</type>
<type>
    <name>feedback</name>
    <description>Guidance the user has given about how to approach work — corrections AND confirmations.
    Record from failure AND success to remain coherent. Include *why* so you can judge edge cases.</description>
    <when_to_save>When the user corrects your approach OR confirms a non-obvious approach worked</when_to_save>
</type>
<type>
    <name>project</name>
    <description>Information about ongoing work, goals, bugs, or incidents not derivable from code/git.
    Convert relative dates to absolute. These decay fast — include *why* for future judgment.</description>
    <when_to_save>When you learn who is doing what, why, or by when</when_to_save>
</type>
<type>
    <name>reference</name>
    <description>Pointers to external resources — dashboards, trackers, channels, docs.</description>
    <when_to_save>When you learn about resources in external systems</when_to_save>
</type>
</types>

## What NOT to save
""" + "\n".join(f"- {item}" for item in WHAT_NOT_TO_SAVE)
