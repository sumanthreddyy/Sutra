"""Memory directory manager — MEMORY.md index + topic file management.

- MEMORY.md is an INDEX (not a dump) — one line per memory, ~150 chars max
- MEMORY_ARCHIVE.md holds overflow entries — still searchable, not hot
- Topic files contain the actual memory content with frontmatter
- Index is capped at 200 lines / 25KB
- v0.2: Auto-embeds memories into vector store on save
- v0.7: Non-lossy — archive instead of delete, multi-layer search
"""

import logging
import os
from pathlib import Path
from typing import Any

from .types import MemoryFile, MemoryType

logger = logging.getLogger(__name__)

ENTRYPOINT_NAME = "MEMORY.md"
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25000


class MemoryDir:
    """Manages the persistent file-based memory directory."""

    def __init__(self, base_dir: str, vector_store: Any | None = None):
        self.base_dir = Path(base_dir)
        self.entrypoint = self.base_dir / ENTRYPOINT_NAME
        self.vector_store = vector_store
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure memory directory exists."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def read_index(self) -> str:
        """Read MEMORY.md index, truncating if over limits."""
        if not self.entrypoint.exists():
            return ""

        raw = self.entrypoint.read_text(encoding="utf-8").strip()
        lines = raw.split("\n")
        byte_count = len(raw.encode("utf-8"))

        truncated = False
        if len(lines) > MAX_INDEX_LINES:
            lines = lines[:MAX_INDEX_LINES]
            truncated = True

        content = "\n".join(lines)
        if len(content.encode("utf-8")) > MAX_INDEX_BYTES:
            # Truncate at last newline before byte cap
            encoded = content.encode("utf-8")[:MAX_INDEX_BYTES]
            content = encoded.rsplit(b"\n", 1)[0].decode("utf-8", errors="ignore")
            truncated = True

        if truncated:
            content += (
                f"\n\n> WARNING: {ENTRYPOINT_NAME} exceeded limits "
                f"({len(lines)} lines, {byte_count} bytes). Truncated. "
                "Keep entries to one line under ~150 chars."
            )

        return content

    def write_index(self, content: str) -> None:
        """Write MEMORY.md index."""
        self._ensure_dir()
        self.entrypoint.write_text(content, encoding="utf-8")

    def add_index_entry(self, title: str, filename: str, hook: str) -> None:
        """Add a one-line entry to MEMORY.md index."""
        current = self.read_index()
        entry = f"- [{title}]({filename}) — {hook}"
        if current:
            new_content = current + "\n" + entry
        else:
            new_content = f"# Memory Index\n\n{entry}"
        self.write_index(new_content)

    def remove_index_entry(self, filename: str) -> None:
        """Remove an entry from MEMORY.md by filename."""
        if not self.entrypoint.exists():
            return
        lines = self.entrypoint.read_text(encoding="utf-8").split("\n")
        filtered = [l for l in lines if filename not in l]
        self.write_index("\n".join(filtered))

    def save_memory(self, memory: MemoryFile) -> str:
        """Save a memory to a topic file and update the index."""
        self._ensure_dir()
        filepath = self.base_dir / memory.path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(memory.to_frontmatter(), encoding="utf-8")

        # Update index
        self.add_index_entry(memory.name, memory.path, memory.description)

        # Auto-embed into vector store
        if self.vector_store:
            try:
                self.vector_store.embed_memory(memory)
            except Exception as e:
                logger.warning(f"Failed to embed {memory.path}: {e}")

        logger.info(f"Saved memory: {memory.path}")
        return str(filepath)

    def delete_memory(self, filename: str) -> bool:
        """Archive a memory file instead of deleting. Never lose data.

        Sets archived=true in frontmatter, moves index entry to MEMORY_ARCHIVE.md.
        The file stays on disk and in the vector store (tagged as archived).
        """
        return self.archive_memory(filename)

    def archive_memory(self, filename: str) -> bool:
        """Mark a memory as archived — it stays on disk but leaves the active index."""
        filepath = self.base_dir / filename
        if not filepath.exists():
            return False

        # Read, mark archived, write back
        mem = self.read_memory(filename)
        if mem is None:
            return False

        mem.archived = True
        filepath.write_text(mem.to_frontmatter(), encoding="utf-8")

        # Move from active index to archive index
        self.remove_index_entry(filename)
        self._add_archive_entry(mem.name, filename, mem.description)

        logger.info(f"Archived memory: {filename}")
        return True

    def _add_archive_entry(self, title: str, filename: str, hook: str) -> None:
        """Add an entry to MEMORY_ARCHIVE.md."""
        archive_path = self.base_dir / "MEMORY_ARCHIVE.md"
        entry = f"- [{title}]({filename}) — {hook}"
        if archive_path.exists():
            current = archive_path.read_text(encoding="utf-8").strip()
            archive_path.write_text(current + "\n" + entry, encoding="utf-8")
        else:
            archive_path.write_text(f"# Memory Archive\n\n{entry}", encoding="utf-8")

    def hard_delete_memory(self, filename: str) -> bool:
        """Actually delete a memory file from disk. Use only for cleanup, not normal operation."""
        filepath = self.base_dir / filename
        if not filepath.exists():
            return False
        filepath.unlink()
        self.remove_index_entry(filename)
        if self.vector_store:
            try:
                self.vector_store.remove_memory(filename)
            except Exception as e:
                logger.warning(f"Failed to remove {filename} from vectors: {e}")
        logger.info(f"Hard deleted memory: {filename}")
        return True

    def read_memory(self, filename: str) -> MemoryFile | None:
        """Read a specific memory file."""
        filepath = self.base_dir / filename
        if not filepath.exists():
            return None
        raw = filepath.read_text(encoding="utf-8")
        return MemoryFile.from_file(filename, raw)

    def list_memories(self, include_archived: bool = False) -> list[MemoryFile]:
        """List all memory files (excluding MEMORY.md, archive, and lock files)."""
        memories = []
        skip = {ENTRYPOINT_NAME, "MEMORY_ARCHIVE.md"}
        for f in self.base_dir.rglob("*.md"):
            if f.name in skip or f.name.startswith("."):
                continue
            try:
                raw = f.read_text(encoding="utf-8")
                rel = str(f.relative_to(self.base_dir))
                mem = MemoryFile.from_file(rel, raw)
                if not include_archived and mem.archived:
                    continue
                memories.append(mem)
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")
        return memories

    def list_active_memories(self) -> list[MemoryFile]:
        """List only non-archived memories."""
        return self.list_memories(include_archived=False)

    def list_archived_memories(self) -> list[MemoryFile]:
        """List only archived memories."""
        all_mems = self.list_memories(include_archived=True)
        return [m for m in all_mems if m.archived]

    def search_memories(self, query: str) -> list[MemoryFile]:
        """Search memories by content (simple substring match)."""
        query_lower = query.lower()
        results = []
        for mem in self.list_memories():
            if (
                query_lower in mem.content.lower()
                or query_lower in mem.name.lower()
                or query_lower in mem.description.lower()
            ):
                results.append(mem)
        return results

    def semantic_search(self, query: str, n_results: int = 5, min_score: float = 0.3) -> list[dict]:
        """Semantic search using vector store. Falls back to keyword search if unavailable."""
        if self.vector_store:
            results = self.vector_store.search(query, n_results=n_results, min_score=min_score)
            # Bump access count for retrieved memories
            for r in results:
                self.vector_store.bump_access(r["path"])
            return results

        # Fallback: keyword search wrapped in the same format
        keyword_results = self.search_memories(query)[:n_results]
        return [
            {
                "path": m.path,
                "name": m.name,
                "description": m.description,
                "type": m.type.value,
                "source": "memory",
                "score": 0.5,
                "semantic_similarity": 0.0,
                "content": f"{m.name}\n{m.description}\n\n{m.content}",
            }
            for m in keyword_results
        ]

    def unified_search(
        self,
        query: str,
        n_results: int = 5,
        min_score: float = 0.3,
        transcript_store: Any = None,
    ) -> list[dict]:
        """Multi-layer search: memories + raw transcripts, ranked together.

        Inspired by MemPalace's "search everything" philosophy.
        Returns results from both consolidated memories and raw transcripts,
        tagged with source so the caller knows the provenance.
        """
        results = []

        # Layer 1+2: Memory files (extracted + consolidated)
        memory_hits = self.semantic_search(query, n_results=n_results, min_score=min_score)
        for hit in memory_hits:
            hit.setdefault("source", "memory")
        results.extend(memory_hits)

        # Layer 0: Raw transcripts
        if transcript_store:
            transcript_hits = transcript_store.search_raw(query, n_results=n_results)
            results.extend(transcript_hits)

        # Re-rank all results by score, deduplicate by content similarity
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Take top n_results
        return results[:n_results]

    def get_memories_by_type(self, mem_type: MemoryType) -> list[MemoryFile]:
        """Get all memories of a specific type."""
        return [m for m in self.list_memories() if m.type == mem_type]

    def get_existing_manifest(self) -> str:
        """Build a manifest of existing memory files for the extraction prompt."""
        memories = self.list_memories()
        if not memories:
            return ""
        lines = []
        for m in memories:
            lines.append(f"- {m.path} ({m.type.value}): {m.name} — {m.description}")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get memory directory stats."""
        memories = self.list_memories()
        type_counts = {}
        for mem in memories:
            type_counts[mem.type.value] = type_counts.get(mem.type.value, 0) + 1

        index_content = self.read_index()
        return {
            "total_memories": len(memories),
            "by_type": type_counts,
            "index_lines": len(index_content.split("\n")) if index_content else 0,
            "index_bytes": len(index_content.encode("utf-8")) if index_content else 0,
        }
