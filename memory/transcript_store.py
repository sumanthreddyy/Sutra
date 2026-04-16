"""Transcript Store — append-only raw conversation storage + search.

Inspired by MemPalace's "never summarize the source" philosophy and
Memvid's append-only frame model.

Layer 0 in Sutra's memory hierarchy:
  L0: Raw transcripts (this file) — append-only, immutable, source of truth
  L1: Extracted memories (.md files) — structured facts pulled from L0
  L2: Consolidated summaries — dream engine merges L1 into higher-level knowledge
  L3: MEMORY.md index — quick-access pointers to L1 + L2

Nothing in L0 is ever deleted or modified. Search spans all layers.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Chunk size for indexing raw transcripts (in conversation turns)
CHUNK_SIZE = 6  # ~3 exchanges (user + assistant pairs)


@dataclass
class TranscriptChunk:
    """An immutable chunk of conversation — the atomic unit of raw memory."""
    id: str                     # transcript_file:chunk_index
    session_id: str             # which conversation session
    timestamp: float            # when this chunk was recorded
    turns: list[dict]           # [{role, content}, ...]
    text: str = ""              # flattened text for embedding
    metadata: dict = field(default_factory=dict)

    def flatten(self) -> str:
        """Flatten turns into a single searchable string."""
        lines = []
        for t in self.turns:
            role = t.get("role", "?")
            content = t.get("content", "")
            if isinstance(content, str) and content.strip():
                lines.append(f"[{role}]: {content}")
        return "\n".join(lines)


class TranscriptStore:
    """Append-only raw transcript storage with chunk-based indexing.

    Raw transcripts are NEVER modified or deleted.
    They are chunked and embedded for semantic search alongside memory files.
    """

    def __init__(self, transcript_dir: str, vector_store: Any | None = None):
        self.transcript_dir = Path(transcript_dir)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = vector_store
        self._indexed_chunks: set[str] = set()  # track what's already embedded

    def _load_indexed_set(self) -> set[str]:
        """Load the set of already-indexed chunk IDs from a tracking file."""
        tracker = self.transcript_dir / ".indexed_chunks"
        if tracker.exists():
            try:
                return set(tracker.read_text(encoding="utf-8").strip().split("\n"))
            except Exception:
                return set()
        return set()

    def _save_indexed_set(self) -> None:
        """Persist the indexed chunk IDs."""
        tracker = self.transcript_dir / ".indexed_chunks"
        tracker.write_text("\n".join(sorted(self._indexed_chunks)), encoding="utf-8")

    def get_all_transcripts(self) -> list[Path]:
        """List all transcript files, sorted by modification time."""
        files = list(self.transcript_dir.glob("*.jsonl"))
        files.sort(key=lambda f: f.stat().st_mtime)
        return files

    def chunk_transcript(self, filepath: Path) -> list[TranscriptChunk]:
        """Read a JSONL transcript and break it into overlapping chunks."""
        chunks = []
        turns = []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        turns.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to read transcript {filepath}: {e}")
            return []

        if not turns:
            return []

        session_id = filepath.stem
        file_mtime = filepath.stat().st_mtime

        # Chunk with 50% overlap for better retrieval
        step = max(1, CHUNK_SIZE // 2)
        for i in range(0, len(turns), step):
            chunk_turns = turns[i:i + CHUNK_SIZE]
            if not chunk_turns:
                break

            chunk = TranscriptChunk(
                id=f"{session_id}:{i}",
                session_id=session_id,
                timestamp=file_mtime,
                turns=chunk_turns,
                metadata={
                    "source": "transcript",
                    "file": filepath.name,
                    "chunk_index": i,
                    "turn_count": len(chunk_turns),
                },
            )
            chunk.text = chunk.flatten()
            chunks.append(chunk)

        return chunks

    def index_transcripts(self, force: bool = False) -> int:
        """Index all unindexed transcript chunks into the vector store.

        Returns the number of newly indexed chunks.
        """
        if not self.vector_store:
            logger.warning("No vector store — skipping transcript indexing")
            return 0

        self._indexed_chunks = self._load_indexed_set()
        count = 0

        for filepath in self.get_all_transcripts():
            chunks = self.chunk_transcript(filepath)
            for chunk in chunks:
                if not force and chunk.id in self._indexed_chunks:
                    continue
                if not chunk.text.strip():
                    continue

                try:
                    self._embed_chunk(chunk)
                    self._indexed_chunks.add(chunk.id)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to embed chunk {chunk.id}: {e}")

        self._save_indexed_set()
        logger.info(f"Indexed {count} new transcript chunks")
        return count

    def _embed_chunk(self, chunk: TranscriptChunk) -> None:
        """Embed a single transcript chunk into the vector store."""
        collection = self.vector_store._get_collection()
        chunk_id = f"transcript:{chunk.id}"

        metadata = {
            "name": f"Conversation {chunk.session_id}",
            "description": chunk.text[:120] + "..." if len(chunk.text) > 120 else chunk.text,
            "type": "transcript",
            "path": chunk_id,
            "source": "transcript",
            "session_id": chunk.session_id,
            "embedded_at": chunk.timestamp,
            "access_count": 0,
        }

        collection.upsert(
            ids=[chunk_id],
            documents=[chunk.text],
            metadatas=[metadata],
        )

    def search_raw(self, query: str, n_results: int = 5) -> list[dict]:
        """Search raw transcripts directly via the vector store.

        Returns results tagged with source='transcript' so the caller
        knows these are raw verbatim matches, not consolidated memories.
        """
        if not self.vector_store:
            return []

        collection = self.vector_store._get_collection()
        if collection.count() == 0:
            return []

        # Search only transcript chunks via metadata filter
        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, collection.count()),
                where={"source": "transcript"},
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # If metadata filter fails (no transcript chunks yet), return empty
            return []

        if not results["ids"] or not results["ids"][0]:
            return []

        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"][0] else {}
            distance = results["distances"][0][i] if results["distances"][0] else 1.0
            document = results["documents"][0][i] if results["documents"][0] else ""

            similarity = max(0.0, 1.0 - distance)

            hits.append({
                "path": doc_id,
                "name": meta.get("name", ""),
                "description": meta.get("description", ""),
                "type": "transcript",
                "source": "transcript",
                "session_id": meta.get("session_id", ""),
                "score": round(similarity, 3),
                "semantic_similarity": round(similarity, 3),
                "content": document,
            })

        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits

    def get_stats(self) -> dict:
        """Get transcript store stats."""
        self._indexed_chunks = self._load_indexed_set()
        transcripts = self.get_all_transcripts()
        total_turns = 0
        for f in transcripts:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    total_turns += sum(1 for line in fh if line.strip())
            except Exception:
                pass

        return {
            "transcript_files": len(transcripts),
            "total_turns": total_turns,
            "indexed_chunks": len(self._indexed_chunks),
        }
