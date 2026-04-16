"""Vector store — ChromaDB-backed semantic memory search.

Wraps ChromaDB to provide:
- Embedding + persistence of memory files
- Semantic search by meaning (not just keywords)
- Relevance scoring: semantic similarity + recency + access frequency
- Supports local (Ollama) and cloud (OpenAI) embedding models
"""

import logging
import time
from pathlib import Path
from typing import Any

from memory.types import MemoryFile

logger = logging.getLogger(__name__)

# Scoring weights for hybrid relevance
W_SEMANTIC = 0.60
W_RECENCY = 0.25
W_FREQUENCY = 0.15

# Recency half-life in days (memories lose half their recency score after this many days)
RECENCY_HALF_LIFE_DAYS = 7.0


def _recency_score(timestamp: float) -> float:
    """Exponential decay based on age. Returns 0.0–1.0."""
    age_days = (time.time() - timestamp) / 86400.0
    import math
    return math.exp(-0.693 * age_days / RECENCY_HALF_LIFE_DAYS)


def _frequency_score(access_count: int, max_count: int) -> float:
    """Normalize access count to 0.0–1.0."""
    if max_count <= 0:
        return 0.0
    return min(access_count / max_count, 1.0)


class VectorStore:
    """ChromaDB-backed vector store for semantic memory search."""

    def __init__(
        self,
        persist_dir: str = "data/vectors",
        collection_name: str = "sutra_memories",
        embedding_provider: str = "default",
        ollama_model: str = "nomic-embed-text",
        openai_api_key: str = "",
    ):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._ollama_model = ollama_model
        self._openai_api_key = openai_api_key
        self._collection = None
        self._client = None

    def _get_collection(self):
        """Lazy-init ChromaDB collection."""
        if self._collection is not None:
            return self._collection

        import chromadb

        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._persist_dir)

        # Pick embedding function
        embedding_fn = None
        if self._embedding_provider == "ollama":
            embedding_fn = self._make_ollama_ef()
        elif self._embedding_provider == "openai" and self._openai_api_key:
            embedding_fn = self._make_openai_ef()
        # else: use ChromaDB's default (all-MiniLM-L6-v2 via sentence-transformers)

        kwargs: dict[str, Any] = {"name": self._collection_name}
        if embedding_fn:
            kwargs["embedding_function"] = embedding_fn

        self._collection = self._client.get_or_create_collection(
            **kwargs,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Vector store ready: {self._collection_name} "
            f"({self._collection.count()} existing embeddings, "
            f"provider={self._embedding_provider})"
        )
        return self._collection

    def _make_ollama_ef(self):
        """Create Ollama embedding function."""
        from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
        return OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name=self._ollama_model,
        )

    def _make_openai_ef(self):
        """Create OpenAI embedding function."""
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        return OpenAIEmbeddingFunction(
            api_key=self._openai_api_key,
            model_name="text-embedding-3-small",
        )

    def embed_memory(self, memory: MemoryFile) -> None:
        """Embed a single memory file into the vector store."""
        collection = self._get_collection()

        # Build the text to embed: name + description + content
        embed_text = f"{memory.name}\n{memory.description}\n\n{memory.content}"

        metadata = {
            "name": memory.name,
            "description": memory.description,
            "type": memory.type.value,
            "path": memory.path,
            "embedded_at": time.time(),
            "access_count": 0,
        }

        # Upsert — update if exists, insert if new
        collection.upsert(
            ids=[memory.path],
            documents=[embed_text],
            metadatas=[metadata],
        )
        logger.debug(f"Embedded memory: {memory.path}")

    def remove_memory(self, path: str) -> None:
        """Remove a memory from the vector store."""
        collection = self._get_collection()
        try:
            collection.delete(ids=[path])
            logger.debug(f"Removed from vector store: {path}")
        except Exception as e:
            logger.warning(f"Failed to remove {path} from vectors: {e}")

    def search(
        self,
        query: str,
        n_results: int = 5,
        min_score: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Semantic search with hybrid relevance scoring.

        Returns list of dicts with: path, name, description, type, score, content
        """
        collection = self._get_collection()

        if collection.count() == 0:
            return []

        # Clamp n_results to collection size
        n = min(n_results, collection.count())

        results = collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        # Find max access count for normalization
        all_counts = [
            m.get("access_count", 0) for m in (results["metadatas"][0] or [])
        ]
        max_count = max(all_counts) if all_counts else 1

        scored = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"][0] else {}
            distance = results["distances"][0][i] if results["distances"][0] else 1.0
            document = results["documents"][0][i] if results["documents"][0] else ""

            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1.0 = identical, 0.0 = opposite
            semantic_sim = max(0.0, 1.0 - distance)

            recency = _recency_score(meta.get("embedded_at", time.time()))
            freq = _frequency_score(meta.get("access_count", 0), max_count)

            hybrid_score = (
                W_SEMANTIC * semantic_sim
                + W_RECENCY * recency
                + W_FREQUENCY * freq
            )

            if hybrid_score < min_score:
                continue

            scored.append({
                "path": meta.get("path", doc_id),
                "name": meta.get("name", ""),
                "description": meta.get("description", ""),
                "type": meta.get("type", ""),
                "score": round(hybrid_score, 3),
                "semantic_similarity": round(semantic_sim, 3),
                "content": document,
            })

        # Sort by hybrid score descending
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def bump_access(self, path: str) -> None:
        """Increment access count for a memory (called when retrieved)."""
        collection = self._get_collection()
        try:
            result = collection.get(ids=[path], include=["metadatas"])
            if result["ids"] and result["metadatas"]:
                meta = result["metadatas"][0]
                meta["access_count"] = meta.get("access_count", 0) + 1
                collection.update(ids=[path], metadatas=[meta])
        except Exception as e:
            logger.warning(f"Failed to bump access for {path}: {e}")

    def reindex_all(self, memories: list[MemoryFile]) -> int:
        """Re-embed all memories. Returns count embedded."""
        count = 0
        for mem in memories:
            try:
                self.embed_memory(mem)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to embed {mem.path}: {e}")
        logger.info(f"Reindexed {count}/{len(memories)} memories")
        return count

    def get_stats(self) -> dict[str, Any]:
        """Get vector store stats."""
        try:
            collection = self._get_collection()
            return {
                "total_embeddings": collection.count(),
                "persist_dir": self._persist_dir,
                "embedding_provider": self._embedding_provider,
            }
        except Exception:
            return {"total_embeddings": 0, "error": "Vector store not initialized"}
