"""Cosine-similarity semantic search over cached chunk embeddings."""
from __future__ import annotations

import numpy as np

from ..pipeline.embedder import Embedder


class SemanticSearch:
    """Dense search using an Embedder + pre-computed chunk embeddings.

    Chunk embeddings are assumed L2-normalized (Embedder.embed returns
    normalized vectors), so cosine similarity reduces to a dot product.
    """

    def __init__(
        self,
        embedder: Embedder,
        chunk_embeddings: np.ndarray,
        chunk_ids: list[str],
    ) -> None:
        self.embedder = embedder
        self.chunk_embeddings = chunk_embeddings.astype(np.float32)
        self.chunk_ids = chunk_ids

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.embedder.embed([query])[0]
        return vec.astype(np.float32)

    def search(self, query: str, top_k: int = 30) -> list[dict]:
        """Cosine-similarity search against all chunks."""
        if not self.chunk_ids or self.chunk_embeddings.size == 0:
            return []
        q = self._embed_query(query)
        scores = self.chunk_embeddings @ q
        top_idx = np.argsort(-scores)[:top_k]
        return [
            {"chunk_id": self.chunk_ids[i], "score": float(scores[i])}
            for i in top_idx
        ]

    def search_by_topic(
        self,
        query_embedding: np.ndarray,
        topic_centroids: dict[int, np.ndarray] | dict[int, list[float]],
    ) -> list[int]:
        """Rank topic ids by cosine similarity between query and centroid."""
        if not topic_centroids:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        qn = q / (np.linalg.norm(q) + 1e-12)
        scored: list[tuple[float, int]] = []
        for tid, centroid in topic_centroids.items():
            c = np.asarray(centroid, dtype=np.float32)
            cn = c / (np.linalg.norm(c) + 1e-12)
            scored.append((float(cn @ qn), tid))
        scored.sort(reverse=True)
        return [tid for _, tid in scored]
