"""Hybrid search: BM25 candidate retrieval + semantic reranking."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bm25_index import BM25Index
from .semantic_search import SemanticSearch


@dataclass
class HybridResultItem:
    chunk_id: str
    chunk_text: str
    heading: str
    source: str
    topic_id: int
    topic_label: str
    bm25_score: float
    semantic_score: float
    final_score: float


@dataclass
class HybridSearchResult:
    results: list[HybridResultItem]
    matched_topic_ids: list[int] = field(default_factory=list)


class HybridSearch:
    """Combine BM25 + semantic reranking over a shared chunk set.

    Pipeline:
      1. BM25 returns top 30 candidates.
      2. Candidates re-scored by cosine similarity against query embedding.
      3. final_score = 0.3 * bm25_norm + 0.7 * semantic (normalized to [0,1]).
      4. Matched topic ids returned in order of topic-centroid similarity.
    """

    BM25_WEIGHT = 0.3
    SEMANTIC_WEIGHT = 0.7
    CANDIDATE_POOL = 30

    def __init__(
        self,
        bm25: BM25Index,
        semantic: SemanticSearch,
        chunks_by_id: dict[str, dict],
        chunk_topic_map: dict[str, int],
        topic_labels: dict[int, str],
        topic_centroids: dict[int, list[float] | np.ndarray] | None = None,
    ) -> None:
        self.bm25 = bm25
        self.semantic = semantic
        self.chunks_by_id = chunks_by_id
        self.chunk_topic_map = chunk_topic_map
        self.topic_labels = topic_labels
        self.topic_centroids = topic_centroids or {}

    def search(self, query: str, top_k: int = 10) -> HybridSearchResult:
        candidates = self.bm25.search(query, top_k=self.CANDIDATE_POOL)
        if not candidates:
            # Fall back to pure semantic search when BM25 returns nothing.
            semantic_hits = self.semantic.search(query, top_k=top_k)
            items = [
                self._build_item(
                    h["chunk_id"], bm25=0.0, semantic=h["score"], final=h["score"]
                )
                for h in semantic_hits
            ]
            return HybridSearchResult(
                results=items,
                matched_topic_ids=self._rank_topics(query, items),
            )

        bm25_by_id = {c["chunk_id"]: c["score"] for c in candidates}
        max_bm25 = max(bm25_by_id.values()) or 1.0

        query_vec = self.semantic._embed_query(query)
        scored: list[HybridResultItem] = []
        for cid, bm25_score in bm25_by_id.items():
            idx = self._chunk_index(cid)
            if idx is None:
                continue
            sem_score = float(self.semantic.chunk_embeddings[idx] @ query_vec)
            final = (
                self.BM25_WEIGHT * (bm25_score / max_bm25)
                + self.SEMANTIC_WEIGHT * sem_score
            )
            scored.append(
                self._build_item(cid, bm25=bm25_score, semantic=sem_score, final=final)
            )

        scored.sort(key=lambda r: r.final_score, reverse=True)
        results = scored[:top_k]
        return HybridSearchResult(
            results=results,
            matched_topic_ids=self._rank_topics(query, results, query_vec=query_vec),
        )

    def _chunk_index(self, chunk_id: str) -> int | None:
        try:
            return self.semantic.chunk_ids.index(chunk_id)
        except ValueError:
            return None

    def _build_item(
        self, chunk_id: str, bm25: float, semantic: float, final: float
    ) -> HybridResultItem:
        chunk = self.chunks_by_id.get(chunk_id, {})
        topic_id = self.chunk_topic_map.get(chunk_id, -1)
        return HybridResultItem(
            chunk_id=chunk_id,
            chunk_text=chunk.get("text", ""),
            heading=chunk.get("heading", ""),
            source=chunk.get("source", ""),
            topic_id=topic_id,
            topic_label=self.topic_labels.get(topic_id, ""),
            bm25_score=float(bm25),
            semantic_score=float(semantic),
            final_score=float(final),
        )

    def _rank_topics(
        self,
        query: str,
        results: list[HybridResultItem],
        query_vec: np.ndarray | None = None,
    ) -> list[int]:
        """Rank topics either by centroid similarity or by hit frequency."""
        if self.topic_centroids:
            vec = query_vec if query_vec is not None else self.semantic._embed_query(query)
            return self.semantic.search_by_topic(vec, self.topic_centroids)
        seen: list[int] = []
        for r in results:
            if r.topic_id != -1 and r.topic_id not in seen:
                seen.append(r.topic_id)
        return seen
