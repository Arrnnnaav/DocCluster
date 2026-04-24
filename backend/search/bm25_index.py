"""BM25 keyword index over chunk texts using rank_bm25."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "")]


class BM25Index:
    """BM25Okapi wrapper keyed by chunk id."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []

    def build(self, chunks: list[dict]) -> None:
        """Tokenize chunk texts and build BM25 index."""
        self._chunk_ids = [c["id"] for c in chunks]
        corpus = [_tokenize(c.get("text", "")) for c in chunks]
        if not corpus:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 30) -> list[dict]:
        """Return top_k chunks by BM25 score for `query`."""
        if self._bm25 is None or not self._chunk_ids:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"chunk_id": self._chunk_ids[i], "score": float(scores[i])}
            for i in top_idx
            if scores[i] > 0
        ]
