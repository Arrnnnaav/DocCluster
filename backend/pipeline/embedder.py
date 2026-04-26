"""Sentence embedder wrapping `thenlper/gte-small` with a lazy-loaded model."""
from __future__ import annotations

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "thenlper/gte-small"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    """Lazy-loads the GTE-small model on first call, caches it on the instance."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=_DEVICE)
            print(f"[embedder] Loaded {self.model_name} on {_DEVICE}", flush=True)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings for a list of texts."""
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)
