"""UMAP reducer producing both 5-D (for clustering) and 2-D (for plotting) views."""
from __future__ import annotations

import numpy as np
import umap


class Reducer:
    """Thin wrapper around two UMAP configurations."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def fit_transform_5d(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce to 5 dimensions for HDBSCAN clustering."""
        reducer = umap.UMAP(
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=self.random_state,
        )
        return reducer.fit_transform(embeddings)

    def fit_transform_2d(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce to 2 dimensions for visualization."""
        n = len(embeddings)
        # Higher n_neighbors creates stronger global structure → clusters group visually
        n_neighbors = min(50, max(10, n // 3))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.05,
            spread=1.0,
            metric="cosine",
            random_state=self.random_state,
            low_memory=False,
        )
        coords = reducer.fit_transform(embeddings)
        assert len(coords) == n, f"UMAP output {len(coords)} != input {n}"
        return coords

    def fit_transform_both(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run both reductions. Returns (reduced_5d, reduced_2d)."""
        return self.fit_transform_5d(embeddings), self.fit_transform_2d(embeddings)
