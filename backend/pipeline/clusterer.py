"""HDBSCAN clusterer + BERTopic outlier reduction."""
from __future__ import annotations

import hdbscan
import numpy as np


class Clusterer:
    """HDBSCAN wrapper. Holds the last fitted label array."""

    def __init__(self) -> None:
        self.labels_: np.ndarray | None = None
        self.model_: hdbscan.HDBSCAN | None = None

    def fit(
        self,
        reduced_embeddings_5d: np.ndarray,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
    ) -> np.ndarray:
        """Fit HDBSCAN on 5-D UMAP embeddings and return cluster labels."""
        self.model_ = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples if min_samples is not None else max(1, min_cluster_size // 3),
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        self.labels_ = self.model_.fit_predict(reduced_embeddings_5d)
        return self.labels_

    def reduce_outliers(
        self,
        topic_model,
        documents: list[str],
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Reassign -1 outliers using BERTopic's embedding-based reduction."""
        if self.labels_ is None:
            raise RuntimeError("Call fit() before reduce_outliers().")
        new_topics = topic_model.reduce_outliers(
            documents,
            self.labels_.tolist(),
            strategy="embeddings",
            embeddings=embeddings,
        )
        self.labels_ = np.asarray(new_topics)
        return self.labels_

    @property
    def cluster_count(self) -> int:
        """Number of unique non-outlier clusters in the last fit."""
        if self.labels_ is None:
            return 0
        unique = set(int(x) for x in self.labels_)
        unique.discard(-1)
        return len(unique)
