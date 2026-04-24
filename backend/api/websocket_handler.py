"""WebSocket pipeline runner. Single in-memory session, streams progress events."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import umap
from fastapi import WebSocket, WebSocketDisconnect

from ..parsers import parse_file
from ..pipeline.clusterer import Clusterer
from ..pipeline.embedder import Embedder
from ..pipeline.reducer import Reducer
from ..pipeline.topic_modeler import LLMConfig, TopicResult, run_topic_modeling
from ..search.bm25_index import BM25Index
from ..search.hybrid_search import HybridSearch
from ..search.semantic_search import SemanticSearch

STAGES = {
    "parsing": 10,
    "embedding": 30,
    "umap": 50,
    "clustering": 65,
    "outliers": 75,
    "ctfidf": 85,
    "labeling": 95,
    "done": 100,
}


@dataclass
class PipelineState:
    """In-memory state for the single active session."""

    uploaded_files: dict[str, dict] = field(default_factory=dict)  # file_id -> {path, filename}
    chunks: list[dict] = field(default_factory=list)
    chunks_by_id: dict[str, dict] = field(default_factory=dict)
    embeddings: np.ndarray | None = None
    chunk_ids: list[str] = field(default_factory=list)
    topic_result: TopicResult | None = None
    bm25: BM25Index | None = None
    semantic: SemanticSearch | None = None
    hybrid: HybridSearch | None = None
    embedder: Embedder = field(default_factory=Embedder)

    def reset(self) -> None:
        self.uploaded_files.clear()
        self.chunks = []
        self.chunks_by_id.clear()
        self.embeddings = None
        self.chunk_ids = []
        self.topic_result = None
        self.bm25 = None
        self.semantic = None
        self.hybrid = None


STATE = PipelineState()


async def _send(ws: WebSocket, stage: str, message: str = "", data: dict[str, Any] | None = None) -> None:
    payload = {"stage": stage, "pct": STAGES.get(stage, 0), "message": message}
    if data is not None:
        payload["data"] = data
    await ws.send_text(json.dumps(payload))


async def _run_pipeline(ws: WebSocket, min_cluster_size: int, llm_config: LLMConfig | None = None) -> None:
    loop = asyncio.get_running_loop()

    await _send(ws, "parsing", "Parsing uploaded files")
    chunks: list[dict] = []
    for f in STATE.uploaded_files.values():
        try:
            parsed = await loop.run_in_executor(None, parse_file, f["path"], f["filename"])
            chunks.extend(parsed)
        except Exception as exc:  # noqa: BLE001
            await _send(ws, "parsing", f"Skipped {f['filename']}: {exc}")
    if not chunks:
        await _send(ws, "error", "No chunks produced from uploads")
        return
    STATE.chunks = chunks
    STATE.chunks_by_id = {c["id"]: c for c in chunks}
    STATE.chunk_ids = [c["id"] for c in chunks]

    await _send(ws, "embedding", f"Embedding {len(chunks)} chunks")
    texts = [c["text"] for c in chunks]
    embeddings = await loop.run_in_executor(None, STATE.embedder.embed, texts)
    STATE.embeddings = embeddings

    await _send(ws, "umap", "Reducing dimensions")
    reducer = Reducer()
    reduced_5d, reduced_2d = await loop.run_in_executor(
        None, reducer.fit_transform_both, embeddings
    )

    await _send(ws, "clustering", "Clustering (HDBSCAN)")
    clusterer = Clusterer()
    labels = await loop.run_in_executor(None, clusterer.fit, reduced_5d, min_cluster_size)

    await _send(ws, "outliers", "Reducing outliers + c-TF-IDF + labels")

    await _send(ws, "ctfidf", "Building topic representations")

    await _send(ws, "labeling", "Generating LLM labels")
    # BERTopic requires a UMAP instance it can call fit_transform on internally.
    bertopic_umap = umap.UMAP(
        n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )
    topic_result: TopicResult = await loop.run_in_executor(
        None,
        run_topic_modeling,
        chunks,
        embeddings,
        labels,
        min_cluster_size,
        bertopic_umap,
        clusterer.model_,
        reduced_2d,
        llm_config or LLMConfig(),
        STATE.embedder.model,
    )
    STATE.topic_result = topic_result

    # Build search indices.
    bm25 = BM25Index()
    bm25.build(chunks)
    STATE.bm25 = bm25

    STATE.semantic = SemanticSearch(STATE.embedder, embeddings, STATE.chunk_ids)

    topic_labels = {t.id: t.label for t in topic_result.topics}
    topic_centroids = {t.id: np.asarray(t.centroid_embedding) for t in topic_result.topics}
    STATE.hybrid = HybridSearch(
        bm25=bm25,
        semantic=STATE.semantic,
        chunks_by_id=STATE.chunks_by_id,
        chunk_topic_map=topic_result.chunk_topic_map,
        topic_labels=topic_labels,
        topic_centroids=topic_centroids,
    )

    done_payload = {
        "topics": [
            {
                "id": t.id,
                "label": t.label,
                "keywords": t.keywords,
                "doc_count": t.doc_count,
                "chunk_ids": t.chunk_ids,
            }
            for t in topic_result.topics
        ],
        "umap_points": topic_result.umap_2d_coords,
        "chunk_map": {
            cid: {
                "heading": STATE.chunks_by_id[cid].get("heading", ""),
                "source": STATE.chunks_by_id[cid].get("source", ""),
                "topic_id": tid,
            }
            for cid, tid in topic_result.chunk_topic_map.items()
        },
    }
    await _send(ws, "done", "Pipeline complete", data=done_payload)


async def pipeline_websocket(ws: WebSocket) -> None:
    """Accept messages of the form {action: 'run', min_cluster_size: int}."""
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send(ws, "error", "Invalid JSON")
                continue
            if msg.get("action") == "run":
                min_cs = int(msg.get("min_cluster_size", 5))
                raw_llm = msg.get("llm_config")
                llm_cfg: LLMConfig | None = None
                if isinstance(raw_llm, dict):
                    llm_cfg = LLMConfig(
                        provider=raw_llm.get("provider", "local"),
                        model_name=raw_llm.get("model_name", "google/flan-t5-base"),
                        base_url=raw_llm.get("base_url", "http://localhost:11434"),
                    )
                try:
                    await _run_pipeline(ws, min_cs, llm_cfg)
                except Exception as exc:  # noqa: BLE001
                    await _send(ws, "error", f"Pipeline failed: {exc}")
            else:
                await _send(ws, "error", f"Unknown action: {msg.get('action')}")
    except WebSocketDisconnect:
        return
