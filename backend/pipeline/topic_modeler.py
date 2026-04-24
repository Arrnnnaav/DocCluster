"""Topic modeling over pre-computed embeddings / reduction / clusters.

Pipeline (in order):
    c-TF-IDF (BERTopic default)  ->  KeyBERTInspired rerank  ->  LLM label

BERTopic runs c-TF-IDF + KeyBERTInspired inside ``fit``. LLM labels are
produced as a post-processing step here — this keeps us compatible with
transformers versions that no longer register the ``text2text-generation``
pipeline and lets us reuse the same code path for local seq2seq and for
Ollama's HTTP API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_LLM_MODEL = "google/flan-t5-base"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass
class LLMConfig:
    """Config for the label-generation LLM."""

    provider: Literal["local", "ollama"] = "local"
    model_name: str = DEFAULT_LLM_MODEL
    base_url: str = DEFAULT_OLLAMA_URL


@dataclass
class Topic:
    id: int
    label: str
    keywords: list[str]
    chunk_ids: list[str]
    centroid_embedding: list[float]
    doc_count: int


@dataclass
class TopicResult:
    topics: list[Topic]
    chunk_topic_map: dict[str, int]
    umap_2d_coords: list[dict] = field(default_factory=list)


class _Seq2SeqGenerator:
    """HF-pipeline-ish wrapper for encoder-decoder models (flan-t5 etc.)."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 40) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _format_prompt(keywords: list[str], docs: list[str]) -> str:
    sample = "\n".join(d[:200] for d in docs[:3] if d)
    kw = ", ".join(keywords[:8])
    return (
        "I have a topic containing these documents:\n"
        f"{sample}\n"
        f"Keywords: {kw}.\n"
        "Give this topic a short descriptive label."
    )


def _local_labels(
    topics_keywords: dict[int, list[str]],
    topics_docs: dict[int, list[str]],
    llm_config: LLMConfig,
) -> dict[int, str]:
    tokenizer = AutoTokenizer.from_pretrained(llm_config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(llm_config.model_name)
    generator = _Seq2SeqGenerator(model, tokenizer)
    labels: dict[int, str] = {}
    for tid, keywords in topics_keywords.items():
        prompt = _format_prompt(keywords, topics_docs.get(tid, []))
        try:
            labels[tid] = generator(prompt)
        except Exception:
            labels[tid] = ""
    return labels


def _ollama_labels(
    topics_keywords: dict[int, list[str]],
    topics_docs: dict[int, list[str]],
    llm_config: LLMConfig,
) -> dict[int, str]:
    try:
        import httpx
    except ImportError:
        return {tid: "" for tid in topics_keywords}
    base = llm_config.base_url.rstrip("/")
    labels: dict[int, str] = {}
    with httpx.Client(timeout=60.0) as client:
        for tid, keywords in topics_keywords.items():
            prompt = _format_prompt(keywords, topics_docs.get(tid, []))
            try:
                r = client.post(
                    f"{base}/api/generate",
                    json={
                        "model": llm_config.model_name,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                r.raise_for_status()
                labels[tid] = (r.json().get("response") or "").strip()
            except Exception:
                labels[tid] = ""
    return labels


def _generate_llm_labels(
    topics_keywords: dict[int, list[str]],
    topics_docs: dict[int, list[str]],
    llm_config: LLMConfig,
) -> dict[int, str]:
    if llm_config.provider == "ollama":
        return _ollama_labels(topics_keywords, topics_docs, llm_config)
    return _local_labels(topics_keywords, topics_docs, llm_config)


def run_topic_modeling(
    chunks: list[dict],
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,  # noqa: ARG001 — kept for API stability
    min_cluster_size: int,
    umap_model,
    hdbscan_model,
    umap_2d_coords: np.ndarray | None = None,
    llm_config: LLMConfig | None = None,
    embedding_model=None,
) -> TopicResult:
    """Fit BERTopic on pre-computed artifacts and return a TopicResult."""
    if llm_config is None:
        llm_config = LLMConfig()

    documents = [c["text"] for c in chunks]
    chunk_ids = [c["id"] for c in chunks]
    sources = [c.get("source", "") for c in chunks]

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model={"KeyBERT": KeyBERTInspired()}
        if embedding_model is not None
        else None,
        min_topic_size=min_cluster_size,
        calculate_probabilities=False,
        verbose=False,
    )
    topic_model.fit(documents, embeddings=embeddings)

    initial_topics = list(topic_model.topics_)
    try:
        new_labels = topic_model.reduce_outliers(
            documents,
            initial_topics,
            strategy="embeddings",
            embeddings=embeddings,
        )
        if embedding_model is not None:
            topic_model.update_topics(
                documents,
                topics=new_labels,
                representation_model={"KeyBERT": KeyBERTInspired()},
            )
        else:
            topic_model.update_topics(documents, topics=new_labels)
    except Exception:
        new_labels = initial_topics
    final_labels = np.asarray(new_labels)

    chunk_topic_map = {cid: int(t) for cid, t in zip(chunk_ids, final_labels)}

    # Gather per-topic members + keywords before calling the LLM.
    topic_ids = sorted({int(t) for t in final_labels} - {-1})
    topics_keywords: dict[int, list[str]] = {}
    topics_docs: dict[int, list[str]] = {}
    topic_members: dict[int, list[int]] = {}
    for tid in topic_ids:
        member_idx = [i for i, t in enumerate(final_labels) if int(t) == tid]
        topic_members[tid] = member_idx
        keyword_tuples = topic_model.get_topic(tid) or []
        topics_keywords[tid] = [
            kw for kw, _ in keyword_tuples if isinstance(kw, str)
        ]
        topics_docs[tid] = [documents[i] for i in member_idx[:3]]

    try:
        llm_labels = _generate_llm_labels(topics_keywords, topics_docs, llm_config)
    except Exception:
        llm_labels = {}

    topics: list[Topic] = []
    for tid in topic_ids:
        member_idx = topic_members[tid]
        if not member_idx:
            continue
        centroid = embeddings[member_idx].mean(axis=0)
        keywords = topics_keywords[tid]
        label = llm_labels.get(tid) or (
            ", ".join(keywords[:3]) if keywords else f"Topic {tid}"
        )
        topics.append(
            Topic(
                id=tid,
                label=label,
                keywords=keywords,
                chunk_ids=[chunk_ids[i] for i in member_idx],
                centroid_embedding=centroid.astype(float).tolist(),
                doc_count=len(member_idx),
            )
        )

    coords: list[dict] = []
    if umap_2d_coords is not None:
        for i, (cid, src) in enumerate(zip(chunk_ids, sources)):
            x, y = umap_2d_coords[i]
            coords.append(
                {
                    "chunk_id": cid,
                    "x": float(x),
                    "y": float(y),
                    "topic_id": int(final_labels[i]),
                    "source": src,
                }
            )

    return TopicResult(
        topics=topics,
        chunk_topic_map=chunk_topic_map,
        umap_2d_coords=coords,
    )
