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

import os

import nltk
import numpy as np
import torch
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Reduce CUDA allocator fragmentation so both gte-small + flan-t5 coexist in 4 GB
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as _nltk_stopwords  # noqa: E402

_STOP: set[str] = set(_nltk_stopwords.words("english"))
_STOP.update([
    # contractions / punctuation tokens
    "'s", "n't", "'re", "'ve", "''", "--",
    # explicit leakages from BERTopic c-TF-IDF
    "the", "to", "of", "in", "a", "an", "is", "it", "its", "be", "as", "at",
    "by", "or", "and", "for", "on", "with", "from", "are", "was", "were",
    "has", "have", "had", "not", "but", "this", "that", "these", "those",
    "their", "they", "them", "then", "than", "so", "do", "did", "does",
    "into", "about", "after", "before", "through", "between", "also",
    "would", "could", "should", "may", "might", "will", "can", "been",
    "more", "over", "such", "any", "all", "each", "both", "he", "she",
    "his", "her", "our", "we", "you", "your", "my", "me", "us", "no", "if",
    # common but low-signal words that slip past NLTK
    "one", "use", "used", "using", "like", "just", "get", "got", "make",
    "made", "know", "see", "new", "good", "high", "low", "well", "way",
    "even", "still", "back", "much", "many", "first", "last", "long",
    "large", "small", "old", "great", "single", "two", "three",
])

DEFAULT_LLM_MODEL = "google/flan-t5-base"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class LocalLLMCache:
    """Lazy-loads and caches a seq2seq model+tokenizer by name across pipeline runs."""

    def __init__(self) -> None:
        self._cache: dict[str, _Seq2SeqGenerator] = {}

    def get_generator(self, model_name: str) -> "_Seq2SeqGenerator":
        if model_name not in self._cache:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            device = _DEVICE
            dtype = torch.float16 if device == "cuda" else torch.float32
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, torch_dtype=dtype
                ).to(device)
                torch.cuda.synchronize() if device == "cuda" else None
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                # Fall back to CPU if VRAM is exhausted
                device = "cpu"
                dtype = torch.float32
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                print(f"[topic_modeler] VRAM full — {model_name} loaded on CPU", flush=True)
            print(f"[topic_modeler] Loaded {model_name} on {device} ({dtype})", flush=True)
            self._cache[model_name] = _Seq2SeqGenerator(model, tokenizer)
        return self._cache[model_name]


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
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _format_prompt(keywords: list[str], _docs: list[str]) -> str:
    clean_kws = [k for k in keywords if k and k != "—"]
    kw = ", ".join(clean_kws[:4]) if clean_kws else "general topic"
    return (
        f"Give a 2-4 word topic name for these keywords: {kw}. "
        f"Examples: 'Object Detection', 'Image Segmentation', "
        f"'Language Modeling'. Only output the topic name."
    )


def _clean_label(raw: str, keywords: list[str]) -> str:
    """Validate LLM output; fall back to title-cased keywords if output is garbage."""
    if not raw:
        return " ".join(w.title() for w in keywords[:2] if w and w != "—")

    raw = raw.strip().rstrip(".").strip()
    words = raw.split()

    _DATASET_NAMES = {
        "coco", "imagenet", "cifar", "mnist", "pascal",
        "voc", "ade20k", "kinetics", "celeba", "lsun",
    }
    lowered = raw.lower()
    bad = (
        len(raw) < 3
        or len(raw) > 60
        or len(words) > 7
        or "is a" in lowered
        or "is the" in lowered
        or "are a type" in lowered
        or "are the" in lowered
        or lowered.strip() in _DATASET_NAMES
    )

    if bad:
        clean_kws = [w for w in keywords if w and w != "—"]
        return " ".join(w.title() for w in clean_kws[:2])

    return raw.title()


def _local_labels(
    topics_keywords: dict[int, list[str]],
    topics_docs: dict[int, list[str]],
    llm_config: LLMConfig,
    llm_cache: LocalLLMCache | None = None,
) -> dict[int, str]:
    if llm_cache is not None:
        generator = llm_cache.get_generator(llm_config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_config.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(llm_config.model_name)
        generator = _Seq2SeqGenerator(model, tokenizer)
    labels: dict[int, str] = {}
    for tid, keywords in topics_keywords.items():
        prompt = _format_prompt(keywords, topics_docs.get(tid, []))
        print(f"[topic_modeler] LLM prompt (topic {tid}): {prompt}", flush=True)
        try:
            raw = generator(prompt)
            labels[tid] = _clean_label(raw, keywords)
            print(f"[topic_modeler] LLM raw={raw!r} -> label={labels[tid]!r}", flush=True)
        except Exception:
            labels[tid] = _clean_label("", keywords)
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
                raw = (r.json().get("response") or "").strip()
                labels[tid] = _clean_label(raw, topics_keywords.get(tid, []))
            except Exception:
                labels[tid] = _clean_label("", topics_keywords.get(tid, []))
    return labels


def _generate_llm_labels(
    topics_keywords: dict[int, list[str]],
    topics_docs: dict[int, list[str]],
    llm_config: LLMConfig,
    llm_cache: LocalLLMCache | None = None,
) -> dict[int, str]:
    if llm_config.provider == "ollama":
        return _ollama_labels(topics_keywords, topics_docs, llm_config)
    return _local_labels(topics_keywords, topics_docs, llm_config, llm_cache)


def extract_clean_keywords(topic_model, top_n: int = 3) -> dict[int, list[str]]:
    """Extract top-n clean keywords per topic from BERTopic, stripping all stopwords."""
    import re

    STOP = {
        "the", "to", "of", "in", "a", "an", "is", "it", "its", "be", "as", "at",
        "by", "or", "and", "for", "on", "with", "from", "are", "was", "were",
        "has", "have", "had", "not", "but", "this", "that", "these", "those",
        "their", "they", "them", "then", "than", "so", "do", "did", "does",
        "into", "about", "after", "before", "through", "between", "also",
        "would", "could", "should", "may", "might", "will", "can", "been",
        "more", "over", "such", "any", "all", "each", "both", "he", "she",
        "his", "her", "our", "we", "you", "your", "my", "me", "us", "no", "if",
        "one", "two", "use", "used", "using", "llm", "llms", "model", "models",
        "chapter", "section", "figure", "page", "book", "example", "see",
        "like", "get", "make", "take", "new", "good", "well", "just", "way",
        "time", "need", "want", "know", "think", "first", "second", "based",
        "given", "however", "therefore", "thus", "since", "where", "when",
        "which", "what", "how", "why", "who", "there", "here", "now", "prompt",
        "output", "input", "text", "data", "result", "results", "value", "type",
        # code / ML terms that bleed into prose topics
        "model", "models", "tokenizer", "template", "import", "class",
        "function", "return", "print", "self", "none", "true", "false",
        "list", "dict", "string", "token", "tokens", "layer", "layers",
        "weight", "weights", "train", "training", "trained", "learning",
        "neural", "network", "networks", "deep", "loss", "batch", "epoch",
        "vector", "matrix", "size", "num", "step", "steps", "number",
        "name", "param", "params", "gradient", "tensor", "tensors",
    }
    STOP.update(_STOP)  # merge with module-level NLTK set

    result: dict[int, list[str]] = {}
    for topic_id, pairs in topic_model.get_topics().items():
        if topic_id == -1:
            continue
        clean: list[str] = []
        for word, score in pairs:
            w = re.sub(r"[^a-zA-Z]", "", word).strip().lower()
            if w and w not in STOP and len(w) >= 4 and score > 0.005:
                clean.append(w)
            if len(clean) == top_n:
                break
        result[topic_id] = clean if clean else ["—"]

    print(f"[topic_modeler] CLEAN keywords: {result}", flush=True)
    return result


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
    llm_cache: LocalLLMCache | None = None,
) -> TopicResult:
    """Fit BERTopic on pre-computed artifacts and return a TopicResult."""
    if llm_config is None:
        llm_config = LLMConfig()

    documents = [c["text"] for c in chunks]
    chunk_ids = [c["id"] for c in chunks]
    sources = [c.get("source", "") for c in chunks]

    # token_pattern enforces letters-only, min 4 chars — blocks "the","to","of","is","we" at source
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{3,}\b",
    )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
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
                vectorizer_model=vectorizer_model,
                representation_model={"KeyBERT": KeyBERTInspired()},
            )
        else:
            topic_model.update_topics(
                documents,
                topics=new_labels,
                vectorizer_model=vectorizer_model,
            )
    except Exception:
        new_labels = initial_topics
    final_labels = np.asarray(new_labels)

    chunk_topic_map = {cid: int(t) for cid, t in zip(chunk_ids, final_labels)}

    # Gather per-topic members + keywords before calling the LLM.
    topic_ids = sorted({int(t) for t in final_labels} - {-1})
    topics_keywords: dict[int, list[str]] = extract_clean_keywords(topic_model, top_n=3)
    topics_docs: dict[int, list[str]] = {}
    topic_members: dict[int, list[int]] = {}
    for tid in topic_ids:
        member_idx = [i for i, t in enumerate(final_labels) if int(t) == tid]
        topic_members[tid] = member_idx
        topics_docs[tid] = [documents[i] for i in member_idx[:3]]
        # ensure every topic_id has an entry even if extract_clean_keywords missed it
        if tid not in topics_keywords:
            topics_keywords[tid] = ["—"]

    # Detect code-heavy topics: if majority of member chunks are tagged is_code, skip LLM
    code_topic_ids: set[int] = set()
    for tid in topic_ids:
        member_idx = topic_members[tid]
        code_count = sum(1 for i in member_idx if chunks[i].get("is_code", False))
        if code_count / max(len(member_idx), 1) > 0.5:
            code_topic_ids.add(tid)

    non_code_keywords = {tid: kws for tid, kws in topics_keywords.items() if tid not in code_topic_ids}
    non_code_docs = {tid: docs for tid, docs in topics_docs.items() if tid not in code_topic_ids}

    try:
        llm_labels = _generate_llm_labels(non_code_keywords, non_code_docs, llm_config, llm_cache)
    except Exception:
        llm_labels = {}

    # Code-heavy topics get keyword-only labels (no Flan-T5)
    for tid in code_topic_ids:
        kws = topics_keywords.get(tid, ["—"])
        clean = [k for k in kws if k and k != "—"]
        llm_labels[tid] = " ".join(w.title() for w in clean[:2]) if clean else "Code Snippets"

    topics: list[Topic] = []
    for tid in topic_ids:
        member_idx = topic_members[tid]
        if not member_idx:
            continue
        centroid = embeddings[member_idx].mean(axis=0)
        keywords = topics_keywords[tid]
        label = _clean_label(llm_labels.get(tid) or "", keywords)
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

    assert len(chunk_ids) == len(final_labels) == len(sources), (
        f"LENGTH MISMATCH: chunk_ids={len(chunk_ids)}, "
        f"final_labels={len(final_labels)}, sources={len(sources)}"
    )

    nan_mask = ~np.isfinite(umap_2d_coords).all(axis=1)
    if nan_mask.sum() > 0:
        umap_2d_coords[nan_mask] = [0.0, 0.0]

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
