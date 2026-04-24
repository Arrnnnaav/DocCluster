"""End-to-end smoke test for the DocuCluster pipeline.

Creates 3 mock TXT docs (ML / cooking / space), runs the full pipeline, and
verifies cross-document clustering + hybrid search.

Run from project root:
    python -m backend.test_pipeline
"""
from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import umap

from .parsers import parse_file
from .pipeline.clusterer import Clusterer
from .pipeline.embedder import Embedder
from .pipeline.reducer import Reducer
from .pipeline.topic_modeler import LLMConfig, run_topic_modeling
from .search.bm25_index import BM25Index
from .search.hybrid_search import HybridSearch
from .search.semantic_search import SemanticSearch

ML_DOC = """# Introduction to Machine Learning
Machine learning is a branch of artificial intelligence where systems learn patterns
from data without being explicitly programmed. It powers recommendation engines,
spam filters, and voice assistants.

# Supervised Learning
Supervised learning uses labeled training data to fit a model. Classification predicts
discrete classes like spam vs not-spam, while regression predicts continuous values
like housing prices.

# Neural Networks
Deep neural networks stack layers of weighted transformations with nonlinear
activations. Backpropagation computes gradients so the optimizer can update
weights and minimize loss.

# Unsupervised Learning
Clustering algorithms like k-means and HDBSCAN group similar data points without
labels. Dimensionality reduction methods like PCA and UMAP project high-dimensional
vectors into a space we can visualize.

# Model Evaluation
Accuracy, precision, recall, and F1 score summarize classifier performance.
Cross-validation protects against overfitting on a single train/test split.

# Overfitting and Regularization
A model that memorizes the training set performs poorly on new data. Dropout and
L2 weight decay regularize neural networks. Early stopping halts training when
validation loss stops improving.

# Transfer Learning
Pretrained transformer models like BERT and GPT learn broad language understanding
from huge corpora. Fine-tuning adapts them to downstream tasks with little data.

# Reinforcement Learning
An agent learns a policy by interacting with an environment and receiving rewards.
Q-learning and policy gradient methods scale to games, robotics, and recommender
systems.

# Embeddings
Dense vector embeddings map words, sentences, or documents into a semantic space
where distance corresponds to meaning. They underpin modern search and retrieval.

# Ethics in AI
Bias in training data propagates into model predictions. Responsible machine
learning requires auditing datasets, explaining decisions, and monitoring
deployments.
"""

COOKING_DOC = """# Home Cooking Basics
Great home cooking starts with fresh ingredients, a sharp knife, and patience.
Taste as you go and season every layer, not just at the end.

# Knife Skills
A dull knife is dangerous. Hone the blade before every session and sharpen it
monthly. Practice the claw grip to keep fingertips clear while dicing onions.

# Stocks and Broths
Simmer roasted bones with carrots, celery, onion, and herbs for six hours. Skim
fat from the surface and strain through cheesecloth for a clear, glossy broth.

# Pasta from Scratch
Mix flour and eggs on a wooden board, knead for ten minutes, then rest the dough
under plastic wrap. Roll thin sheets and cut tagliatelle with a floured rolling pin.

# Pizza Dough
Ferment a high-hydration dough slowly in the fridge for 48 hours. Stretch gently
by hand to preserve the airy crumb. Bake on a preheated steel at maximum oven
temperature.

# Baking Bread
A sourdough starter lives on flour and water, burping carbon dioxide as wild
yeast multiplies. A hot Dutch oven traps steam and gives the crust its blistered
crackle.

# Roasting Vegetables
Cut vegetables into even pieces, toss in olive oil and salt, and spread on a hot
sheet pan. High heat caramelizes the sugars and crisps the edges.

# Braising Meat
Sear tough cuts like short ribs or pork shoulder to build flavor. Deglaze with
wine, add stock, and simmer covered for hours until the collagen melts into
gelatin.

# Sauces and Emulsions
A classic beurre blanc balances acid from white wine and butter, whisked off the
heat so it stays silky. Mayonnaise, hollandaise, and aioli all emulsify oil into
water with egg yolk.

# Desserts
Precise measurements matter in pastry. A tempered chocolate snaps crisply; a
well-laminated croissant fans into hundreds of buttery layers when baked.
"""

SPACE_DOC = """# The Solar System
Our solar system contains eight planets orbiting a single yellow dwarf star.
Jupiter dominates the outer system with a mass greater than all other planets
combined.

# Mars Exploration
Rovers like Curiosity and Perseverance drill into Martian rock looking for
biosignatures. The thin carbon dioxide atmosphere and red iron oxide surface
hint at a warmer, wetter past.

# The Moon and Artemis
NASA's Artemis program aims to return humans to the lunar south pole where
permanently shadowed craters may hold water ice. A sustained presence is the
stepping stone to crewed Mars missions.

# Rocket Propulsion
Chemical rockets trade propellant mass for exhaust velocity. The Tsiolkovsky
equation links delta-v to specific impulse, which is why staged rockets reach
orbit while single-stage designs struggle.

# Orbits and Gravity
A spacecraft in low Earth orbit falls around the planet at seven kilometers per
second. Geostationary orbits match Earth's rotation, keeping communications
satellites fixed above one point.

# The James Webb Telescope
Webb's segmented gold-coated mirror observes in the infrared from the L2
Lagrange point. It peers through dust to study galaxy formation, exoplanet
atmospheres, and star-forming regions.

# Exoplanets and Habitability
Transiting planets dim their host star as they pass in front of it. Thousands
have been confirmed. A few orbit within the habitable zone where liquid water
could exist.

# Black Holes
A black hole's event horizon is the point of no return. The Event Horizon
Telescope imaged Sagittarius A* at the center of our galaxy by combining radio
dishes across four continents.

# The Search for Life
Astrobiologists look for methane plumes on Enceladus, organic molecules on
Mars, and technosignatures around distant stars. Even a single confirmed
microbe elsewhere would reshape biology.

# Commercial Spaceflight
Reusable boosters from private launch providers dropped the cost of low Earth
orbit by more than an order of magnitude, opening space to small satellites,
research payloads, and paying tourists.
"""


def write_docs(tmp: Path) -> dict[str, Path]:
    docs = {
        "ml.md": ML_DOC,
        "cooking.md": COOKING_DOC,
        "space.md": SPACE_DOC,
    }
    out: dict[str, Path] = {}
    for name, body in docs.items():
        p = tmp / name
        p.write_text(body, encoding="utf-8")
        out[name] = p
    return out


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print(f"=== DocuCluster pipeline smoke test ===\ntmpdir: {tmp}")
        paths = write_docs(tmp)

        chunks: list[dict] = []
        for filename, path in paths.items():
            chunks.extend(parse_file(path, filename))
        print(f"\n[parse] {len(chunks)} chunks across {len(paths)} documents")
        per_file = Counter(c["source"] for c in chunks)
        for src, n in per_file.items():
            print(f"  {src}: {n} chunks")

        print("\n[embed] loading GTE-small (first call downloads model)…")
        embedder = Embedder()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed(texts)
        print(f"[embed] shape={embeddings.shape}")

        reducer = Reducer()
        reduced_5d, reduced_2d = reducer.fit_transform_both(embeddings)
        print(f"[umap]  5d={reduced_5d.shape}  2d={reduced_2d.shape}")

        clusterer = Clusterer()
        # Tiny corpus: small min_cluster_size so we get multiple clusters.
        labels = clusterer.fit(reduced_5d, min_cluster_size=3)
        print(f"[hdbscan] {clusterer.cluster_count} non-outlier clusters, "
              f"{(labels == -1).sum()} outliers")

        bertopic_umap = umap.UMAP(
            n_components=5, min_dist=0.0, metric="cosine", random_state=42
        )
        topic_result = run_topic_modeling(
            chunks=chunks,
            embeddings=embeddings,
            cluster_labels=labels,
            min_cluster_size=3,
            umap_model=bertopic_umap,
            hdbscan_model=clusterer.model_,
            umap_2d_coords=reduced_2d,
            llm_config=LLMConfig(),
            embedding_model=embedder.model,
        )

        print("\n=== Topics ===")
        for t in topic_result.topics:
            kw = ", ".join(t.keywords[:5])
            print(f"  [{t.id}] {t.label}  ({t.doc_count} chunks) — kw: {kw}")

        print("\n=== Doc → cluster distribution ===")
        chunk_topic = topic_result.chunk_topic_map
        source_of = {c["id"]: c["source"] for c in chunks}
        per_doc: dict[str, Counter[int]] = {}
        for cid, tid in chunk_topic.items():
            src = source_of[cid]
            per_doc.setdefault(src, Counter())[tid] += 1
        for src, counts in per_doc.items():
            summary = ", ".join(f"T{tid}:{n}" for tid, n in sorted(counts.items()))
            print(f"  {src}: {summary}")

        # Cross-doc check: does any cluster contain chunks from >1 source?
        cluster_sources: dict[int, set[str]] = {}
        for cid, tid in chunk_topic.items():
            cluster_sources.setdefault(tid, set()).add(source_of[cid])
        mixed = {tid: s for tid, s in cluster_sources.items()
                 if tid != -1 and len(s) > 1}
        print("\n=== Cross-document clusters ===")
        if mixed:
            for tid, s in mixed.items():
                print(f"  cluster {tid} spans: {sorted(s)}")
        else:
            print("  (no cluster contained chunks from multiple documents —"
                  " topics stayed per-document)")

        # ---- Hybrid search ----
        bm25 = BM25Index()
        bm25.build(chunks)
        semantic = SemanticSearch(embedder, embeddings, [c["id"] for c in chunks])
        topic_labels = {t.id: t.label for t in topic_result.topics}
        centroids = {t.id: np.asarray(t.centroid_embedding) for t in topic_result.topics}
        hybrid = HybridSearch(
            bm25=bm25,
            semantic=semantic,
            chunks_by_id={c["id"]: c for c in chunks},
            chunk_topic_map=chunk_topic,
            topic_labels=topic_labels,
            topic_centroids=centroids,
        )

        queries = [
            "how do neural networks train",
            "best way to make bread at home",
            "exploring other planets in the solar system",
        ]
        print("\n=== Hybrid search ===")
        for q in queries:
            print(f"\n  query: {q!r}")
            result = hybrid.search(q, top_k=3)
            for r in result.results:
                snippet = r.chunk_text.replace("\n", " ")[:80]
                print(f"    {r.source:12s} T{r.topic_id:<2d} "
                      f"final={r.final_score:.3f}  {snippet}…")
            if result.matched_topic_ids:
                print(f"    matched topics: {result.matched_topic_ids}")

        # Assertions (loose — tiny corpus, stochastic).
        assert len(chunks) >= 15, "expected at least 15 chunks across 3 docs"
        assert clusterer.cluster_count >= 2, "expected at least 2 clusters"
        print("\n=== OK ===")


if __name__ == "__main__":
    main()
