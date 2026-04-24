# DocuCluster

Local-first semantic document clustering and topic explorer. Drop in PDFs, DOCX,
TXT, or Markdown and get an Obsidian-style workspace: live UMAP scatter of every
chunk, HDBSCAN-discovered topics with LLM-generated labels, and hybrid
BM25 + semantic search that highlights matching clusters on the map.

Inspired by the **"Semantic Clustering and Topic Modeling"** chapter from
*Hands-On Large Language Models* (Alammar & Grootendorst) — BERTopic's modular
pipeline of embeddings → dimensionality reduction → clustering →
representation tuning → LLM labeling. All embeddings, clustering, and LLM
labeling run locally by default — no external APIs.

**Demonstrates:** NLP pipelines · unsupervised ML · vector embeddings ·
full-stack development · real-time WebSockets.

---

## Tech Stack

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![BERTopic](https://img.shields.io/badge/BERTopic-7c3aed)
![UMAP](https://img.shields.io/badge/UMAP-ff6f00)
![HDBSCAN](https://img.shields.io/badge/HDBSCAN-10b981)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-ee4c2c)
![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-646CFF?logo=vite&logoColor=white)
![Plotly.js](https://img.shields.io/badge/Plotly.js-3F4F75?logo=plotly&logoColor=white)
![Tailwind](https://img.shields.io/badge/Tailwind-38BDF8?logo=tailwindcss&logoColor=white)

---

## Architecture

```
                     ┌────────────────────────────────────────┐
   PDF / DOCX /      │              Frontend (Vite)           │
   TXT / MD  ──►     │  ┌──────────┬──────────┬──────────┐    │
                     │  │LeftPanel │ UMAPPlot │RightPanel│    │
                     │  │ upload   │ Plotly   │ topics + │    │
                     │  │ slider   │ scatter  │ chunks + │    │
                     │  │ settings │ annots   │ search   │    │
                     │  └──────────┴──────────┴──────────┘    │
                     │        Zustand store + WS client       │
                     └────────────────┬───────────────────────┘
                                      │ HTTP /api + WS /ws/pipeline
                     ┌────────────────┴───────────────────────┐
                     │             Backend (FastAPI)          │
                     │   ┌──────────────────────────────┐     │
                     │   │ api/routes + websocket_handler│    │
                     │   └──────────────┬────────────────┘    │
                     │                  ▼                     │
                     │   parsers ─► chunker ─► Embedder       │
                     │                           │            │
                     │                           ▼            │
                     │                       Reducer (UMAP)   │
                     │                           │            │
                     │                  ┌────────┴────────┐   │
                     │                  ▼                 ▼   │
                     │             Clusterer        2-D coords│
                     │             (HDBSCAN +       (plot)    │
                     │             reduce_outliers)           │
                     │                  │                     │
                     │                  ▼                     │
                     │       BERTopic (c-TF-IDF →             │
                     │         KeyBERTInspired →              │
                     │         flan-t5-base labels)           │
                     │                  │                     │
                     │                  ▼                     │
                     │       BM25Index + SemanticSearch       │
                     │           → HybridSearch               │
                     └────────────────────────────────────────┘
```

---

## Features

- **Local by default**: GTE-small embeddings, flan-t5-base labels, Ollama
  optional via base URL setting.
- **Heading-aware chunking**: heading split first, paragraph fallback at 400
  words, tiny chunks forward-merged below 40 words.
- **Cross-document clustering**: every chunk shares one embedding space; points
  are colored by cluster, marker symbol cycles by source file.
- **Live pipeline progress**: WebSocket streams `parsing → embedding → umap
  → clustering → outliers → ctfidf → labeling` stages with percent complete.
- **Hybrid search**: BM25 top 30 → semantic rerank → weighted final score.
  Results highlight matching points on the UMAP and dim everything else.
- **Cluster granularity slider**: tune `min_cluster_size` 2–20 and re-cluster.
- **Obsidian-inspired dark UI**: 3-pane layout, JetBrains Mono labels, purple
  accent, colored file dots, ⌘K search shortcut.

---

## How It Works

The pipeline follows the five-step recipe from the *Hands-On Large Language
Models* topic-modeling chapter, adapted for streaming progress over a
WebSocket:

1. **Parse and chunk.** Each uploaded file is split at headings (Markdown `#`,
   DOCX `Heading N` styles, or PDF runs with font size > 13). A paragraph
   fallback kicks in if a section is longer than 400 words, and tiny sections
   (under 40 words) are merged forward. Every chunk carries its source
   filename, heading, and page number.
2. **Embed.** Every chunk is passed through `thenlper/gte-small` (a compact
   Sentence-Transformer) to produce L2-normalized 384-D vectors. All documents
   share one embedding space — that's what makes cross-document clustering
   possible.
3. **Reduce.** UMAP projects the 384-D embeddings twice — once to 5-D for
   clustering (HDBSCAN struggles in high dimensions) and once to 2-D for
   plotting. Both use cosine distance, `min_dist=0.0`, and a fixed seed so
   topology stays stable across runs.
4. **Cluster.** HDBSCAN finds density-based clusters on the 5-D projection
   without needing a preset `k`. BERTopic's `reduce_outliers(strategy="embeddings")`
   reassigns the leftover `-1` noise points to their nearest cluster by
   cosine similarity.
5. **Represent.** BERTopic ranks keywords per cluster with c-TF-IDF, reranks
   them with KeyBERTInspired (embedding-based relevance), and finally a local
   flan-t5-base generator converts the top documents + keywords into a short
   human-readable label. The chapter calls this the "modular representation"
   pattern: each stage is swappable.

On top of that, a **BM25 index** over raw chunk text and a **semantic index**
over the same embeddings back a hybrid search that reranks BM25 candidates by
cosine similarity and highlights matching points on the UMAP scatter.

---

## Key Technical Decisions

- **HDBSCAN over k-means.** k-means demands a preset `k`, assumes spherical
  clusters, and has no concept of noise. Real documents don't cooperate: some
  topics are tight, some diffuse, and plenty of chunks are truly off-topic.
  HDBSCAN discovers the number of clusters from density, tolerates irregular
  shapes, and flags outliers as `-1` — which we then resolve with BERTopic's
  embedding-based `reduce_outliers` so every chunk still lands somewhere.
- **UMAP before clustering, not after.** Clustering 384-D embeddings directly
  produces unstable results (curse of dimensionality + HDBSCAN's reachability
  metric). Reducing to 5-D first preserves the manifold structure the
  embedding model captured while letting density-based clustering actually
  work. The 2-D view is a separate projection used *only* for the plot.
- **Hybrid search beats pure semantic.** Pure BM25 misses synonyms
  ("vehicle" ≠ "car"). Pure dense search drifts on rare named entities and
  exact phrases ("Section 4.2", product codes, people's names). The hybrid
  funnel — BM25 top 30 → cosine rerank → weighted blend
  (`0.3 * bm25_norm + 0.7 * cosine`) — gets lexical recall and semantic
  precision in one pass, at roughly the cost of a single dense search over
  the candidate pool.
- **Modular BERTopic pipeline.** Each stage (embed / reduce / cluster /
  represent) is a swappable class behind a stable interface. That's the whole
  point of the chapter: you can drop in Instructor-XL embeddings, DBSCAN, or
  a different LLM without rewriting the surrounding code. `LLMConfig` already
  exposes the local-vs-Ollama switch.
- **WebSocket for progress, REST for queries.** Search is a short
  request/response. The pipeline is a multi-minute process with seven
  distinct stages; pushing events over a WebSocket keeps the UI honest and
  makes progress visible without polling.

---

## Future Improvements

Drawn from the "BERTopic modularity" discussion in the chapter:

- **Online / incremental clustering.** Current pipeline re-fits HDBSCAN when
  files change. River's online HDBSCAN or BERTopic's `partial_fit` would let
  us add chunks without recomputing the full projection — essential for
  long-running sessions or streaming ingestion.
- **Hierarchical topic modeling.** BERTopic's `hierarchical_topics()` builds
  a dendrogram so users can zoom from broad themes ("machine learning") down
  to specific sub-topics ("backpropagation", "regularization"). A collapsible
  tree in the right panel would expose this naturally.
- **Dynamic topic modeling over time.** When chunks carry timestamps,
  `topics_over_time()` shows how topic prevalence shifts over days, months,
  or years — great for research corpora, meeting notes, or news archives.
  Would plug in as an extra view tab beside the UMAP scatter.
- **Guided / seed topic modeling.** BERTopic's seed-word API nudges topics
  toward user-specified themes, useful when a domain expert already knows
  what clusters they want to see surface.
- **Topic diff between runs.** Persist graph-like metadata across sessions
  so users can see which chunks moved clusters after tweaking
  `min_cluster_size` or uploading new files.

---

## Setup

### Prerequisites
- Python 3.11+
- Node 18+

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

First run downloads `thenlper/gte-small` (~125 MB) and `google/flan-t5-base`
(~250 MB) into the local HuggingFace cache.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173.

### One-shot (bash / WSL / macOS / Linux)

```bash
./start.sh
```

Runs backend + frontend and opens the browser.

### Smoke test

```bash
python -m backend.test_pipeline
```

Creates 3 mock docs (ML / cooking / space), runs the full pipeline, and prints
topics + cross-document cluster mix + hybrid search results.

---

## Screenshots

_(placeholder — add after first run)_

```
docs/screenshots/
  01-upload.png        # left panel + drop zone
  02-umap.png          # clustered scatter with labels
  03-search.png        # hybrid search highlighting a cluster
  04-topic-detail.png  # right panel topic view
```

---

## Project Layout

```
backend/
  parsers/       # pdf, docx, txt + chunker
  pipeline/      # embedder, reducer, clusterer, topic_modeler
  search/        # bm25, semantic, hybrid
  api/           # FastAPI routes + websocket
  models/        # Pydantic schemas
  main.py        # FastAPI app
  test_pipeline.py

frontend/
  src/
    components/  # LeftPanel, UMAPPlot, SearchBar, RightPanel, ...
    store/       # Zustand
    hooks/       # useWebSocket, usePipeline
```

---

## License

MIT.
