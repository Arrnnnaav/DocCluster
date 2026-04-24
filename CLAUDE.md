# CLAUDE.md

## Knowledge Graph (Graphify)
Before answering any architecture question or searching for a file, read
`graphify-out/GRAPH_REPORT.md` if it exists. It contains god nodes (highest-degree
concepts everything routes through), community clusters, and surprising cross-file
connections. Use `/graphify query <term>` for precise traversal. This reduces token
cost dramatically — navigate the graph, don't grep raw files.

Re-run `/graphify .` after completing each major build step to keep the graph current.

---

# DocuCluster — Semantic Document Clustering & Topic Explorer

## Project Purpose
Local-first web app. Ingests PDF/DOCX/TXT/MD, chunks by heading (paragraph fallback), embeds via sentence-transformers, reduces with UMAP, clusters with HDBSCAN + `reduce_outliers()`, models topics with BERTopic (c-TF-IDF → KeyBERTInspired → local LLM labels). Obsidian-style UI: live UMAP scatter, hybrid search (BM25 + semantic), cross-document cluster exploration.

## Tech Stack
- Backend: Python 3.11, FastAPI, WebSockets, BERTopic, sentence-transformers, UMAP-learn, hdbscan, rank-bm25, pymupdf, python-docx, transformers (flan-t5-base)
- Frontend: React 18, Vite, Plotly.js, Tailwind CSS, Zustand
- Communication: WebSockets for pipeline streaming, REST for search

## Architecture
- `backend/` FastAPI app
  - `parsers/` PDF, DOCX, TXT, MD parsers
  - `pipeline/` embedding, umap, clustering, topic modeling
  - `search/` BM25 + semantic hybrid
  - `api/` FastAPI routes, websocket handlers
  - `models/` Pydantic schemas
- `frontend/` React + Vite
  - `components/` UI
  - `store/` Zustand state
  - `hooks/` custom hooks

## Key Decisions
- Chunking: heading-based first, paragraph fallback, 512 token max per chunk
- Clustering: HDBSCAN primary, `reduce_outliers()` post-process, min_cluster_size slider 2–20
- Representation: c-TF-IDF → KeyBERTInspired reranking → flan-t5-base LLM labels
- LLM: local only (flan-t5-base default, ollama optional via base URL setting)
- Search: BM25 candidate retrieval → semantic reranking → cluster highlight on UMAP
- Cross-document: all files share one embedding space, points colored by cluster

## Code Style
- Python: type hints everywhere, async pipeline functions, docstrings on all classes
- React: functional components only, TypeScript strict, no class components
- No external API calls by default — fully local
- Add `# WHY:` comments on non-obvious architectural decisions so Graphify captures rationale
