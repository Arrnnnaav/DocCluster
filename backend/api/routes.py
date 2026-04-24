"""HTTP routes: upload, search, topics, chunks, reset."""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..models.schemas import (
    ChunkInfo,
    ChunksResponse,
    ResetResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    TopicInfo,
    TopicsResponse,
    UploadResponse,
    UploadedFile,
)
from .websocket_handler import STATE

router = APIRouter(prefix="/api")

UPLOAD_DIR = Path(tempfile.gettempdir()) / "docucluster_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


@router.post("/upload", response_model=UploadResponse)
async def upload_files(files: list[UploadFile] = File(...)) -> UploadResponse:
    saved: list[UploadedFile] = []
    for upload in files:
        name = upload.filename or "unnamed"
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        file_id = uuid.uuid4().hex
        dest = UPLOAD_DIR / f"{file_id}{ext}"
        with dest.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        size = dest.stat().st_size
        STATE.uploaded_files[file_id] = {"path": str(dest), "filename": name}
        saved.append(UploadedFile(file_id=file_id, filename=name, path=str(dest), size=size))
    return UploadResponse(files=saved)


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    if STATE.hybrid is None:
        raise HTTPException(status_code=409, detail="Pipeline has not been run yet")
    result = STATE.hybrid.search(req.query, top_k=req.top_k)

    items = result.results
    if req.source_filter:
        allowed = set(req.source_filter)
        items = [r for r in items if r.source in allowed]
    if req.topic_filter:
        allowed_t = set(req.topic_filter)
        items = [r for r in items if r.topic_id in allowed_t]

    return SearchResponse(
        results=[
            SearchResultItem(
                chunk_id=r.chunk_id,
                chunk_text=r.chunk_text,
                heading=r.heading,
                source=r.source,
                topic_id=r.topic_id,
                topic_label=r.topic_label,
                bm25_score=r.bm25_score,
                semantic_score=r.semantic_score,
                final_score=r.final_score,
            )
            for r in items
        ],
        matched_topic_ids=result.matched_topic_ids,
    )


@router.get("/topics", response_model=TopicsResponse)
async def list_topics() -> TopicsResponse:
    if STATE.topic_result is None:
        return TopicsResponse(topics=[])
    return TopicsResponse(
        topics=[
            TopicInfo(
                id=t.id,
                label=t.label,
                keywords=t.keywords,
                doc_count=t.doc_count,
                chunk_ids=t.chunk_ids,
            )
            for t in STATE.topic_result.topics
        ]
    )


@router.get("/chunks/{topic_id}", response_model=ChunksResponse)
async def get_chunks_for_topic(topic_id: int) -> ChunksResponse:
    if STATE.topic_result is None:
        raise HTTPException(status_code=409, detail="Pipeline has not been run yet")
    topic = next((t for t in STATE.topic_result.topics if t.id == topic_id), None)
    if topic is None:
        raise HTTPException(status_code=404, detail=f"Topic {topic_id} not found")
    chunks: list[ChunkInfo] = []
    for cid in topic.chunk_ids:
        c = STATE.chunks_by_id.get(cid)
        if not c:
            continue
        chunks.append(
            ChunkInfo(
                id=c["id"],
                text=c["text"],
                heading=c.get("heading", ""),
                source=c.get("source", ""),
                page=c.get("page", 1),
                word_count=c.get("word_count", len(c.get("text", "").split())),
                topic_id=topic_id,
            )
        )
    return ChunksResponse(chunks=chunks)


@router.delete("/reset", response_model=ResetResponse)
async def reset_state() -> ResetResponse:
    for f in STATE.uploaded_files.values():
        try:
            Path(f["path"]).unlink(missing_ok=True)
        except OSError:
            pass
    STATE.reset()
    return ResetResponse(ok=True)
