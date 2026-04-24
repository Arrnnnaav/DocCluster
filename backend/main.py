"""FastAPI entry point for DocuCluster."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as api_router
from .api.websocket_handler import STATE, pipeline_websocket

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the embedding model so the first request doesn't pay the download/load cost.
    _ = STATE.embedder.model
    yield


app = FastAPI(title="DocuCluster", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.websocket("/ws/pipeline")
async def ws_pipeline(websocket: WebSocket) -> None:
    await pipeline_websocket(websocket)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "app": "DocuCluster"}
