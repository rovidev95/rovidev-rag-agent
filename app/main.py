"""FastAPI surface for the RAG agent."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings, build_pipeline
from app.models import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    app.state.pipeline = build_pipeline(settings)
    yield


app = FastAPI(
    title="RoviDev RAG Agent",
    description="Grounded answers with inline source citations. By RoviDev.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    settings: Settings = app.state.settings
    return {"status": "ok", "backend": settings.backend}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    docs, chunks = app.state.pipeline.ingest(request.documents)
    return IngestResponse(ingested_documents=docs, total_chunks=chunks)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return app.state.pipeline.ask(request.question, top_k=request.top_k)
