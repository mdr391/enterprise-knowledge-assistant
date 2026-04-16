"""
Shared Pydantic models — single source of truth for all API schemas.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Ingestion ─────────────────────────────────────────────────────────────────

class DocumentSource(str, Enum):
    UPLOAD = "upload"
    URL = "url"
    TEXT = "text"


class IngestRequest(BaseModel):
    content: str = Field(..., min_length=10, description="Raw document text to ingest")
    title: str = Field(..., min_length=1, max_length=256)
    source: DocumentSource = DocumentSource.TEXT
    tags: List[str] = Field(default_factory=list, max_length=10)
    metadata: dict = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def lowercase_tags(cls, v: List[str]) -> List[str]:
        return [t.lower().strip() for t in v]


class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int
    title: str
    message: str


class DocumentMeta(BaseModel):
    document_id: str
    title: str
    source: str
    tags: List[str]
    chunk_count: int
    ingested_at: datetime


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Natural language question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of context chunks to retrieve")
    tags_filter: Optional[List[str]] = Field(None, description="Restrict retrieval to documents with these tags")
    stream: bool = Field(True, description="Stream the LLM response via SSE")

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        return v.strip()


class SourceChunk(BaseModel):
    document_id: str
    title: str
    chunk_index: int
    content_preview: str     # First 200 chars — shown in citations
    relevance_score: float


class QueryResponse(BaseModel):
    """Used for non-streaming responses."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str
    answer: str
    sources: List[SourceChunk]
    model: str
    latency_ms: float
    retrieved_chunks: int


# ── Health ────────────────────────────────────────────────────────────────────

class HealthStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class ComponentHealth(BaseModel):
    status: HealthStatus
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: HealthStatus
    version: str
    components: dict[str, ComponentHealth]
    uptime_seconds: float
