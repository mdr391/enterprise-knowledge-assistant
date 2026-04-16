"""
Ingestion pipeline: text → chunks → embeddings → vector store.

Chunking strategy: sentence-aware sliding window with token overlap.
This prevents context loss at chunk boundaries — a common production failure.
"""

from __future__ import annotations

import re
import uuid
from typing import List

from app.core.config import settings
from app.core.logging import get_logger
from app.core.models import IngestRequest, IngestResponse
from app.core.vector_store import get_vector_store
from app.llm.embeddings import get_embedder

logger = get_logger(__name__)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving structure."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    # Split on sentence boundaries while keeping the delimiter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    # Also split on double newlines (paragraph breaks)
    result = []
    for s in sentences:
        parts = s.split("\n\n")
        result.extend(p.strip() for p in parts if p.strip())
    return result


def _token_count(text: str) -> int:
    """Rough token estimate: ~4 chars per token (GPT-style)."""
    return len(text) // 4


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Sentence-aware chunking with overlap.

    Rather than splitting mid-sentence, accumulate sentences until the chunk
    reaches the token budget, then start a new chunk overlapping the last
    `overlap` tokens from the previous one.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    sentences = _split_into_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = _token_count(sentence)

        if current_tokens + s_tokens > chunk_size and current:
            chunk_text = " ".join(current)
            chunks.append(chunk_text)

            # Build overlap: walk backward from end of current until we've
            # collected ~overlap tokens
            overlap_sentences: List[str] = []
            overlap_tokens = 0
            for sent in reversed(current):
                t = _token_count(sent)
                if overlap_tokens + t > overlap:
                    break
                overlap_sentences.insert(0, sent)
                overlap_tokens += t

            current = overlap_sentences
            current_tokens = overlap_tokens

        current.append(sentence)
        current_tokens += s_tokens

    if current:
        chunks.append(" ".join(current))

    logger.debug("chunking complete", total_chunks=len(chunks), input_tokens=_token_count(text))
    return [c for c in chunks if c.strip()]


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def ingest_document(request: IngestRequest) -> IngestResponse:
    """
    Full ingestion pipeline:
      1. Validate and chunk the document
      2. Embed all chunks in a single batched API call
      3. Upsert to vector store
    """
    document_id = str(uuid.uuid4())
    logger.info("ingestion started", document_id=document_id, title=request.title)

    # 1. Chunk
    chunks = chunk_text(request.content)
    if not chunks:
        raise ValueError("Document produced no valid chunks after processing.")

    logger.info("document chunked", document_id=document_id, chunks=len(chunks))

    # 2. Embed (batched — one API call regardless of chunk count)
    embedder = get_embedder()
    embeddings = await embedder.embed_batch(chunks)

    # 3. Store
    store = get_vector_store()
    n_stored = store.add_chunks(
        document_id=document_id,
        title=request.title,
        chunks=chunks,
        embeddings=embeddings,
        tags=request.tags,
        extra_metadata={"source": request.source.value, **request.metadata},
    )

    logger.info("ingestion complete", document_id=document_id, chunks_stored=n_stored)

    return IngestResponse(
        document_id=document_id,
        chunks_created=n_stored,
        title=request.title,
        message=f"Successfully ingested '{request.title}' into {n_stored} searchable chunks.",
    )
