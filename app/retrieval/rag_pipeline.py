"""
Retrieval orchestration — the core of the RAG pipeline.

Flow:
  question → embed → vector search → fetch full chunk text → LLM → answer

Design notes:
  - Full chunk text is fetched from ChromaDB after scoring so we can apply
    the score threshold without fetching unnecessary data.
  - The score threshold is a hard gate: below it, chunks are dropped even
    if they are the "best" available. This prevents the LLM from hallucinating
    answers from weakly-related passages.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from app.core.config import settings
from app.core.logging import get_logger
from app.core.models import QueryRequest, QueryResponse, SourceChunk
from app.core.vector_store import get_vector_store
from app.llm.embeddings import get_embedder
from app.llm.claude_client import get_llm_client

logger = get_logger(__name__)


async def retrieve_chunks(
    question: str,
    top_k: int,
    tags_filter: Optional[List[str]],
) -> Tuple[List[SourceChunk], List[str]]:
    """
    Embed the question and retrieve relevant chunks.
    Returns (chunk_metadata_list, full_text_list).
    """
    embedder = get_embedder()
    query_embedding = await embedder.embed_single(question)

    store = get_vector_store()
    chunks = store.query(
        query_embedding=query_embedding,
        top_k=top_k,
        tags_filter=tags_filter,
        score_threshold=settings.RETRIEVAL_SCORE_THRESHOLD,
    )

    # Fetch the full text for each matched chunk (for LLM context)
    full_texts: List[str] = []
    for chunk in chunks:
        doc_chunks = store.get_document_chunks(chunk.document_id)
        matching = next(
            (c["content"] for c in doc_chunks if c["chunk_index"] == chunk.chunk_index),
            chunk.content_preview,  # fallback to preview if not found
        )
        full_texts.append(matching)

    logger.info(
        "retrieval done",
        question_snippet=question[:60],
        chunks_returned=len(chunks),
        tags_filter=tags_filter,
    )
    return chunks, full_texts


async def answer_query(request: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline for non-streaming responses.
    Used by the non-streaming endpoint and for integration tests.
    """
    top_k = request.top_k or settings.RETRIEVAL_TOP_K
    start = time.perf_counter()

    chunks, full_texts = await retrieve_chunks(
        question=request.question,
        top_k=top_k,
        tags_filter=request.tags_filter,
    )

    llm = get_llm_client()
    answer, llm_latency = await llm.complete_answer(
        question=request.question,
        chunks=chunks,
        full_chunk_texts=full_texts,
    )

    total_latency = round((time.perf_counter() - start) * 1000, 1)

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=chunks,
        model=settings.LLM_MODEL,
        latency_ms=total_latency,
        retrieved_chunks=len(chunks),
    )
