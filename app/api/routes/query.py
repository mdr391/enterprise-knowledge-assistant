"""
Query endpoints — streaming and non-streaming RAG responses.
"""

import json
import time
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.logging import get_logger
from app.core.models import QueryRequest, QueryResponse
from app.retrieval.rag_pipeline import answer_query, retrieve_chunks
from app.llm.claude_client import get_llm_client
from app.core.config import settings

logger = get_logger(__name__)
router = APIRouter()


async def _sse_stream(request: QueryRequest, http_request: Request) -> AsyncIterator[str]:
    """
    Server-Sent Events generator.

    SSE format:
        data: <json>\n\n          ← token chunk
        data: [SOURCES]\n\n       ← source metadata after streaming
        data: [DONE]\n\n          ← terminal event
    """
    top_k = request.top_k or settings.RETRIEVAL_TOP_K

    # 1. Retrieve context
    try:
        chunks, full_texts = await retrieve_chunks(
            question=request.question,
            top_k=top_k,
            tags_filter=request.tags_filter,
        )
    except Exception as e:
        logger.error("retrieval failed", error=str(e))
        yield f"data: {json.dumps({'error': 'Retrieval failed. Please try again.'})}\n\n"
        return

    # 2. Stream LLM tokens
    llm = get_llm_client()
    try:
        async for token in llm.stream_answer(
            question=request.question,
            chunks=chunks,
            full_chunk_texts=full_texts,
        ):
            # Check if client disconnected mid-stream
            if await http_request.is_disconnected():
                logger.info("client disconnected mid-stream")
                return
            yield f"data: {json.dumps({'token': token})}\n\n"
    except Exception as e:
        logger.error("LLM streaming failed", error=str(e))
        yield f"data: {json.dumps({'error': 'LLM error. Please try again.'})}\n\n"
        return

    # 3. Send source metadata after tokens complete
    sources_payload = [
        {
            "document_id": c.document_id,
            "title": c.title,
            "chunk_index": c.chunk_index,
            "content_preview": c.content_preview,
            "relevance_score": c.relevance_score,
        }
        for c in chunks
    ]
    yield f"data: {json.dumps({'sources': sources_payload})}\n\n"
    yield "data: [DONE]\n\n"


@router.post(
    "/stream",
    summary="Ask a question — streaming response (SSE)",
    description=(
        "Retrieves relevant document chunks then streams the LLM answer token by token "
        "via Server-Sent Events. The final SSE event contains source citations."
    ),
)
async def query_stream(request: QueryRequest, http_request: Request):
    return StreamingResponse(
        _sse_stream(request, http_request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # Disable nginx buffering
        },
    )


@router.post(
    "/",
    response_model=QueryResponse,
    summary="Ask a question — synchronous response",
    description="Full RAG pipeline — returns the complete answer and sources in one response.",
)
async def query_sync(request: QueryRequest):
    request.stream = False
    try:
        return await answer_query(request)
    except Exception as e:
        logger.error("query failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
