"""Ingestion endpoints."""

from fastapi import APIRouter, HTTPException

from app.core.logging import get_logger
from app.core.models import IngestRequest, IngestResponse
from app.core.vector_store import get_vector_store
from app.ingestion.pipeline import ingest_document

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/",
    response_model=IngestResponse,
    summary="Ingest a document",
    description="Chunk, embed, and store a document in the knowledge base.",
)
async def ingest(request: IngestRequest):
    try:
        return await ingest_document(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("ingestion failed", title=request.title, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Ingestion failed. Check logs for details.")


@router.delete(
    "/{document_id}",
    summary="Delete a document",
    description="Remove all chunks for a document from the knowledge base.",
)
async def delete_document(document_id: str):
    store = get_vector_store()
    deleted = store.delete_document(document_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No document found with id '{document_id}'")
    return {"document_id": document_id, "chunks_deleted": deleted, "message": "Document removed."}


@router.get(
    "/stats",
    summary="Knowledge base stats",
)
async def kb_stats():
    store = get_vector_store()
    return {"total_chunks": store.count()}
