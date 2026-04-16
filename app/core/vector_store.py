"""
Vector store abstraction over ChromaDB.

Design decision: ChromaDB is used for zero-infrastructure local development
and CI. In production, swap the adapter for Pinecone / Weaviate / pgvector
without changing any calling code — the interface stays the same.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.logging import get_logger
from app.core.models import SourceChunk

logger = get_logger(__name__)

_store: Optional["VectorStore"] = None


class VectorStore:
    """Thin wrapper around ChromaDB that speaks the app's domain language."""

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_name = settings.CHROMA_COLLECTION

    def ensure_collection(self) -> None:
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("collection ready", collection=self._collection_name, count=self._collection.count())

    @property
    def collection(self):
        if not hasattr(self, "_collection"):
            self.ensure_collection()
        return self._collection

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        document_id: str,
        title: str,
        chunks: List[str],
        embeddings: List[List[float]],
        tags: List[str],
        extra_metadata: dict,
    ) -> int:
        """Upsert chunks for a document. Returns number of chunks stored."""
        ids, docs, metas, embeds = [], [], [], []
        now = datetime.now(timezone.utc).isoformat()

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}__chunk_{i}"
            ids.append(chunk_id)
            docs.append(chunk)
            embeds.append(emb)
            metas.append({
                "document_id": document_id,
                "title": title,
                "chunk_index": i,
                "tags": ",".join(tags),
                "ingested_at": now,
                **{k: str(v) for k, v in extra_metadata.items()},
            })

        self.collection.upsert(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
        logger.info("chunks upserted", document_id=document_id, count=len(ids))
        return len(ids)

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        tags_filter: Optional[List[str]] = None,
        score_threshold: float = 0.35,
    ) -> List[SourceChunk]:
        """Return top-k relevant chunks above the similarity threshold."""
        where = None
        if tags_filter:
            # ChromaDB 'where' filter on comma-joined tags field
            where = {"$or": [{"tags": {"$contains": t}} for t in tags_filter]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(1, self.collection.count())),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[SourceChunk] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance → similarity score
            score = round(1.0 - dist, 4)
            if score < score_threshold:
                continue
            chunks.append(SourceChunk(
                document_id=meta["document_id"],
                title=meta["title"],
                chunk_index=meta["chunk_index"],
                content_preview=doc[:200],
                relevance_score=score,
            ))

        logger.debug("retrieval complete", returned=len(chunks), top_k=top_k)
        return chunks

    def get_document_chunks(self, document_id: str) -> List[dict]:
        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        return [
            {"chunk_index": m["chunk_index"], "content": d}
            for d, m in zip(results["documents"], results["metadatas"])
        ]

    def delete_document(self, document_id: str) -> int:
        results = self.collection.get(where={"document_id": document_id})
        ids = results["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self.collection.count()


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
