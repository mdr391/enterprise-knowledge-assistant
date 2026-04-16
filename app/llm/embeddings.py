"""
Embeddings layer — wraps OpenAI's embedding API with:
  - Async batch processing (one API call per ingestion)
  - In-memory LRU cache for repeated queries (saves cost in dev)
  - Swap-friendly interface: replace with sentence-transformers locally
"""

from __future__ import annotations

import asyncio
import hashlib
from functools import lru_cache
from typing import List, Optional

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_embedder: Optional["Embedder"] = None


class Embedder:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.EMBEDDING_MODEL
        self._cache: dict[str, List[float]] = {}  # simple in-memory cache
        logger.info("embedder initialised", model=self._model)

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def embed_single(self, text: str) -> List[float]:
        """Embed a single string with cache lookup."""
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        response = await self._client.embeddings.create(
            input=[text],
            model=self._model,
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )
        embedding = response.data[0].embedding
        self._cache[key] = embedding
        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch in one API call.
        Falls back to individual calls for cache hits to avoid re-embedding.
        """
        # Separate cached from uncached
        results: dict[int, List[float]] = {}
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Single batched API call for uncached texts
        if uncached_texts:
            logger.debug("embedding batch", count=len(uncached_texts), model=self._model)
            response = await self._client.embeddings.create(
                input=uncached_texts,
                model=self._model,
                dimensions=settings.EMBEDDING_DIMENSIONS,
            )
            for idx, emb_obj in zip(uncached_indices, response.data):
                embedding = emb_obj.embedding
                results[idx] = embedding
                # Cache for future queries
                key = self._cache_key(texts[idx])
                self._cache[key] = embedding

        return [results[i] for i in range(len(texts))]


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
