"""
Integration tests for the full RAG pipeline.
Uses mocked LLM and embedding APIs — no real API keys needed.
Run with: pytest tests/integration/ -v
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.models import IngestRequest


# ── Fixtures ──────────────────────────────────────────────────────────────────

FAKE_EMBEDDING = [0.1] * 1536  # matches EMBEDDING_DIMENSIONS
FAKE_ANSWER = "The vacation policy allows 20 days per year."


@pytest.fixture(autouse=True)
def mock_openai_embedder():
    """Patch the embedder so no real OpenAI calls are made."""
    mock = AsyncMock()
    mock.embed_single.return_value = FAKE_EMBEDDING
    mock.embed_batch.return_value = [FAKE_EMBEDDING]
    with patch("app.llm.embeddings.get_embedder", return_value=mock):
        with patch("app.ingestion.pipeline.get_embedder", return_value=mock):
            with patch("app.retrieval.rag_pipeline.get_embedder", return_value=mock):
                yield mock


@pytest.fixture(autouse=True)
def mock_llm_client():
    """Patch the LLM client so no real Anthropic calls are made."""
    mock = AsyncMock()
    mock.complete_answer.return_value = (FAKE_ANSWER, 42.0)

    async def fake_stream(*args, **kwargs):
        for token in FAKE_ANSWER.split():
            yield token + " "

    mock.stream_answer = fake_stream
    with patch("app.retrieval.rag_pipeline.get_llm_client", return_value=mock):
        with patch("app.api.routes.query.get_llm_client", return_value=mock):
            yield mock


@pytest_asyncio.fixture
async def client(tmp_path):
    """Async test client with isolated ChromaDB in a temp directory."""
    with patch("app.core.config.settings.CHROMA_PERSIST_DIR", str(tmp_path / "chroma")):
        with patch("app.core.vector_store._store", None):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                yield ac


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


@pytest.mark.asyncio
async def test_ingest_document(client):
    payload = {
        "content": "Employees are entitled to 20 days of vacation per year. "
                   "Unused days can be carried over to the following year up to a maximum of 5 days. "
                   "Vacation must be approved by your direct manager at least two weeks in advance.",
        "title": "Vacation Policy 2024",
        "tags": ["HR", "policy"],
        "source": "text",
    }
    response = await client.post("/ingest/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_created"] >= 1
    assert data["title"] == "Vacation Policy 2024"
    assert "document_id" in data


@pytest.mark.asyncio
async def test_query_sync_after_ingest(client):
    # First ingest
    ingest_payload = {
        "content": "The expense reimbursement limit is $500 per month for remote work equipment. "
                   "Receipts must be submitted within 30 days of purchase via the internal portal.",
        "title": "Expense Policy",
        "tags": ["finance"],
        "source": "text",
    }
    await client.post("/ingest/", json=ingest_payload)

    # Then query
    query_payload = {"question": "What is the expense reimbursement limit?", "stream": False}
    response = await client.post("/query/", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "latency_ms" in data


@pytest.mark.asyncio
async def test_query_returns_no_hallucination_signal(client):
    """When knowledge base is empty, the answer should indicate insufficient info."""
    # Empty KB — override LLM to return the expected no-info response
    with patch("app.retrieval.rag_pipeline.get_llm_client") as mock_llm:
        instance = AsyncMock()
        instance.complete_answer.return_value = (
            "I don't have enough information in the knowledge base to answer this question.",
            10.0,
        )
        mock_llm.return_value = instance
        response = await client.post(
            "/query/",
            json={"question": "What is the parental leave policy?", "stream": False},
        )
    assert response.status_code == 200
    data = response.json()
    assert "don't have enough information" in data["answer"].lower() or data["retrieved_chunks"] == 0


@pytest.mark.asyncio
async def test_ingest_validation_error(client):
    response = await client.post("/ingest/", json={"content": "Short", "title": "Doc"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_kb_stats(client):
    response = await client.get("/ingest/stats")
    assert response.status_code == 200
    assert "total_chunks" in response.json()


@pytest.mark.asyncio
async def test_delete_nonexistent_document(client):
    response = await client.delete("/ingest/nonexistent-doc-id")
    assert response.status_code == 404
