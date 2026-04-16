"""
Unit tests for Pydantic model validation.
No external dependencies required.
"""

import pytest
from pydantic import ValidationError
from app.core.models import IngestRequest, QueryRequest, DocumentSource


class TestIngestRequest:
    def test_valid_request(self):
        req = IngestRequest(
            content="This is a valid document with enough content to process.",
            title="Test Document",
            tags=["policy", "HR"],
        )
        assert req.title == "Test Document"
        assert req.tags == ["policy", "hr"]  # lowercased

    def test_tags_are_lowercased(self):
        req = IngestRequest(
            content="Content here for testing purposes.",
            title="Doc",
            tags=["  HR  ", "POLICY", "Finance"],
        )
        assert req.tags == ["hr", "policy", "finance"]

    def test_content_too_short_raises(self):
        with pytest.raises(ValidationError):
            IngestRequest(content="Short", title="Doc")

    def test_empty_title_raises(self):
        with pytest.raises(ValidationError):
            IngestRequest(content="Valid content here that is long enough.", title="")

    def test_default_source_is_text(self):
        req = IngestRequest(content="Enough content for testing here.", title="Doc")
        assert req.source == DocumentSource.TEXT

    def test_metadata_defaults_to_empty(self):
        req = IngestRequest(content="Enough content for testing here.", title="Doc")
        assert req.metadata == {}


class TestQueryRequest:
    def test_valid_query(self):
        req = QueryRequest(question="What is the vacation policy?")
        assert req.question == "What is the vacation policy?"
        assert req.stream is True  # default

    def test_question_is_stripped(self):
        req = QueryRequest(question="  What is the policy?  ")
        assert req.question == "What is the policy?"

    def test_question_too_short_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Hi")

    def test_top_k_bounds(self):
        # Valid
        QueryRequest(question="Valid question here?", top_k=5)
        # Too low
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question here?", top_k=0)
        # Too high
        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question here?", top_k=21)

    def test_tags_filter_optional(self):
        req = QueryRequest(question="What is the expense policy?")
        assert req.tags_filter is None

        req2 = QueryRequest(question="What is the expense policy?", tags_filter=["finance"])
        assert req2.tags_filter == ["finance"]
