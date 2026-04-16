"""
Unit tests for the ingestion pipeline.
These tests run with zero external dependencies — no API keys, no DB.
"""

import pytest
from app.ingestion.pipeline import chunk_text, _token_count


class TestChunkText:
    def test_short_text_produces_single_chunk(self):
        text = "This is a short document. It has two sentences."
        chunks = chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_multiple_chunks(self):
        # Generate ~2000 tokens of text
        sentence = "The enterprise platform serves thousands of employees daily. "
        text = sentence * 100
        chunks = chunk_text(text, chunk_size=200, overlap=30)
        assert len(chunks) > 1

    def test_chunks_have_overlap(self):
        """Consecutive chunks should share some content due to overlap."""
        sentence = "Sentence number {}. "
        text = "".join(sentence.format(i) for i in range(60))
        chunks = chunk_text(text, chunk_size=100, overlap=30)
        assert len(chunks) >= 2
        # At least some tokens from chunk N should appear in chunk N+1
        words_in_first = set(chunks[0].split())
        words_in_second = set(chunks[1].split())
        overlap_words = words_in_first & words_in_second
        assert len(overlap_words) > 0, "Expected overlap between consecutive chunks"

    def test_empty_text_returns_empty(self):
        assert chunk_text("   ") == []

    def test_no_chunks_exceed_size_budget(self):
        sentence = "This is a normal enterprise document sentence about policies and procedures. "
        text = sentence * 50
        chunk_size = 150
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=20)
        for chunk in chunks:
            assert _token_count(chunk) <= chunk_size * 1.2  # 20% tolerance for boundary sentences

    def test_paragraph_breaks_respected(self):
        # Each paragraph is ~7 tokens; chunk_size=8 forces splits between them
        text = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird paragraph content."
        chunks = chunk_text(text, chunk_size=8, overlap=2)
        assert len(chunks) >= 2

    def test_all_content_preserved(self):
        """No content should be silently dropped during chunking."""
        words = ["word{}".format(i) for i in range(200)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        all_chunk_text = " ".join(chunks)
        # Every unique word should appear in at least one chunk
        for word in words:
            assert word in all_chunk_text, f"'{word}' was lost during chunking"


class TestTokenCount:
    def test_empty_string(self):
        assert _token_count("") == 0

    def test_approximate_count(self):
        # 4 chars ≈ 1 token
        text = "a" * 400
        assert _token_count(text) == 100
