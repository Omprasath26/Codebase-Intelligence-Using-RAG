"""Tests for the first-release Streamlit UI boundaries."""

from __future__ import annotations
from unittest.mock import patch
import httpx
import pytest
from src.generation import RAGResponse, SourceReference
from src.ui import (_build_query_payload, _format_source, _parse_rag_response,submit_query)


def make_source() -> SourceReference:
    """Create deterministic repository provenance."""

    return SourceReference(
        stable_id="chunk-request",
        artifact_id="artifact-request",
        repository="scrapy/scrapy",
        artifact_type="code",
        source_path_or_object_id=(
            "scrapy/http/request.py"
        ),
        source_url=(
            "https://github.com/scrapy/scrapy/"
            "blob/master/scrapy/http/request.py"
        ),
        commit_sha="abc123",
        ref="master",
        language="python",
        start_line=10,
        end_line=25,
        symbol="Request",
        parent_symbol=None,
    )


def make_response() -> RAGResponse:
    """Create a deterministic RAG response."""

    return RAGResponse(
        answer=(
            "The Request class is implemented "
            "in scrapy/http/request.py."
        ),
        sources=[make_source()],
        confidence=0.75,
        limitation=None,
        repository_revision="abc123",
        evidence_metadata={
            "dense_score": 0.91,
            "bm25_score": 0.82,
            "retrieval_method": "hybrid",
            "evidence_confidence": 0.75,
        },
        validated=True,
    )


def test_build_query_payload_includes_scope_and_conversation() -> None:
    """API payload preserves repository scope and conversation ID."""

    payload = _build_query_payload(
        query="Where is Request implemented?",
        repository="scrapy/scrapy",
        ref="master",
        conversation_id="conversation-123",
    )

    assert payload == {
        "query": "Where is Request implemented?",
        "repository": "scrapy/scrapy",
        "ref": "master",
        "conversation_id": "conversation-123",
    }


def test_build_query_payload_omits_optional_empty_values() -> None:
    """Optional API fields are omitted when not supplied."""

    payload = _build_query_payload(
        query="Explain Request.",
        repository=None,
        ref=None,
        conversation_id=None,
    )

    assert payload == {
        "query": "Explain Request.",
    }


def test_build_query_payload_rejects_empty_query() -> None:
    """The UI rejects empty queries before making an API call."""

    with pytest.raises(
        ValueError,
        match="query must be a non-empty string",):
        _build_query_payload(
            query="",
            repository=None,
            ref=None,
            conversation_id=None,
        )


def test_submit_query_posts_to_existing_query_endpoint() -> None:
    """UI submits the structured request to FastAPI."""

    response_data = {
        "conversation_id": "conversation-123",
        "correlation_id": "correlation-456",
        "result": make_response().model_dump(),
    }

    request = httpx.Request(
        "POST",
        "http://127.0.0.1:8000/query",
    )

    response = httpx.Response(
        200,
        request=request,
        json=response_data,
    )

    with patch(
        "src.ui.httpx.post",
        return_value=response,
    ) as mock_post:
        result = submit_query(
            api_url="http://127.0.0.1:8000",
            query="Where is Request implemented?",
            repository="scrapy/scrapy",
            ref="master",
            conversation_id="conversation-123",
        )

    assert result == response_data

    mock_post.assert_called_once_with(
        "http://127.0.0.1:8000/query",
        json={
            "query": "Where is Request implemented?",
            "repository": "scrapy/scrapy",
            "ref": "master",
            "conversation_id": "conversation-123",
        },
        timeout=60.0,
    )


def test_submit_query_reports_api_error() -> None:
    """HTTP API failures become a clear UI-level error."""

    request = httpx.Request(
        "POST",
        "http://127.0.0.1:8000/query",
    )

    response = httpx.Response(
        503,
        request=request,
        json={
            "detail": (
                "Query service is not configured."
            ),
        },
    )

    with patch(
        "src.ui.httpx.post",
        return_value=response,):
        with pytest.raises(
            RuntimeError,
            match="Query service is not configured",):
            submit_query(
                api_url="http://127.0.0.1:8000",
                query="Where is Request implemented?",
            )


def test_submit_query_reports_connection_error() -> None:
    """Connection failures are surfaced without exposing internals."""

    with patch(
        "src.ui.httpx.post",
        side_effect=httpx.ConnectError(
            "connection failed"
        ),):
        with pytest.raises(
            RuntimeError,
            match="Unable to connect to the query API"):
            submit_query(
                api_url="http://127.0.0.1:8000",
                query="Where is Request implemented?",
            )


def test_parse_rag_response_preserves_structured_result() -> None:
    """The UI consumes the existing RAGResponse contract."""

    response = make_response()

    data = {
        "conversation_id": "conversation-123",
        "correlation_id": "correlation-456",
        "result": response.model_dump(),
    }

    parsed = _parse_rag_response(
        data
    )

    assert parsed == response
    assert parsed.sources[0].artifact_type == "code"
    assert (
        parsed.sources[0].source_path_or_object_id
        == "scrapy/http/request.py"
    )
    assert parsed.sources[0].symbol == "Request"
    assert parsed.sources[0].start_line == 10
    assert parsed.sources[0].end_line == 25


def test_parse_rag_response_rejects_missing_result() -> None:
    """Malformed API responses are rejected."""

    with pytest.raises(
        RuntimeError,
        match="does not contain a valid result"):
        _parse_rag_response({})


def test_format_source_includes_path_lines_and_symbol() -> None:
    """Source display contains the important provenance location."""

    source = make_source()

    formatted = _format_source(
        source
    )

    assert (
        formatted
        == "scrapy/http/request.py:10-25 — Request"
    )


def test_format_source_supports_path_without_symbol() -> None:
    """Source formatting remains valid without a symbol."""

    source = SourceReference(
        stable_id="chunk-doc",
        artifact_id="artifact-doc",
        repository="scrapy/scrapy",
        artifact_type="documentation",
        source_path_or_object_id="docs/index.rst",
        start_line=None,
        end_line=None,
    )

    assert (
        _format_source(source)
        == "docs/index.rst")