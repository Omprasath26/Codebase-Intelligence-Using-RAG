"""Tests for the FastAPI query and health API."""

from fastapi.testclient import TestClient

from src.api import (
    APIQueryService,
    QueryRequest,
    create_app,
)
from src.generation import RAGResponse


def make_response() -> RAGResponse:
    """Create a deterministic RAG response for API tests."""
    return RAGResponse(
        answer="The Request class is implemented in the request module.",
        sources=[],
        confidence=0.9,
        limitation=None,
        repository_revision="abc123",
        evidence_metadata={
            "validation_valid": True,
        },
        validated=True,
    )


def test_health_endpoint_returns_ok():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_status_endpoint_reports_unconfigured_service():
    client = TestClient(create_app())

    response = client.get("/status")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "query_handler_configured": False,
    }


def test_query_endpoint_returns_503_when_service_is_unconfigured():
    client = TestClient(create_app())

    response = client.post(
        "/query",
        json={
            "query": "Where is the Request class implemented?",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Query service is not configured."
    )


def test_query_request_accepts_repository_scope():
    request = QueryRequest(
        query="Where is Request implemented?",
        repository="scrapy/scrapy",
        ref="master",
    )

    assert request.query == (
        "Where is Request implemented?"
    )
    assert request.repository == "scrapy/scrapy"
    assert request.ref == "master"
    assert request.conversation_id is None


def test_query_request_accepts_conversation_id():
    request = QueryRequest(
        query="What does its constructor do?",
        repository="scrapy/scrapy",
        ref="master",
        conversation_id="conversation-123",
    )

    assert request.conversation_id == (
        "conversation-123"
    )


def test_query_request_rejects_empty_query():
    try:
        QueryRequest(query="")
    except Exception:
        pass
    else:
        raise AssertionError(
            "Expected validation failure for empty query."
        )


def test_query_service_rejects_non_callable_handler():
    try:
        APIQueryService(
            handler="invalid"
        )  # type: ignore[arg-type]
    except TypeError as exc:
        assert str(exc) == (
            "handler must be callable or None."
        )
    else:
        raise AssertionError(
            "Expected TypeError for invalid handler."
        )


def test_query_service_reports_configuration():
    unconfigured = APIQueryService()

    configured = APIQueryService(
        handler=lambda request: make_response()
    )

    assert unconfigured.configured is False
    assert configured.configured is True


def test_query_service_executes_handler():
    calls = []

    def handler(request: QueryRequest) -> RAGResponse:
        calls.append(request)
        return make_response()

    service = APIQueryService(
        handler=handler
    )

    request = QueryRequest(
        query="Where is Request implemented?",
        repository="scrapy/scrapy",
    )

    result = service.execute(request)

    assert result == make_response()
    assert calls == [request]


def test_query_service_rejects_invalid_request():
    service = APIQueryService(
        handler=lambda request: make_response()
    )

    try:
        service.execute(
            "invalid"  # type: ignore[arg-type]
        )
    except TypeError as exc:
        assert str(exc) == (
            "request must be a QueryRequest."
        )
    else:
        raise AssertionError(
            "Expected TypeError for invalid request."
        )


def test_query_service_requires_handler():
    service = APIQueryService()

    try:
        service.execute(
            QueryRequest(
                query="Where is Request implemented?"
            )
        )
    except RuntimeError as exc:
        assert str(exc) == (
            "Query service is not configured."
        )
    else:
        raise AssertionError(
            "Expected RuntimeError for missing handler."
        )


def test_query_service_requires_rag_response():
    service = APIQueryService(
        handler=lambda request: "invalid"
    )

    try:
        service.execute(
            QueryRequest(
                query="Where is Request implemented?"
            )
        )
    except RuntimeError as exc:
        assert str(exc) == (
            "Query handler must return a RAGResponse."
        )
    else:
        raise AssertionError(
            "Expected RuntimeError for invalid handler result."
        )


def test_query_endpoint_returns_structured_response():
    service = APIQueryService(
        handler=lambda request: make_response()
    )

    client = TestClient(
        create_app(service)
    )

    response = client.post(
        "/query",
        json={
            "query": "Where is the Request class implemented?",
            "repository": "scrapy/scrapy",
            "ref": "master",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["conversation_id"]
    assert body["correlation_id"]

    assert body["result"]["answer"] == (
        "The Request class is implemented in the request module."
    )

    assert body["result"]["validated"] is True
    assert body["result"]["repository_revision"] == (
        "abc123"
    )


def test_query_endpoint_preserves_supplied_conversation_id():
    service = APIQueryService(
        handler=lambda request: make_response()
    )

    client = TestClient(
        create_app(service)
    )

    response = client.post(
        "/query",
        json={
            "query": "What does its constructor do?",
            "repository": "scrapy/scrapy",
            "ref": "master",
            "conversation_id": "conversation-123",
        },
    )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == (
        "conversation-123"
    )


def test_query_endpoint_generates_conversation_id_when_missing():
    service = APIQueryService(
        handler=lambda request: make_response()
    )

    client = TestClient(
        create_app(service)
    )

    response = client.post(
        "/query",
        json={
            "query": "Where is Request implemented?",
        },
    )

    assert response.status_code == 200

    conversation_id = response.json()["conversation_id"]

    assert conversation_id
    assert isinstance(conversation_id, str)


def test_query_endpoint_generates_correlation_id():
    service = APIQueryService(
        handler=lambda request: make_response()
    )

    client = TestClient(
        create_app(service)
    )

    response = client.post(
        "/query",
        json={
            "query": "Where is Request implemented?",
        },
    )

    assert response.status_code == 200

    correlation_id = response.json()["correlation_id"]

    assert correlation_id
    assert isinstance(correlation_id, str)


def test_query_endpoint_rejects_empty_query():
    service = APIQueryService(
        handler=lambda request: make_response()
    )

    client = TestClient(
        create_app(service)
    )

    response = client.post(
        "/query",
        json={
            "query": "",
        },
    )

    assert response.status_code == 422


def test_query_endpoint_rejects_missing_query():
    service = APIQueryService(
        handler=lambda request: make_response()
    )

    client = TestClient(
        create_app(service)
    )

    response = client.post(
        "/query",
        json={},
    )

    assert response.status_code == 422


def test_query_endpoint_maps_handler_failure_to_500():
    def failing_handler(request: QueryRequest) -> RAGResponse:
        raise ValueError("internal failure")

    service = APIQueryService(
        handler=failing_handler
    )

    client = TestClient(
        create_app(service)
    )

    response = client.post(
        "/query",
        json={
            "query": "Where is Request implemented?",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Query processing failed."
    )


def test_query_endpoint_passes_request_to_handler():
    received = []

    def handler(request: QueryRequest) -> RAGResponse:
        received.append(request)
        return make_response()

    service = APIQueryService(
        handler=handler
    )

    client = TestClient(
        create_app(service)
    )

    response = client.post(
        "/query",
        json={
            "query": "Where is Request implemented?",
            "repository": "scrapy/scrapy",
            "ref": "master",
            "conversation_id": "conversation-123",
        },
    )

    assert response.status_code == 200
    assert len(received) == 1

    assert received[0].query == (
        "Where is Request implemented?"
    )
    assert received[0].repository == "scrapy/scrapy"
    assert received[0].ref == "master"
    assert received[0].conversation_id == (
        "conversation-123"
    )


def test_query_service_wraps_handler_failure():
    def failing_handler(request: QueryRequest) -> RAGResponse:
        raise ValueError("failure")

    service = APIQueryService(
        handler=failing_handler
    )

    try:
        service.execute(
            QueryRequest(
                query="Where is Request implemented?"
            )
        )
    except RuntimeError as exc:
        assert str(exc) == (
            "Query processing failed."
        )
    else:
        raise AssertionError(
            "Expected RuntimeError for handler failure."
        )


def test_query_endpoint_returns_fresh_correlation_ids():
    service = APIQueryService(
        handler=lambda request: make_response()
    )

    client = TestClient(
        create_app(service)
    )

    first = client.post(
        "/query",
        json={
            "query": "Where is Request implemented?",
        },
    )

    second = client.post(
        "/query",
        json={
            "query": "Where is Response implemented?",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200

    assert (
        first.json()["correlation_id"]
        != second.json()["correlation_id"]
    )


def test_status_reports_configured_service():
    service = APIQueryService(
        handler=lambda request: make_response()
    )

    client = TestClient(
        create_app(service)
    )

    response = client.get("/status")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "query_handler_configured": True,
    }