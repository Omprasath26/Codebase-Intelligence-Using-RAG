"""FastAPI query, health, and status API for the RAG system."""

from __future__ import annotations

from typing import Protocol
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.generation import RAGResponse


class QueryRequest(BaseModel):
    """Structured request submitted to the developer QA API."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1)
    repository: str | None = None
    ref: str | None = None
    conversation_id: str | None = None


class QueryResponse(BaseModel):
    """Structured API response containing the generated RAG response."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str
    correlation_id: str
    result: RAGResponse


class HealthResponse(BaseModel):
    """Basic service health response."""

    model_config = ConfigDict(frozen=True)

    status: str


class StatusResponse(BaseModel):
    """API and query-service status response."""

    model_config = ConfigDict(frozen=True)

    status: str
    query_handler_configured: bool


class QueryHandler(Protocol):
    """Protocol for the application query orchestration boundary."""

    def __call__(self,request: QueryRequest) -> RAGResponse:
        """Process a structured query request."""
        ...


class APIQueryService:
    """Adapter around the injected application query handler.

    This class keeps the HTTP layer independent from the concrete
    query orchestration implementation. Task 30 can inject the
    complete RAG pipeline without changing the API contract.
    """

    def __init__(self,handler: QueryHandler | None = None) -> None:
        if handler is not None and not callable(handler):
            raise TypeError(
                "handler must be callable or None."
            )

        self._handler = handler

    @property
    def configured(self) -> bool:
        """Return whether a query handler has been configured."""
        return self._handler is not None

    def execute(self,request: QueryRequest) -> RAGResponse:
        """Execute a query through the configured application handler."""
        if not isinstance(request,QueryRequest):
            raise TypeError(
                "request must be a QueryRequest."
            )

        if self._handler is None:
            raise RuntimeError(
                "Query service is not configured."
            )

        try:
            result = self._handler(request)
        except Exception as exc:
            raise RuntimeError(
                "Query processing failed."
            ) from exc

        if not isinstance(result,RAGResponse):
            raise RuntimeError(
                "Query handler must return a RAGResponse."
            )

        return result


def create_app(query_service: APIQueryService | None = None) -> FastAPI:
    """Create the FastAPI application.

    A query service may be injected for production orchestration
    or deterministic testing. No repository, retrieval, or LLM
    dependencies are created by the API layer.
    """
    service = (
        query_service
        if query_service is not None
        else APIQueryService()
    )

    app = FastAPI(
        title="Codebase Intelligence System",
        version="1.0.0",
    )

    app.state.query_service = service

    @app.post(
        "/query",
        response_model=QueryResponse,
    )
    def query(request: QueryRequest) -> QueryResponse:
        """Submit a developer question to the RAG query service."""
        conversation_id = (
            request.conversation_id
            or str(uuid4())
        )
        correlation_id = str(uuid4())

        try:
            result = service.execute(request)
        except RuntimeError as exc:
            if str(exc) == "Query service is not configured.":
                raise HTTPException(
                    status_code=503,
                    detail="Query service is not configured.",
                ) from exc

            raise HTTPException(
                status_code=500,
                detail="Query processing failed.",
            ) from exc

        return QueryResponse(
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            result=result,
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
    )
    def health() -> HealthResponse:
        """Return basic API liveness."""
        return HealthResponse(
            status="ok",
        )

    @app.get(
        "/status",
        response_model=StatusResponse,
    )
    def status() -> StatusResponse:
        """Return API and query-service configuration status."""
        return StatusResponse(
            status="ok",
            query_handler_configured=service.configured,
        )

    return app


app = create_app()