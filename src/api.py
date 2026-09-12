"""FastAPI query API and end-to-end RAG query orchestration."""

from __future__ import annotations

from typing import Any, Protocol
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.context_assembly import ContextAssembly
from src.conversation import ConversationState
from src.generation import (DraftResponse,GenerationInput,PromptBuilder,RAGResponse)
from src.llm import LLM
from src.query_analysis import QueryAnalysis, QueryAnalyzer
from src.retrieval import Retrieval
from src.validation import ResponseValidator


class QueryRequest(BaseModel):
    """Structured request submitted to the developer QA API."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1)
    repository: str | None = None
    ref: str | None = None
    conversation_id: str | None = None


class SyncRequest(BaseModel):
    """Structured request submitted to the manual synchronization API."""

    model_config = ConfigDict(frozen=True)

    repository: str = Field(min_length=1)
    ref: str = Field(min_length=1)


class SyncResponse(BaseModel):
    """Structured response returned by the manual synchronization API."""

    model_config = ConfigDict(frozen=True)

    correlation_id: str
    repository: str
    ref: str
    result: Any


class SyncHandler(Protocol):
    """Protocol for the manual synchronization boundary."""

    def __call__(self, request: SyncRequest) -> Any:
        """Execute one repository synchronization."""
        ...


class APISyncService:
    """Adapter around the injected repository synchronization callable."""

    def __init__(self, handler: SyncHandler | None = None) -> None:
        if handler is not None and not callable(handler):
            raise TypeError("handler must be callable or None.")
        self._handler = handler

    @property
    def configured(self) -> bool:
        """Return whether a synchronization handler is configured."""
        return self._handler is not None

    def execute(self, request: SyncRequest) -> Any:
        """Execute one manual synchronization."""
        if not isinstance(request, SyncRequest):
            raise TypeError("request must be a SyncRequest.")
        if self._handler is None:
            raise RuntimeError("Sync service is not configured.")
        try:
            return self._handler(request)
        except Exception as exc:
            raise RuntimeError("Synchronization failed.") from exc


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

    def __call__(self, request: QueryRequest) -> RAGResponse:
        """Process a structured query request."""
        ...


class APIQueryService:
    """Adapter around the injected application query handler.

    This class keeps the HTTP layer independent from the concrete
    query orchestration implementation.
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
        if not isinstance(request, QueryRequest):
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

        if not isinstance(result, RAGResponse):
            raise RuntimeError(
                "Query handler must return a RAGResponse."
            )

        return result


class RAGQueryPipeline:
    """Execute the complete grounded developer QA pipeline.

    Pipeline:

        QueryRequest
            -> QueryAnalyzer
            -> Retrieval
            -> ContextAssembly
            -> PromptBuilder
            -> LLM
            -> ResponseValidator
            -> RAGResponse

    Conversation state is maintained at this orchestration boundary.
    The individual domain modules remain responsible for their own
    processing stages.
    """

    def __init__(self,retrieval: Retrieval,context_assembly: ContextAssembly,prompt_builder: PromptBuilder,llm: LLM,validator: ResponseValidator,query_analyzer: QueryAnalyzer | None = None) -> None:
        if not hasattr(retrieval, "retrieve"):
            raise TypeError(
                "retrieval must implement retrieve()."
            )

        if not isinstance(context_assembly,ContextAssembly):
            raise TypeError(
                "context_assembly must be a ContextAssembly."
            )

        if not isinstance(prompt_builder,PromptBuilder):
            raise TypeError(
                "prompt_builder must be a PromptBuilder."
            )

        if not isinstance(llm, LLM):
            raise TypeError(
                "llm must be an LLM."
            )

        if not isinstance(validator,ResponseValidator):
            raise TypeError(
                "validator must be a ResponseValidator."
            )

        if query_analyzer is not None and not isinstance(query_analyzer,QueryAnalyzer):
            raise TypeError(
                "query_analyzer must be a QueryAnalyzer or None."
            )

        self.retrieval = retrieval
        self.context_assembly = context_assembly
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.validator = validator
        self.query_analyzer = (
            query_analyzer
            if query_analyzer is not None
            else QueryAnalyzer()
        )

        self._conversations: dict[
            str,
            ConversationState,
        ] = {}

    def __call__(self,request: QueryRequest) -> RAGResponse:
        """Allow the pipeline to be used directly as an API handler."""
        return self.run(request)

    def run(self,request: QueryRequest) -> RAGResponse:
        """Run one complete grounded query."""
        if not isinstance(request, QueryRequest):
            raise TypeError(
                "request must be a QueryRequest."
            )

        conversation_id = (
            request.conversation_id
            or str(uuid4())
        )

        state = self._get_or_create_conversation(
            conversation_id=conversation_id,
            repository=request.repository,
            ref=request.ref,
        )

        current_analysis = self.query_analyzer.analyze(
            request.query
        )

        effective_analysis = self._resolve_follow_up_analysis(
            current_analysis=current_analysis,
            previous_analysis=state.previous_analysis,
        )

        # Capture previous conversation context BEFORE adding the
        # current turn. This is important because the current turn
        # does not yet have a generated response.
        conversation_context = (
            self._conversation_prompt_context(state)
        )

        state = state.add_turn(
            query=request.query,
            analysis=effective_analysis,
        )

        metadata_filters = self._build_metadata_filters(
            repository=request.repository,
            ref=request.ref,
        )

        evidence = self.retrieval.retrieve(
            effective_analysis,
            metadata_filters=metadata_filters,
        )

        evidence_package = self.context_assembly.assemble(
            query=request.query,
            evidence=evidence,
            query_analysis=effective_analysis,
        )

        state = state.update_evidence(
            evidence_package
        )

        generation_input = GenerationInput(
            query=request.query,
            evidence_package=evidence_package,
        )

        prompt = self.prompt_builder.build(
            generation_input=generation_input,
            conversation_context=conversation_context,
        )

        draft = self.llm.generate(prompt)

        response = self.validator.build_response(
            draft=draft,
            evidence_package=evidence_package,
        )

        # Evidence confidence belongs to the evidence/context stage,
        # while the draft confidence belongs to the generation stage.
        # Expose the deterministic evidence confidence explicitly.
        evidence_metadata = dict(
            response.evidence_metadata
        )

        evidence_metadata.update(
            {
                "evidence_confidence": (
                    evidence_package.confidence
                ),
                "refinement_required": (
                    evidence_package.refinement_required
                ),
            }
        )

        response = response.model_copy(
            update={
                "evidence_metadata": evidence_metadata,
            }
        )

        state = state.update_response(
            response
        )

        self._conversations[
            conversation_id
        ] = state

        return response

    def get_conversation(self,conversation_id: str) -> ConversationState | None:
        """Return stored conversation state when available."""
        return self._conversations.get(
            conversation_id
        )

    def reset_conversation(self,conversation_id: str) -> ConversationState:
        """Reset one conversation and return its fresh state."""
        state = self._conversations.get(
            conversation_id
        )

        if state is None:
            new_state = ConversationState(
                conversation_id=str(uuid4())
            )
        else:
            new_state = state.reset()

        self._conversations[
            conversation_id
        ] = new_state

        return new_state

    def _get_or_create_conversation(self,conversation_id: str,repository: str | None,ref: str | None) -> ConversationState:
        """Return existing state or create repository-scoped state."""
        state = self._conversations.get(
            conversation_id
        )

        if state is None:
            state = ConversationState(
                conversation_id=conversation_id,
                repository=repository,
                ref=ref,
            )
            self._conversations[
                conversation_id
            ] = state
            return state

        if repository is not None:
            state = state.with_repository_scope(
                repository=repository,
                ref=ref,
            )

            self._conversations[
                conversation_id
            ] = state

        return state

    def _resolve_follow_up_analysis(self,current_analysis: QueryAnalysis,previous_analysis: QueryAnalysis | None) -> QueryAnalysis:
        """Preserve prior entities for underspecified follow-ups.

        An entity-free follow-up such as "What does it do?" should
        continue referring to the previously identified symbol,
        path, error, issue, or PR.

        Explicit entities in the new query always take precedence.
        """

        if previous_analysis is None:
            return current_analysis

        current_has_entities = any(
            (
                current_analysis.symbols,
                current_analysis.paths,
                current_analysis.error_messages,
                current_analysis.issue_numbers,
                current_analysis.pr_numbers,
            )
        )

        if current_has_entities:
            return current_analysis

        return current_analysis.model_copy(
            update={
                "symbols": list(
                    previous_analysis.symbols
                ),
                "paths": list(
                    previous_analysis.paths
                ),
                "error_messages": list(
                    previous_analysis.error_messages
                ),
                "issue_numbers": list(
                    previous_analysis.issue_numbers
                ),
                "pr_numbers": list(
                    previous_analysis.pr_numbers
                ),
            }
        )

    def _conversation_prompt_context(self,state: ConversationState) -> list[str]:
        """Build deterministic prompt context from prior turns."""
        context: list[str] = []

        for turn in state.turns:
            context.append(
                f"Previous developer question: {turn.query}"
            )

            if turn.response is not None:
                context.append(
                    "Previous answer: "
                    + turn.response.answer
                )

                if turn.response.limitation:
                    context.append(
                        "Previous limitation: "
                        + turn.response.limitation
                    )

        return context

    def _build_metadata_filters(self,repository: str | None,ref: str | None) -> dict[str, str]:
        """Build retrieval scope filters from the API request."""
        filters: dict[str, str] = {}

        if repository is not None:
            filters["repository"] = repository

        if ref is not None:
            filters["ref"] = ref

        return filters


def create_app(query_service: APIQueryService | None = None,sync_service: APISyncService | None = None) -> FastAPI:
    """Create the FastAPI application.

    The query service is injected so the HTTP layer remains independent
    from retrieval, generation, and repository-processing dependencies.
    """

    service = (
        query_service
        if query_service is not None
        else APIQueryService()
    )
    synchronization_service = (
        sync_service
        if sync_service is not None
        else APISyncService()
    )

    app = FastAPI(
        title="Codebase Intelligence System",
        version="1.0.0",
    )

    app.state.query_service = service
    app.state.sync_service = synchronization_service

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

        # Preserve the Task 29 API contract while ensuring that a
        # generated conversation ID is also available to Task 30.
        effective_request = request.model_copy(
            update={
                "conversation_id": conversation_id,
            }
        )

        try:
            result = service.execute(
                effective_request
            )
        except RuntimeError as exc:
            if str(exc) == (
                "Query service is not configured."):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Query service is not configured."
                    ),
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

    @app.post(
        "/sync",
        response_model=SyncResponse,
    )
    def sync(request: SyncRequest) -> SyncResponse:
        """Trigger one manual repository synchronization."""
        correlation_id = str(uuid4())
        try:
            result = synchronization_service.execute(request)
        except RuntimeError as exc:
            if str(exc) == "Sync service is not configured.":
                raise HTTPException(
                    status_code=503,
                    detail="Sync service is not configured.",
                ) from exc
            raise HTTPException(
                status_code=500,
                detail="Synchronization failed.",
            ) from exc

        return SyncResponse(
            correlation_id=correlation_id,
            repository=request.repository,
            ref=request.ref,
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