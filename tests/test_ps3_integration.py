"""End-to-end integration tests for the PS3 grounded QA pipeline."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from fastapi.testclient import TestClient
from src.api import (APIQueryService,QueryRequest,RAGQueryPipeline,create_app)
from src.code_processing import KnowledgeChunk
from src.context_assembly import ContextAssembly
from src.generation import (DraftResponse,PromptBuilder,RAGResponse,SourceReference)
from src.llm import (FakeLLMProvider,LLM,LLMConfig)
from src.retrieval import RetrievedEvidence
from src.validation import ResponseValidator


@dataclass
class StubRetrieval:
    """Deterministic retrieval boundary for integration tests."""

    evidence: list[RetrievedEvidence]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(self,query_analysis,top_k: int = 5,candidate_k: int | None = None,metadata_filters: dict[str, Any] | None = None,rerank: bool = True,rerank_k: int | None = None) -> list[RetrievedEvidence]:
        self.calls.append(
            {
                "query_analysis": query_analysis,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "metadata_filters": metadata_filters,
                "rerank": rerank,
                "rerank_k": rerank_k,
            }
        )

        return list(self.evidence)


def make_chunk(stable_id: str = "chunk-request",symbol: str = "Request",content: str | None = None,commit_sha: str = "abc123") -> KnowledgeChunk:
    """Create deterministic repository evidence."""
    return KnowledgeChunk(
        stable_id=stable_id,
        artifact_id="artifact-request",
        repository="scrapy/scrapy",
        artifact_type="code",
        content=(
            content
            if content is not None
            else (
                "class Request: "
                "The Request class represents an HTTP request."
            )
        ),
        source_path_or_object_id="scrapy/http/request.py",
        source_url=(
            "https://github.com/scrapy/scrapy/"
            "blob/abc123/scrapy/http/request.py"
        ),
        commit_sha=commit_sha,
        ref="master",
        language="python",
        start_line=10,
        end_line=40,
        module="scrapy.http.request",
        symbol=symbol,
        parent_symbol=None,
        metadata={
            "symbol_type": "class",
            "module": "scrapy.http.request",
            "package": "scrapy.http",
        },
    )


def make_retrieved_evidence(chunk: KnowledgeChunk,score: float = 1.0) -> RetrievedEvidence:
    """Wrap a chunk in deterministic retrieval metadata."""
    return RetrievedEvidence(
        chunk=chunk,
        score=score,
        retrieval_method="dense+bm25",
        dense_score=0.9,
        bm25_score=1.0,
        rerank_score=None,
    )


def make_source(chunk: KnowledgeChunk) -> SourceReference:
    """Create a source reference matching a knowledge chunk."""
    return SourceReference(
        stable_id=chunk.stable_id,
        artifact_id=chunk.artifact_id,
        repository=chunk.repository,
        artifact_type=chunk.artifact_type,
        source_path_or_object_id=(
            chunk.source_path_or_object_id),
        source_url=chunk.source_url,
        commit_sha=chunk.commit_sha,
        ref=chunk.ref,
        language=chunk.language,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        symbol=chunk.symbol,
        parent_symbol=chunk.parent_symbol,
    )


def make_pipeline(chunk: KnowledgeChunk | None = None,llm_provider: FakeLLMProvider | None = None) -> tuple[RAGQueryPipeline,StubRetrieval,FakeLLMProvider]:
    """Build a deterministic end-to-end pipeline."""
    chunk = chunk or make_chunk()

    evidence = [
        make_retrieved_evidence(chunk)
    ]

    retrieval = StubRetrieval(
        evidence=evidence
    )

    context_assembly = ContextAssembly(
        chunks=[chunk],
        max_context_characters=12000,
        neighbor_limit=1,
    )

    source = make_source(chunk)

    provider = llm_provider or FakeLLMProvider(
        response=DraftResponse(
            answer=(
                "The Request class is implemented in "
                "scrapy/http/request.py."
            ),
            sources=[source],
            confidence=0.9,
            limitation=None,
        )
    )

    llm = LLM(
        config=LLMConfig(
            provider="fake",
            model="deterministic-test",
            timeout_seconds=5.0,
        ),
        provider=provider,
    )

    pipeline = RAGQueryPipeline(
        retrieval=retrieval,
        context_assembly=context_assembly,
        prompt_builder=PromptBuilder(),
        llm=llm,
        validator=ResponseValidator(),
    )

    return (
        pipeline,
        retrieval,
        provider,
    )


def test_complete_pipeline_returns_validated_grounded_response() -> None:
    """Verify the complete RAG pipeline returns a grounded response."""
    pipeline, retrieval, provider = make_pipeline()

    response = pipeline.run(
        QueryRequest(
            query="Where is the Request class implemented?",
            repository="scrapy/scrapy",
            ref="master",
            conversation_id="integration-conversation",
        )
    )

    assert isinstance(
        response,
        RAGResponse,
    )

    assert response.validated is True
    assert response.repository_revision == "abc123"

    assert len(response.sources) == 1
    assert response.sources[0].stable_id == "chunk-request"

    assert response.evidence_metadata[
        "validation_valid"
    ] is True

    # Task 27 evidence confidence:
    # base 0.50 + matching symbol 0.25 = 0.75.
    # The LLM draft confidence of 0.90 is intentionally separate.
    assert response.evidence_metadata[
        "evidence_confidence"
    ] == 0.75

    assert response.evidence_metadata[
        "refinement_required"
    ] is False

    assert len(retrieval.calls) == 1
    assert provider.calls == 1


def test_grounded_prompt_contains_repository_evidence() -> None:
    """Verify the LLM receives repository evidence rather than raw query only."""
    pipeline, _, provider = make_pipeline()

    pipeline.run(
        QueryRequest(
            query="Where is the Request class implemented?",
            conversation_id="prompt-conversation",
        )
    )

    assert provider.last_prompt is not None

    prompt = provider.last_prompt

    assert "DEVELOPER QUESTION" in prompt.user_prompt
    assert "Request" in prompt.user_prompt
    assert "EVIDENCE" in prompt.user_prompt
    assert "scrapy/http/request.py" in prompt.user_prompt
    assert "Chunk ID: chunk-request" in prompt.user_prompt
    assert "Commit SHA: abc123" in prompt.user_prompt

    assert (
        "Do not invent repository facts"
        in prompt.system_prompt
    )


def test_api_executes_complete_pipeline() -> None:
    """Verify FastAPI can execute the complete pipeline."""
    pipeline, _, _ = make_pipeline()

    service = APIQueryService(
        handler=pipeline
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
            "conversation_id": "api-integration",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["conversation_id"] == (
        "api-integration"
    )

    assert body["correlation_id"]

    assert body["result"]["validated"] is True

    assert body["result"]["repository_revision"] == (
        "abc123"
    )

    assert body["result"]["sources"][0][
        "stable_id"
    ] == "chunk-request"


def test_api_propagates_generated_conversation_id() -> None:
    """Verify the API-generated conversation ID reaches the pipeline."""
    pipeline, _, _ = make_pipeline()

    service = APIQueryService(
        handler=pipeline
    )

    client = TestClient(
        create_app(service)
    )

    response = client.post(
        "/query",
        json={
            "query": "Where is the Request class implemented?",
        },
    )

    assert response.status_code == 200

    conversation_id = response.json()[
        "conversation_id"
    ]

    assert conversation_id

    stored = pipeline.get_conversation(
        conversation_id
    )

    assert stored is not None
    assert stored.conversation_id == conversation_id


def test_follow_up_preserves_previous_query_context() -> None:
    """Verify an underspecified follow-up reuses prior query entities."""
    pipeline, retrieval, provider = make_pipeline()

    first = pipeline.run(
        QueryRequest(
            query="Where is the Request class implemented?",
            conversation_id="follow-up-conversation",
        )
    )

    second = pipeline.run(
        QueryRequest(
            query="What does it do?",
            conversation_id="follow-up-conversation",
        )
    )

    assert first.validated is True
    assert second.validated is True

    assert len(retrieval.calls) == 2

    first_analysis = retrieval.calls[0][
        "query_analysis"
    ]

    second_analysis = retrieval.calls[1][
        "query_analysis"
    ]

    assert first_analysis.symbols == [
        "implemented",
        "Request",
    ]

    assert second_analysis.symbols == [
        "implemented",
        "Request",
    ]

    assert second_analysis.intent == "explanation"

    assert provider.calls == 2

    stored = pipeline.get_conversation(
        "follow-up-conversation"
    )

    assert stored is not None
    assert len(stored.turns) == 2

    assert (
        stored.turns[0].query
        == "Where is the Request class implemented?"
    )

    assert (
        stored.turns[1].query
        == "What does it do?"
    )

    assert stored.turns[0].response is not None
    assert stored.turns[1].response is not None

    assert provider.last_prompt is not None

    assert "Previous developer question" in (
        provider.last_prompt.user_prompt
    )

    assert (
        "Where is the Request class implemented?"
        in provider.last_prompt.user_prompt
    )


def test_new_evidence_replaces_previous_conversation_evidence() -> None:
    """Verify newer repository evidence replaces stale evidence."""
    first_chunk = make_chunk(
        stable_id="chunk-request-old",
        content=(
            "class Request: "
            "Old repository revision implementation."
        ),
        commit_sha="old123",
    )

    second_chunk = make_chunk(
        stable_id="chunk-request-new",
        content=(
            "class Request: "
            "New repository revision implementation."
        ),
        commit_sha="new456",
    )

    first_evidence = [
        make_retrieved_evidence(
            first_chunk
        )
    ]

    second_evidence = [
        make_retrieved_evidence(
            second_chunk
        )
    ]

    retrieval = StubRetrieval(
        evidence=first_evidence
    )

    context_assembly = ContextAssembly(
        chunks=[
            first_chunk,
            second_chunk,
        ],
        max_context_characters=12000,
        neighbor_limit=0,
    )

    first_source = make_source(
        first_chunk
    )

    provider = FakeLLMProvider(
        response=DraftResponse(
            answer=(
                "The Request class is implemented "
                "in the repository."
            ),
            sources=[first_source],
            confidence=0.9,
        )
    )

    llm = LLM(
        config=LLMConfig(
            provider="fake",
            model="deterministic-test",
            timeout_seconds=5.0,
        ),
        provider=provider,
    )

    pipeline = RAGQueryPipeline(
        retrieval=retrieval,
        context_assembly=context_assembly,
        prompt_builder=PromptBuilder(),
        llm=llm,
        validator=ResponseValidator(),
    )

    first_response = pipeline.run(
        QueryRequest(
            query="Where is the Request class implemented?",
            conversation_id="revision-conversation",
        )
    )

    assert first_response.validated is True

    first_state = pipeline.get_conversation(
        "revision-conversation"
    )

    assert first_state is not None
    assert first_state.latest_evidence is not None

    assert (
        first_state.latest_evidence.repository_revision
        == "old123"
    )

    retrieval.evidence = second_evidence

    second_source = make_source(
        second_chunk
    )

    provider.response = DraftResponse(
        answer=(
            "The Request class is implemented "
            "in the newer repository revision."
        ),
        sources=[second_source],
        confidence=0.9,
    )

    second_response = pipeline.run(
        QueryRequest(
            query="Where is the Request class implemented?",
            conversation_id="revision-conversation",
        )
    )

    assert second_response.validated is True

    second_state = pipeline.get_conversation(
        "revision-conversation"
    )

    assert second_state is not None
    assert second_state.latest_evidence is not None

    assert (
        second_state.latest_evidence.repository_revision
        == "new456"
    )

    assert (
        second_state.latest_evidence.context_blocks[0]
        .chunk.stable_id
        == "chunk-request-new"
    )


def test_insufficient_evidence_produces_explicit_limitation() -> None:
    """Verify insufficient evidence is never silently presented as fact."""
    wrong_chunk = make_chunk(
        stable_id="chunk-other",
        symbol="OtherClass",
        content=(
            "class OtherClass: "
            "This class is unrelated to the requested symbol."
        ),
    )

    source = make_source(
        wrong_chunk
    )

    provider = FakeLLMProvider(
        response=DraftResponse(
            answer=(
                "The OtherClass class is unrelated to "
                "the requested symbol, so the Request "
                "implementation is not established."
            ),
            sources=[source],
            limitation=(
                "The retrieved evidence does not contain "
                "the requested Request symbol."
            ),
        )
    )

    pipeline, _, _ = make_pipeline(
        chunk=wrong_chunk,
        llm_provider=provider,
    )

    response = pipeline.run(
        QueryRequest(
            query="Where is the Request class implemented?",
            conversation_id="insufficient-conversation",
        )
    )

    assert response.validated is True

    assert response.limitation is not None

    assert (
        "requested Request symbol"
        in response.limitation
    )

    assert response.evidence_metadata[
        "evidence_confidence"
    ] == 0.0

    assert response.evidence_metadata[
        "refinement_required"
    ] is True

    assert response.sources[0].stable_id == (
        "chunk-other"
    )