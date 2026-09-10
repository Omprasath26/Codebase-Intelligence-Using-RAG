"""Tests for generation contracts and prompt orchestration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.code_processing import KnowledgeChunk
from src.context_assembly import (
    ContextBlock,
    EvidencePackage,
)
from src.generation import (
    DraftResponse,
    GenerationInput,
    GenerationPrompt,
    PromptBuilder,
    RAGResponse,
    SourceReference,
)


def make_chunk() -> KnowledgeChunk:
    """Create deterministic repository evidence for tests."""
    return KnowledgeChunk(
        stable_id="chunk-1",
        artifact_id="artifact-1",
        repository="scrapy/scrapy",
        artifact_type="code",
        content=(
            "class Request:\n"
            "    pass"
        ),
        source_path_or_object_id=(
            "scrapy/http/request.py"
        ),
        source_url=(
            "https://github.com/scrapy/scrapy/"
            "blob/abc123/scrapy/http/request.py"
        ),
        commit_sha="abc123",
        ref="master",
        language="python",
        start_line=10,
        end_line=11,
        module="scrapy.http.request",
        symbol="Request",
        parent_symbol=None,
        metadata={},
    )


def make_context_block() -> ContextBlock:
    """Create deterministic context assembly evidence."""
    return ContextBlock(
        chunk=make_chunk(),
        score=0.95,
        retrieval_method="hybrid",
        dense_score=0.90,
        bm25_score=0.80,
        rerank_score=None,
        expansion_type="retrieved",
    )


def make_evidence_package() -> EvidencePackage:
    """Create deterministic evidence for generation tests."""
    chunk = make_chunk()

    return EvidencePackage(
        query="Where is the Request class implemented?",
        context_blocks=[
            make_context_block(),
        ],
        sufficient=True,
        limitation=None,
        repository_revision="abc123",
        total_characters=len(chunk.content),
    )


def make_empty_evidence_package() -> EvidencePackage:
    """Create an explicit insufficient-evidence package."""
    return EvidencePackage(
        query="Why does this behavior happen?",
        context_blocks=[],
        sufficient=False,
        limitation=(
            "No relevant repository evidence was found."
        ),
        repository_revision="abc123",
        total_characters=0,
    )


def test_generation_input_accepts_evidence_package() -> None:
    """GenerationInput must accept the context assembly contract."""
    evidence_package = make_evidence_package()

    generation_input = GenerationInput(
        query="Explain Request.",
        evidence_package=evidence_package,
    )

    assert generation_input.query == "Explain Request."
    assert (
        generation_input.evidence_package
        == evidence_package
    )


def test_generation_input_rejects_empty_query() -> None:
    """GenerationInput must reject an empty query."""
    with pytest.raises(ValidationError):
        GenerationInput(
            query="",
            evidence_package=make_evidence_package(),
        )


def test_source_reference_preserves_provenance() -> None:
    """SourceReference must retain repository provenance."""
    source = SourceReference(
        stable_id="chunk-1",
        artifact_id="artifact-1",
        repository="scrapy/scrapy",
        artifact_type="code",
        source_path_or_object_id=(
            "scrapy/http/request.py"
        ),
        source_url=( "https://github.com/scrapy/scrapy"),
        commit_sha="abc123",
        ref="master",
        language="python",
        start_line=10,
        end_line=20,
        symbol="Request",
        parent_symbol=None,
    )

    assert source.stable_id == "chunk-1"
    assert source.artifact_id == "artifact-1"
    assert source.repository == "scrapy/scrapy"
    assert (
        source.source_path_or_object_id
        == "scrapy/http/request.py"
    )
    assert source.commit_sha == "abc123"
    assert source.ref == "master"
    assert source.symbol == "Request"


def test_draft_response_defaults_to_empty_sources() -> None:
    """DraftResponse should allow a generated answer without sources."""
    response = DraftResponse(
        answer="The class is implemented in request.py."
    )

    assert response.answer
    assert response.sources == []


def test_draft_response_validates_confidence() -> None:
    """Confidence must remain within the defined range."""
    with pytest.raises(ValidationError):
        DraftResponse(
            answer="Answer.",
            confidence=1.5,
        )

    with pytest.raises(ValidationError):
        DraftResponse(
            answer="Answer.",
            confidence=-0.1,
        )


def test_rag_response_is_final_response_contract() -> None:
    """RAGResponse must contain the stable answer contract."""
    source = SourceReference(
        stable_id="chunk-1",
        artifact_id="artifact-1",
        repository="scrapy/scrapy",
        artifact_type="code",
        source_path_or_object_id=(
            "scrapy/http/request.py"
        ),
    )

    response = RAGResponse(
        answer="Request is implemented in request.py.",
        sources=[source],
        confidence=0.9,
        repository_revision="abc123",
        evidence_metadata={
            "retrieval_method": "hybrid",
        },
        validated=False,
    )

    assert response.answer
    assert len(response.sources) == 1
    assert response.confidence == 0.9
    assert response.repository_revision == "abc123"
    assert (
        response.evidence_metadata["retrieval_method"]
        == "hybrid"
    )
    assert response.validated is False


def test_rag_response_supports_limitation() -> None:
    """RAGResponse must explicitly represent limitations."""
    response = RAGResponse(
        answer=(
            "The available evidence is insufficient "
            "to determine this."
        ),
        limitation=(
            "No relevant historical evidence was retrieved."
        ),
    )

    assert response.limitation is not None
    assert "historical" in response.limitation


def test_generation_models_are_immutable() -> None:
    """Generation contracts must be immutable."""
    generation_input = GenerationInput(
        query="Explain Request.",
        evidence_package=make_evidence_package(),
    )

    with pytest.raises(ValidationError):
        generation_input.query = "Changed"


def test_generation_serialization_is_deterministic() -> None:
    """Identical generation inputs must serialize identically."""
    evidence_package = make_evidence_package()

    first = GenerationInput(
        query="Explain Request.",
        evidence_package=evidence_package,
    )

    second = GenerationInput(
        query="Explain Request.",
        evidence_package=evidence_package,
    )

    assert first.model_dump() == second.model_dump()


def test_prompt_builder_returns_structured_prompt() -> None:
    """PromptBuilder must produce a structured generation prompt."""
    generation_input = GenerationInput(
        query="Where is the Request class implemented?",
        evidence_package=make_evidence_package(),
    )

    prompt = PromptBuilder().build(
        generation_input
    )

    assert isinstance(prompt, GenerationPrompt)
    assert prompt.system_prompt
    assert prompt.user_prompt


def test_prompt_contains_developer_question() -> None:
    """The developer question must reach the generation prompt."""
    generation_input = GenerationInput(
        query="Where is the Request class implemented?",
        evidence_package=make_evidence_package(),
    )

    prompt = PromptBuilder().build(
        generation_input
    )

    assert (
        "Where is the Request class implemented?"
        in prompt.user_prompt
    )


def test_prompt_contains_evidence_provenance() -> None:
    """Repository provenance must remain attached to evidence."""
    generation_input = GenerationInput(
        query="Where is the Request class implemented?",
        evidence_package=make_evidence_package(),
    )

    prompt = PromptBuilder().build(
        generation_input
    )

    assert "chunk-1" in prompt.user_prompt
    assert "artifact-1" in prompt.user_prompt
    assert "scrapy/scrapy" in prompt.user_prompt
    assert "scrapy/http/request.py" in prompt.user_prompt
    assert "abc123" in prompt.user_prompt
    assert "Request" in prompt.user_prompt
    assert "10-11" in prompt.user_prompt


def test_prompt_contains_grounding_rules() -> None:
    """The prompt must explicitly prohibit unsupported claims."""
    generation_input = GenerationInput(
        query="Explain Request.",
        evidence_package=make_evidence_package(),
    )

    prompt = PromptBuilder().build(
        generation_input
    )

    assert (
        "Do not invent repository facts"
        in prompt.system_prompt
    )
    assert (
        "fabricate missing information"
        in prompt.user_prompt
    )
    assert (
        "state the limitation explicitly"
        in prompt.user_prompt
    )


def test_prompt_contains_revision() -> None:
    """The repository revision must reach the LLM prompt."""
    generation_input = GenerationInput(
        query="Explain Request.",
        evidence_package=make_evidence_package(),
    )

    prompt = PromptBuilder().build(
        generation_input
    )

    assert "REPOSITORY REVISION" in prompt.user_prompt
    assert "abc123" in prompt.user_prompt


def test_prompt_contains_retrieval_metadata() -> None:
    """Retrieval metadata must remain visible to generation."""
    generation_input = GenerationInput(
        query="Explain Request.",
        evidence_package=make_evidence_package(),
    )

    prompt = PromptBuilder().build(
        generation_input
    )

    assert "Retrieval method: hybrid" in (
        prompt.user_prompt
    )
    assert "Retrieval score: 0.95" in (
        prompt.user_prompt
    )
    assert "Expansion type: retrieved" in (
        prompt.user_prompt
    )


def test_prompt_represents_empty_evidence() -> None:
    """Missing evidence must be explicitly represented."""
    generation_input = GenerationInput(
        query="Why does this behavior happen?",
        evidence_package=(
            make_empty_evidence_package()
        ),
    )

    prompt = PromptBuilder().build(
        generation_input
    )

    assert (
        "No repository evidence was retrieved."
        in prompt.user_prompt
    )
    assert (
        "No relevant repository evidence was found."
        in prompt.user_prompt
    )
    assert "EVIDENCE SUFFICIENCY" in prompt.user_prompt
    assert "False" in prompt.user_prompt


def test_prompt_includes_conversation_context() -> None:
    """Explicitly supplied conversation context must be preserved."""
    generation_input = GenerationInput(
        query="What does it do?",
        evidence_package=make_evidence_package(),
    )

    prompt = PromptBuilder().build(
        generation_input,
        conversation_context=[
            "Previous question: Explain the Request class.",
            (
                "Previous answer identified Request "
                "in request.py."
            ),
        ],
    )

    assert "CONVERSATION CONTEXT" in prompt.user_prompt
    assert (
        "Previous question: Explain the Request class."
        in prompt.user_prompt
    )
    assert (
        "Previous answer identified Request in request.py."
        in prompt.user_prompt
    )


def test_prompt_without_conversation_context_is_valid() -> None:
    """Conversation context is optional."""
    generation_input = GenerationInput(
        query="Explain Request.",
        evidence_package=make_evidence_package(),
    )

    prompt = PromptBuilder().build(
        generation_input
    )

    assert "CONVERSATION CONTEXT" not in (
        prompt.user_prompt
    )


def test_prompt_builder_is_deterministic() -> None:
    """Identical input must produce identical prompt output."""
    generation_input = GenerationInput(
        query="Explain Request.",
        evidence_package=make_evidence_package(),
    )

    builder = PromptBuilder()

    first = builder.build(
        generation_input,
        conversation_context=["Earlier context."],
    )

    second = builder.build(
        generation_input,
        conversation_context=["Earlier context."],
    )

    assert first == second


def test_prompt_builder_rejects_invalid_generation_input() -> None:
    """PromptBuilder must enforce its generation boundary."""
    with pytest.raises(
        TypeError,
        match="GenerationInput",
    ):
        PromptBuilder().build(
            "not generation input",  # type: ignore[arg-type]
        )


def test_prompt_builder_rejects_invalid_conversation_context() -> None:
    """Conversation context must contain only strings."""
    generation_input = GenerationInput(
        query="Explain Request.",
        evidence_package=make_evidence_package(),
    )

    with pytest.raises(
        TypeError,
        match="strings"):
        PromptBuilder().build(
            generation_input,
            conversation_context=[
                123,  # type: ignore[list-item]
            ],
        )


def test_prompt_preserves_line_range_in_source_location() -> None:
    """Source location should include the chunk line range."""
    generation_input = GenerationInput(
        query="Where is Request?",
        evidence_package=make_evidence_package(),
    )

    prompt = PromptBuilder().build(
        generation_input
    )

    assert (
        "Source: scrapy/http/request.py:10-11"
        in prompt.user_prompt
    )


def test_prompt_contains_answering_requirements() -> None:
    """The user prompt must contain explicit answer constraints."""
    generation_input = GenerationInput(
        query="Explain Request.",
        evidence_package=make_evidence_package(),
    )

    prompt = PromptBuilder().build(
        generation_input
    )

    assert "ANSWERING REQUIREMENTS" in (
        prompt.user_prompt
    )
    assert (
        "Answer the developer's question directly."
        in prompt.user_prompt
    )
    assert (
        "Use repository evidence as the basis"
        in prompt.user_prompt
    )
    assert (
        "Clearly distinguish repository facts"
        in prompt.user_prompt
    )