"""Generation contracts and prompt orchestration for the RAG pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.context_assembly import EvidencePackage


class GenerationInput(BaseModel):
    """Input contract consumed by the answer-generation stage."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1)
    evidence_package: EvidencePackage


class SourceReference(BaseModel):
    """Repository provenance attached to an answer source."""

    model_config = ConfigDict(frozen=True)

    stable_id: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    source_path_or_object_id: str = Field(min_length=1)
    source_url: str | None = None
    commit_sha: str | None = None
    ref: str | None = None
    language: str | None = None
    start_line: int | None = Field(
        default=None,
        ge=1,
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
    )
    symbol: str | None = None
    parent_symbol: str | None = None


class DraftResponse(BaseModel):
    """Raw generation result before grounding validation."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)
    sources: list[SourceReference] = Field(
        default_factory=list,
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    limitation: str | None = None


class RAGResponse(BaseModel):
    """Stable response contract exposed by the RAG answer pipeline."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)
    sources: list[SourceReference] = Field(
        default_factory=list,
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    limitation: str | None = None
    repository_revision: str | None = None
    evidence_metadata: dict[str, Any] = Field(
        default_factory=dict,
    )
    validated: bool = False


class GenerationPrompt(BaseModel):
    """Structured prompt supplied to an LLM provider."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)


class PromptBuilder:
    """Build deterministic, repository-grounded generation prompts."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a codebase intelligence assistant. "
        "Answer the developer's question using only the "
        "repository evidence provided in the context and "
        "the explicitly provided conversation context. "
        "Do not invent repository facts, identifiers, files, "
        "symbols, behavior, history, or test results. "
        "Distinguish repository-supported facts from "
        "interpretation or general recommendations. "
        "Use source references for important repository claims. "
        "If the available evidence is insufficient, conflicting, "
        "or does not establish an answer, explicitly state "
        "the limitation instead of guessing. "
        "Do not claim that code was tested unless the provided "
        "evidence explicitly establishes that it was tested."
    )

    def build(self,generation_input: GenerationInput,conversation_context: list[str] | None = None) -> GenerationPrompt:
        """Build a deterministic grounded prompt."""
        if not isinstance(
            generation_input,
            GenerationInput,
        ):
            raise TypeError(
                "generation_input must be a GenerationInput."
            )

        context = conversation_context or []

        if not isinstance(context, list):
            raise TypeError(
                "conversation_context must be a list of strings."
            )

        if not all(
            isinstance(item, str)
            for item in context
        ):
            raise TypeError(
                "conversation_context must contain only strings."
            )

        return GenerationPrompt(
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(
                generation_input,
                context,
            ),
        )

    def _build_user_prompt(self,generation_input: GenerationInput,conversation_context: list[str]) -> str:
        """Format the query, evidence, revision and conversation."""
        evidence_package = (
            generation_input.evidence_package
        )

        sections: list[str] = [
            "DEVELOPER QUESTION",
            generation_input.query,
            "",
            "EVIDENCE SUFFICIENCY",
            str(evidence_package.sufficient),
            "",
            "REPOSITORY REVISION",
            (
                evidence_package.repository_revision
                or "Not available"
            ),
            "",
            "EVIDENCE",
        ]

        if not evidence_package.context_blocks:
            sections.append(
                "No repository evidence was retrieved."
            )
        else:
            for index, block in enumerate(
                evidence_package.context_blocks,
                start=1,
            ):
                sections.extend(
                    self._format_context_block(
                        index,
                        block,
                    )
                )

        if evidence_package.limitation:
            sections.extend(
                [
                    "",
                    "EVIDENCE LIMITATION",
                    evidence_package.limitation,
                ]
            )

        if conversation_context:
            sections.extend(
                [
                    "",
                    "CONVERSATION CONTEXT",
                ]
            )
            sections.extend(conversation_context)

        sections.extend(
            [
                "",
                "ANSWERING REQUIREMENTS",
                "1. Answer the developer's question directly.",
                (
                    "2. Use repository evidence as the basis "
                    "for repository-specific claims."
                ),
                (
                    "3. Reference relevant source paths, symbols, "
                    "and line ranges when available."
                ),
                (
                    "4. Clearly distinguish repository facts "
                    "from interpretation."
                ),
                "5. Do not fabricate missing information.",
                (
                    "6. If evidence is insufficient, state the "
                    "limitation explicitly."
                ),
            ]
        )

        return "\n".join(sections)

    def _format_context_block(self,rank: int,block: Any) -> list[str]:
        """Format one assembled evidence block."""
        chunk = block.chunk

        source_location = (
            chunk.source_path_or_object_id
        )

        if (
            chunk.start_line is not None
            and chunk.end_line is not None
        ):
            source_location = (
                f"{source_location}:"
                f"{chunk.start_line}-"
                f"{chunk.end_line}"
            )

        symbol = chunk.symbol or "Not available"

        return [
            "",
            f"[EVIDENCE {rank}]",
            f"Chunk ID: {chunk.stable_id}",
            f"Artifact ID: {chunk.artifact_id}",
            f"Artifact type: {chunk.artifact_type}",
            f"Repository: {chunk.repository}",
            f"Source: {source_location}",
            (
                "URL: "
                f"{chunk.source_url or 'Not available'}"
            ),
            (
                "Commit SHA: "
                f"{chunk.commit_sha or 'Not available'}"
            ),
            f"Ref: {chunk.ref or 'Not available'}",
            (
                "Language: "
                f"{chunk.language or 'Not available'}"
            ),
            f"Symbol: {symbol}",
            (
                "Parent symbol: "
                f"{chunk.parent_symbol or 'Not available'}"
            ),
            (
                "Retrieval method: "
                f"{block.retrieval_method}"
            ),
            f"Retrieval score: {block.score}",
            (
                "Expansion type: "
                f"{block.expansion_type}"
            ),
            "Content:",
            chunk.content,
        ]