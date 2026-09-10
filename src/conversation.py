"""Conversation state management for developer question answering."""

from __future__ import annotations
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field
from src.context_assembly import EvidencePackage
from src.generation import RAGResponse
from src.query_analysis import QueryAnalysis


class ConversationTurn(BaseModel):
    """Immutable record of one question-answer interaction."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1)
    analysis: QueryAnalysis
    response: RAGResponse | None = None


class ConversationState(BaseModel):
    """State for one repository-scoped developer conversation.

    The state preserves the conversational information required to
    interpret follow-up questions without performing retrieval,
    generation, or validation itself.
    """

    model_config = ConfigDict(frozen=True)

    conversation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
    )
    repository: str | None = None
    ref: str | None = None
    turns: tuple[ConversationTurn, ...] = ()
    latest_evidence: EvidencePackage | None = None

    @property
    def last_turn(self) -> ConversationTurn | None:
        """Return the most recent conversation turn."""
        if not self.turns:
            return None

        return self.turns[-1]

    @property
    def previous_analysis(self) -> QueryAnalysis | None:
        """Return the most recent query analysis."""
        if self.last_turn is None:
            return None

        return self.last_turn.analysis

    @property
    def previous_response(self) -> RAGResponse | None:
        """Return the most recent generated response."""
        if self.last_turn is None:
            return None

        return self.last_turn.response

    def add_turn(self, query: str,analysis: QueryAnalysis,response: RAGResponse | None = None) -> "ConversationState":
        """Return a new state containing the supplied conversation turn."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "query must be a non-empty string."
            )

        if not isinstance(analysis,QueryAnalysis):
            raise TypeError(
                "analysis must be a QueryAnalysis."
            )

        if response is not None and not isinstance(response,RAGResponse):
            raise TypeError(
                "response must be a RAGResponse or None."
            )

        repository = self.repository

        if repository is None:
            repository = self._repository_from_analysis(
                analysis
            )

        return self.model_copy(
            update={
                "repository": repository,
                "turns": (
                    *self.turns,
                    ConversationTurn(
                        query=query.strip(),
                        analysis=analysis,
                        response=response,
                    ),
                ),
            }
        )

    def update_response(self,response: RAGResponse) -> "ConversationState":
        """Return a new state with the latest turn response updated."""
        if not isinstance(response,RAGResponse):
            raise TypeError(
                "response must be a RAGResponse."
            )

        if not self.turns:
            raise ValueError(
                "Cannot update a response without a conversation turn."
            )

        updated_turn = self.turns[-1].model_copy(
            update={
                "response": response,
            }
        )

        return self.model_copy(
            update={
                "turns": (
                    *self.turns[:-1],
                    updated_turn,
                ),
            }
        )

    def update_evidence(self,evidence_package: EvidencePackage) -> "ConversationState":
        """Return a new state using the newest retrieved evidence.

        New repository evidence replaces older evidence rather than
        being merged into it. This prevents stale repository evidence
        from remaining authoritative after a newer retrieval result.
        """
        if not isinstance(evidence_package,EvidencePackage):
            raise TypeError(
                "evidence_package must be an EvidencePackage."
            )

        repository = self.repository

        if repository is None:
            repository = self._repository_from_evidence(
                evidence_package
            )

        return self.model_copy(
            update={
                "repository": repository,
                "latest_evidence": evidence_package,
            }
        )

    def follow_up_context(self) -> dict[str, object]:
        """Return the relevant state needed for a follow-up query."""
        context: dict[str, object] = {
            "conversation_id": self.conversation_id,
            "repository": self.repository,
            "ref": self.ref,
        }

        if self.previous_analysis is not None:
            context["previous_analysis"] = (
                self.previous_analysis
            )

        if self.previous_response is not None:
            context["previous_response"] = (
                self.previous_response
            )

        if self.latest_evidence is not None:
            context["latest_evidence"] = (
                self.latest_evidence
            )

        return context

    def reset(self) -> "ConversationState":
        """Return a fresh conversation state with a new conversation ID."""
        return ConversationState(
            repository=None,
            ref=None,
        )

    def with_repository_scope(self,repository: str,ref: str | None = None) -> "ConversationState":
        """Return a state explicitly scoped to a repository and ref."""
        if not isinstance(
            repository,
            str,
        ) or not repository.strip():
            raise ValueError(
                "repository must be a non-empty string."
            )

        return self.model_copy(
            update={
                "repository": repository.strip(),
                "ref": ref,
            }
        )

    @staticmethod
    def _repository_from_analysis(analysis: QueryAnalysis) -> str | None:
        """Extract repository scope when represented by the analysis.

        QueryAnalysis currently does not have a dedicated repository
        field, so repository scope is normally supplied explicitly by
        the conversation/API orchestration layer.
        """
        return None

    @staticmethod
    def _repository_from_evidence(evidence_package: EvidencePackage) -> str | None:
        """Extract repository scope from retrieved evidence."""
        for block in evidence_package.context_blocks:
            return block.chunk.repository

        return None