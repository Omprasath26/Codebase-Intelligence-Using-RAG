"""Citation, provenance, and grounding validation for RAG responses."""

from __future__ import annotations
import re
from dataclasses import dataclass
from src.context_assembly import EvidencePackage
from src.generation import DraftResponse, RAGResponse, SourceReference


@dataclass(frozen=True)
class ValidationResult:
    """Deterministic validation result for a draft response."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    cited_source_ids: tuple[str, ...]
    grounding_score: float


class ResponseValidator:
    """Validate generated responses against retrieved repository evidence.

    This validator does not generate answers and does not perform retrieval.
    It verifies that a DraftResponse remains traceable to the EvidencePackage
    from which it was generated.
    """

    _TOKEN_PATTERN = re.compile(
        r"[A-Za-z_][A-Za-z0-9_./:-]{2,}"
    )

    _STOPWORDS = frozenset(
        {
            "the",
            "and",
            "that",
            "this",
            "with",
            "from",
            "into",
            "for",
            "are",
            "was",
            "were",
            "has",
            "have",
            "does",
            "how",
            "what",
            "where",
            "when",
            "which",
            "their",
            "there",
            "then",
            "than",
            "also",
            "only",
            "using",
            "used",
            "uses",
            "can",
            "will",
            "its",
            "about",
            "through",
            "repository",
            "code",
        }
    )

    def __init__(self,*,minimum_grounding_score: float = 0.20) -> None:
        """Configure the deterministic grounding threshold."""
        if not isinstance( minimum_grounding_score,(int, float)):
            raise TypeError(
                "minimum_grounding_score must be numeric."
            )

        if not 0.0 <= minimum_grounding_score <= 1.0:
            raise ValueError(
                "minimum_grounding_score must be between 0.0 and 1.0."
            )

        self.minimum_grounding_score = float(
            minimum_grounding_score
        )

    def validate(self,draft: DraftResponse,evidence_package: EvidencePackage) -> ValidationResult:
        """Validate a draft response against its evidence package."""
        if not isinstance(draft,DraftResponse):
            raise TypeError(
                "draft must be a DraftResponse."
            )

        if not isinstance(evidence_package,EvidencePackage):
            raise TypeError(
                "evidence_package must be an EvidencePackage."
            )

        errors: list[str] = []
        warnings: list[str] = []

        evidence_by_id = {
            block.chunk.stable_id: block
            for block in evidence_package.context_blocks
        }

        cited_source_ids = tuple(
            source.stable_id
            for source in draft.sources
        )

        self._validate_citations(
            draft.sources,
            evidence_by_id,
            errors,
        )

        self._validate_revision_consistency(
            draft.sources,
            evidence_by_id,
            evidence_package,
            errors,
        )

        grounding_score = self._grounding_score(
            draft.answer,
            evidence_package,
            draft.sources,
        )

        self._validate_grounding(
            draft,
            evidence_package,
            grounding_score,
            errors,
            warnings,
        )

        self._validate_limitation(
            draft,
            evidence_package,
            errors,
            warnings,
        )

        return ValidationResult(
            valid=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
            cited_source_ids=cited_source_ids,
            grounding_score=grounding_score,
        )

    def build_response(self,draft: DraftResponse,evidence_package: EvidencePackage) -> RAGResponse:
        """Validate a draft and convert it into a RAGResponse."""
        result = self.validate(
            draft,
            evidence_package,
        )

        evidence_metadata = {
            "validation_valid": result.valid,
            "grounding_score": result.grounding_score,
            "validation_errors": list(result.errors),
            "validation_warnings": list(result.warnings),
            "cited_source_ids": list(
                result.cited_source_ids
            ),
        }

        limitation = draft.limitation

        if not result.valid:
            validation_message = (
                "Response failed grounding validation: "
                + "; ".join(result.errors)
            )

            if limitation:
                limitation = (
                    f"{limitation} {validation_message}"
                )
            else:
                limitation = validation_message

        return RAGResponse(
            answer=draft.answer,
            sources=list(draft.sources),
            confidence=draft.confidence,
            limitation=limitation,
            repository_revision=(
                evidence_package.repository_revision
            ),
            evidence_metadata=evidence_metadata,
            validated=result.valid,
        )

    def _validate_citations(self,sources: list[SourceReference],evidence_by_id: dict[str, object],errors: list[str]) -> None:
        """Ensure every cited source exists in retrieved evidence."""
        seen: set[str] = set()

        for source in sources:
            if source.stable_id in seen:
                errors.append(
                    "Duplicate citation: "
                    f"{source.stable_id}"
                )
                continue

            seen.add(source.stable_id)

            if source.stable_id not in evidence_by_id:
                errors.append(
                    "Citation does not map to retrieved "
                    f"evidence: {source.stable_id}"
                )
                continue

            block = evidence_by_id[source.stable_id]

            if not self._source_matches_chunk(source,block.chunk):
                errors.append(
                    "Citation provenance does not match "
                    f"retrieved chunk: {source.stable_id}"
                )

    def _source_matches_chunk(self,source: SourceReference,chunk: object) -> bool:
        """Compare source-reference provenance with a knowledge chunk."""
        fields = (
            "stable_id",
            "artifact_id",
            "repository",
            "artifact_type",
            "source_path_or_object_id",
            "source_url",
            "commit_sha",
            "ref",
            "language",
            "start_line",
            "end_line",
            "symbol",
            "parent_symbol",
        )

        return all(
            getattr(source, field)
            == getattr(chunk, field)
            for field in fields
        )

    def _validate_revision_consistency(self,sources: list[SourceReference],evidence_by_id: dict[str, object],evidence_package: EvidencePackage,errors: list[str]) -> None:
        """Ensure cited evidence belongs to the package revision."""
        revision = evidence_package.repository_revision

        if not revision:
            return

        for source in sources:
            block = evidence_by_id.get(source.stable_id)

            if block is None:
                continue

            commit_sha = block.chunk.commit_sha

            if (commit_sha is not None and commit_sha != revision):
                errors.append(
                    "Citation revision does not match "
                    f"evidence package revision: "
                    f"{source.stable_id}"
                )

    def _grounding_score(self,answer: str,evidence_package: EvidencePackage,sources: list[SourceReference]) -> float:
        """Calculate conservative lexical grounding against cited evidence."""
        if not answer.strip():
            return 0.0

        cited_ids = {
            source.stable_id
            for source in sources
        }

        evidence_text_parts: list[str] = []

        for block in evidence_package.context_blocks:
            if (
                not cited_ids
                or block.chunk.stable_id in cited_ids
            ):
                evidence_text_parts.append(
                    block.chunk.content
                )

        if not evidence_text_parts:
            return 0.0

        answer_terms = self._meaningful_terms(answer)

        if not answer_terms:
            return 0.0

        evidence_terms = self._meaningful_terms(
            " ".join(evidence_text_parts)
        )

        if not evidence_terms:
            return 0.0

        overlap = answer_terms.intersection(
            evidence_terms
        )

        return len(overlap) / len(answer_terms)

    def _validate_grounding(self,draft: DraftResponse,evidence_package: EvidencePackage,grounding_score: float,errors: list[str],warnings: list[str]) -> None:
        """Apply minimum evidence and citation grounding rules."""
        if not evidence_package.context_blocks:
            if draft.sources:
                errors.append(
                    "Response contains citations but "
                    "the evidence package is empty."
                )

            if not draft.limitation:
                errors.append(
                    "No retrieved evidence is available and "
                    "the response does not state a limitation."
                )

            return

        if not draft.sources:
            errors.append(
                "Repository-specific response has no "
                "source citations."
            )
            return

        if grounding_score < self.minimum_grounding_score:
            errors.append(
                "Answer grounding score "
                f"{grounding_score:.3f} is below the "
                f"minimum required "
                f"{self.minimum_grounding_score:.3f}."
            )

        if grounding_score < 0.50:
            warnings.append(
                "Answer has limited lexical overlap with "
                "the cited evidence; semantic grounding "
                "should be reviewed."
            )

    def _validate_limitation(self,draft: DraftResponse,evidence_package: EvidencePackage,errors: list[str],warnings: list[str]) -> None:
        """Ensure insufficient evidence is surfaced explicitly."""
        if not evidence_package.sufficient:
            if not draft.limitation:
                errors.append(
                    "Evidence package is insufficient but "
                    "the response does not state a limitation."
                )
            return

        if draft.limitation:
            warnings.append(
                "Response contains a limitation despite "
                "the evidence package being marked sufficient."
            )

    def _meaningful_terms(self,text: str) -> set[str]:
        """Extract deterministic terms useful for grounding comparison."""
        terms: set[str] = set()

        for token in self._TOKEN_PATTERN.findall(text):
            normalized = token.lower()

            if normalized in self._STOPWORDS:
                continue

            if len(normalized) < 3:
                continue

            terms.add(normalized)

        return terms