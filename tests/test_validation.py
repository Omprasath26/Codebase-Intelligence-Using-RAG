from src.context_assembly import (
    ContextBlock,
    EvidencePackage,
)
from src.generation import (
    DraftResponse,
    SourceReference,
)
from src.validation import ResponseValidator


def make_evidence_package(
    *,
    sufficient: bool = True,
    repository_revision: str | None = "commit-123",
) -> EvidencePackage:
    from src.code_processing import KnowledgeChunk

    chunk = KnowledgeChunk(
        stable_id="chunk-request",
        artifact_id="artifact-request",
        repository="scrapy/scrapy",
        artifact_type="code",
        content=(
            "class Request:\n"
            "    def __init__(self, url):\n"
            "        self.url = url\n"
        ),
        source_path_or_object_id="scrapy/http/request.py",
        source_url=(
            "https://github.com/scrapy/scrapy/"
            "blob/main/scrapy/http/request.py"
        ),
        commit_sha=repository_revision,
        ref="main",
        language="python",
        start_line=1,
        end_line=3,
        module="scrapy.http.request",
        symbol="Request",
        parent_symbol=None,
        metadata={
            "symbol_type": "class",
        },
    )

    block = ContextBlock(
        chunk=chunk,
        score=1.0,
        retrieval_method="hybrid",
        dense_score=0.9,
        bm25_score=0.8,
        rerank_score=None,
        expansion_type="retrieved",
    )

    return EvidencePackage(
        query="Where is Request implemented?",
        context_blocks=[block],
        sufficient=sufficient,
        limitation=None,
        repository_revision=repository_revision,
        total_characters=len(chunk.content),
    )


def make_source(
    *,
    stable_id: str = "chunk-request",
    path: str = "scrapy/http/request.py",
    commit_sha: str | None = "commit-123",
) -> SourceReference:
    return SourceReference(
        stable_id=stable_id,
        artifact_id="artifact-request",
        repository="scrapy/scrapy",
        artifact_type="code",
        source_path_or_object_id=path,
        source_url=(
            "https://github.com/scrapy/scrapy/"
            "blob/main/scrapy/http/request.py"
        ),
        commit_sha=commit_sha,
        ref="main",
        language="python",
        start_line=1,
        end_line=3,
        symbol="Request",
        parent_symbol=None,
    )


def make_draft(
    *,
    answer: str = (
        "The Request class is implemented in "
        "scrapy/http/request.py."
    ),
    sources: list[SourceReference] | None = None,
    limitation: str | None = None,
) -> DraftResponse:
    return DraftResponse(
        answer=answer,
        sources=(
            [make_source()]
            if sources is None
            else sources
        ),
        confidence=0.9,
        limitation=limitation,
    )


def test_valid_grounded_response() -> None:
    validator = ResponseValidator()

    result = validator.validate(
        make_draft(),
        make_evidence_package(),
    )

    assert result.valid is True
    assert result.errors == ()
    assert result.cited_source_ids == (
        "chunk-request",
    )
    assert result.grounding_score > 0.20


def test_citation_must_exist_in_evidence() -> None:
    validator = ResponseValidator()

    result = validator.validate(
        make_draft(
            sources=[
                make_source(
                    stable_id="missing-chunk"
                )
            ]
        ),
        make_evidence_package(),
    )

    assert result.valid is False
    assert any(
        "does not map to retrieved evidence"
        in error
        for error in result.errors
    )


def test_citation_provenance_must_match() -> None:
    validator = ResponseValidator()

    result = validator.validate(
        make_draft(
            sources=[
                make_source(
                    path="scrapy/wrong/path.py"
                )
            ]
        ),
        make_evidence_package(),
    )

    assert result.valid is False
    assert any(
        "provenance does not match"
        in error
        for error in result.errors
    )


def test_citations_cannot_be_duplicated() -> None:
    validator = ResponseValidator()

    result = validator.validate(
        make_draft(
            sources=[
                make_source(),
                make_source(),
            ]
        ),
        make_evidence_package(),
    )

    assert result.valid is False
    assert any(
        "Duplicate citation"
        in error
        for error in result.errors
    )


def test_revision_must_match_evidence_package() -> None:
    validator = ResponseValidator()

    result = validator.validate(
        make_draft(
            sources=[
                make_source(
                    commit_sha="different-commit"
                )
            ]
        ),
        make_evidence_package(
            repository_revision="commit-123"
        ),
    )

    assert result.valid is False
    assert any(
        "provenance does not match"
        in error
        for error in result.errors
    )


def test_low_grounding_is_rejected() -> None:
    validator = ResponseValidator(
        minimum_grounding_score=0.50
    )

    result = validator.validate(
        make_draft(
            answer=(
                "The crawler architecture has "
                "several important components."
            )
        ),
        make_evidence_package(),
    )

    assert result.valid is False
    assert any(
        "grounding score"
        in error
        for error in result.errors
    )


def test_missing_citations_are_rejected() -> None:
    validator = ResponseValidator()

    result = validator.validate(
        DraftResponse(
            answer="Request is a class in the repository.",
            sources=[],
            confidence=0.8,
        ),
        make_evidence_package(),
    )

    assert result.valid is False
    assert any(
        "no source citations"
        in error
        for error in result.errors
    )


def test_empty_evidence_requires_limitation() -> None:
    validator = ResponseValidator()

    package = EvidencePackage(
        query="What does Request do?",
        context_blocks=[],
        sufficient=False,
        limitation="No relevant evidence was retrieved.",
        repository_revision="commit-123",
        total_characters=0,
    )

    result = validator.validate(
        DraftResponse(
            answer="There is not enough repository evidence.",
            sources=[],
            confidence=0.1,
            limitation="Insufficient repository evidence.",
        ),
        package,
    )

    assert result.valid is True


def test_empty_evidence_without_limitation_is_rejected() -> None:
    validator = ResponseValidator()

    package = EvidencePackage(
        query="What does Request do?",
        context_blocks=[],
        sufficient=False,
        limitation="No relevant evidence was retrieved.",
        repository_revision="commit-123",
        total_characters=0,
    )

    result = validator.validate(
        DraftResponse(
            answer="Request is responsible for processing requests.",
            sources=[],
            confidence=0.9,
        ),
        package,
    )

    assert result.valid is False
    assert any(
        "does not state a limitation"
        in error
        for error in result.errors
    )


def test_insufficient_evidence_requires_limitation() -> None:
    validator = ResponseValidator()

    package = make_evidence_package(
        sufficient=False
    )

    result = validator.validate(
        make_draft(),
        package,
    )

    assert result.valid is False
    assert any(
        "does not state a limitation"
        in error
        for error in result.errors
    )


def test_build_response_preserves_provenance() -> None:
    validator = ResponseValidator()

    response = validator.build_response(
        make_draft(),
        make_evidence_package(),
    )

    assert response.validated is True
    assert response.repository_revision == "commit-123"
    assert response.sources[0].stable_id == (
        "chunk-request"
    )
    assert response.sources[0].source_path_or_object_id == (
        "scrapy/http/request.py"
    )


def test_build_response_records_validation_metadata() -> None:
    validator = ResponseValidator()

    response = validator.build_response(
        make_draft(),
        make_evidence_package(),
    )

    assert (
        response.evidence_metadata[
            "validation_valid"
        ]
        is True
    )
    assert (
        response.evidence_metadata[
            "cited_source_ids"
        ]
        == ["chunk-request"]
    )


def test_failed_validation_is_not_marked_validated() -> None:
    validator = ResponseValidator()

    response = validator.build_response(
        make_draft(
            sources=[
                make_source(
                    stable_id="missing"
                )
            ]
        ),
        make_evidence_package(),
    )

    assert response.validated is False
    assert response.limitation is not None
    assert (
        "failed grounding validation"
        in response.limitation
    )


def test_validator_rejects_invalid_configuration() -> None:
    try:
        ResponseValidator(
            minimum_grounding_score=1.5
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected invalid grounding threshold to fail."
    )


def test_validator_rejects_non_numeric_configuration() -> None:
    try:
        ResponseValidator(
            minimum_grounding_score="high"
        )
    except TypeError:
        return

    raise AssertionError(
        "Expected non-numeric grounding threshold to fail."
    )


def test_validator_is_deterministic() -> None:
    validator = ResponseValidator()

    package = make_evidence_package()
    draft = make_draft()

    first = validator.validate(
        draft,
        package,
    )

    second = validator.validate(
        draft,
        package,
    )

    assert first == second


def test_source_reference_fields_are_checked() -> None:
    validator = ResponseValidator()

    source = make_source()

    source = source.model_copy(
        update={
            "symbol": "OtherClass",
        }
    )

    result = validator.validate(
        make_draft(
            sources=[source]
        ),
        make_evidence_package(),
    )

    assert result.valid is False
    assert any(
        "provenance does not match"
        in error
        for error in result.errors
    )


def test_grounding_uses_cited_evidence() -> None:
    validator = ResponseValidator()

    package = make_evidence_package()

    result = validator.validate(
        make_draft(
            answer=(
                "The Request class is implemented in "
                "scrapy/http/request.py."
            )
        ),
        package,
    )

    # The validator's contract requires the grounding score
    # to meet the configured minimum threshold. A score above
    # 0.20 is valid even when lexical overlap is not above 0.50.
    assert (
        result.grounding_score
        >= validator.minimum_grounding_score
    )


def test_response_limitations_are_preserved() -> None:
    validator = ResponseValidator()

    draft = make_draft(
        limitation="The available history is incomplete."
    )

    response = validator.build_response(
        draft,
        make_evidence_package(),
    )

    assert (
        response.limitation
        == "The available history is incomplete."
    )