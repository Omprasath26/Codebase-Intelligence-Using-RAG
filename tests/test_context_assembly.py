from src.code_processing import KnowledgeChunk
from src.context_assembly import ContextAssembly
from src.query_analysis import QueryAnalysis
from src.retrieval import RetrievedEvidence


def make_chunk(
    stable_id: str,
    artifact_id: str = "artifact-1",
    artifact_type: str = "code",
    content: str = "content",
    source_path: str = "scrapy/http/request.py",
    commit_sha: str = "abc123",
    symbol: str | None = None,
    parent_symbol: str | None = None,
    start_line: int | None = 1,
    end_line: int | None = 3,
    metadata: dict | None = None
) -> KnowledgeChunk:
    """Create a KnowledgeChunk for context-assembly tests."""

    return KnowledgeChunk(
        stable_id=stable_id,
        artifact_id=artifact_id,
        repository="scrapy/scrapy",
        artifact_type=artifact_type,
        content=content,
        source_path_or_object_id=source_path,
        source_url=None,
        commit_sha=commit_sha,
        ref="master",
        language="python",
        start_line=start_line,
        end_line=end_line,
        module="scrapy.http.request",
        symbol=symbol,
        parent_symbol=parent_symbol,
        metadata=metadata or {},
    )


def make_evidence(chunk: KnowledgeChunk,score: float = 1.0,retrieval_method: str = "hybrid") -> RetrievedEvidence:
    """Create retrieved evidence for testing."""

    return RetrievedEvidence(
        chunk=chunk,
        score=score,
        retrieval_method=retrieval_method,
        dense_score=0.8,
        bm25_score=1.2,
        rerank_score=0.9,
    )


def make_query_analysis(query: str,intent: str = "conceptual",symbols: list[str] | None = None,paths: list[str] | None = None) -> QueryAnalysis:
    """Create query-analysis output for context tests."""

    return QueryAnalysis(
        original_query=query,
        normalized_query=query,
        intent=intent,
        symbols=symbols or [],
        paths=paths or [],
        error_messages=[],
        issue_numbers=[],
        pr_numbers=[],
    )


def test_assembles_retrieved_evidence() -> None:
    """Directly retrieved evidence becomes a context block."""

    chunk = make_chunk(
        stable_id="chunk-1",
        content="class Request: pass",
        symbol="Request",
    )

    package = ContextAssembly(
        chunks=[chunk],
    ).assemble(
        query="Explain Request.",
        evidence=[make_evidence(chunk)],
        query_analysis=make_query_analysis(
            "Explain Request.",
            intent="exact_symbol",
            symbols=["Request"],
        ),
    )

    assert package.sufficient is True
    assert len(package.context_blocks) == 1
    assert package.context_blocks[0].chunk.stable_id == "chunk-1"


def test_expands_parent_chunk() -> None:
    """A method can bring its parent class into the context."""

    parent = make_chunk(
        stable_id="class-request",
        content="class Request:",
        symbol="Request",
        start_line=1,
        end_line=20,
    )

    child = make_chunk(
        stable_id="method-init",
        content="def __init__(self): pass",
        symbol="__init__",
        parent_symbol="Request",
        start_line=5,
        end_line=7,
    )

    package = ContextAssembly(
        chunks=[parent, child],
        neighbor_limit=0,
    ).assemble(
        query="Explain Request initialization.",
        evidence=[make_evidence(child)],
        query_analysis=make_query_analysis(
            "Explain Request initialization.",
            intent="exact_symbol",
            symbols=["__init__"],
        ),
    )

    ids = [
        block.chunk.stable_id
        for block in package.context_blocks
    ]

    assert ids == [
        "method-init",
        "class-request",
    ]


def test_expands_neighbors_from_same_artifact() -> None:
    """Nearby structural chunks from the same artifact are included."""

    first = make_chunk(
        stable_id="chunk-1",
        content="def first(): pass",
        symbol="first",
        start_line=1,
        end_line=3,
    )

    second = make_chunk(
        stable_id="chunk-2",
        content="def second(): pass",
        symbol="second",
        start_line=5,
        end_line=7,
    )

    third = make_chunk(
        stable_id="chunk-3",
        content="def third(): pass",
        symbol="third",
        start_line=9,
        end_line=11,
    )

    package = ContextAssembly(
        chunks=[first, second, third],
        neighbor_limit=1,
    ).assemble(
        query="Explain second.",
        evidence=[make_evidence(second)],
    )

    ids = [
        block.chunk.stable_id
        for block in package.context_blocks
    ]

    assert ids == [
        "chunk-2",
        "chunk-1",
        "chunk-3",
    ]


def test_expands_explicit_related_chunk_ids() -> None:
    """Explicit related chunk IDs can add related context."""

    primary = make_chunk(
        stable_id="chunk-1",
        content="Request implementation",
        metadata={
            "related_chunk_ids": ["chunk-2"],
        },
    )

    related = make_chunk(
        stable_id="chunk-2",
        content="Request test",
        artifact_id="artifact-2",
        source_path="tests/test_request.py",
        symbol="test_request",
    )

    package = ContextAssembly(
        chunks=[primary, related],
        neighbor_limit=0,
    ).assemble(
        query="Explain Request and its test.",
        evidence=[make_evidence(primary)],
    )

    ids = [
        block.chunk.stable_id
        for block in package.context_blocks
    ]

    assert ids == [
        "chunk-1",
        "chunk-2",
    ]

    assert package.context_blocks[1].expansion_type == "related"


def test_deduplicates_parent_and_retrieved_chunk() -> None:
    """The same chunk must not appear more than once."""

    parent = make_chunk(
        stable_id="parent",
        content="class Request: pass",
        symbol="Request",
    )

    child = make_chunk(
        stable_id="child",
        content="def method(self): pass",
        symbol="method",
        parent_symbol="Request",
    )

    package = ContextAssembly(
        chunks=[parent, child],
        neighbor_limit=0,
    ).assemble(
        query="Explain Request method.",
        evidence=[
            make_evidence(child),
            make_evidence(parent),
        ],
    )

    ids = [
        block.chunk.stable_id
        for block in package.context_blocks
    ]

    assert len(ids) == len(set(ids))
    assert ids == [
        "child",
        "parent",
    ]


def test_applies_deterministic_context_budget() -> None:
    """Context selection must stay within the configured budget."""

    chunks = [
        make_chunk(
            stable_id="chunk-1",
            content="a" * 40,
            symbol="one",
        ),
        make_chunk(
            stable_id="chunk-2",
            content="b" * 40,
            symbol="two",
        ),
        make_chunk(
            stable_id="chunk-3",
            content="c" * 40,
            symbol="three",
        ),
    ]

    package = ContextAssembly(
        chunks=chunks,
        max_context_characters=80,
        neighbor_limit=0,
    ).assemble(
        query="Explain the code.",
        evidence=[
            make_evidence(chunks[0], score=1.0),
            make_evidence(chunks[1], score=0.9),
            make_evidence(chunks[2], score=0.8),
        ],
    )

    ids = [
        block.chunk.stable_id
        for block in package.context_blocks
    ]

    assert ids == [
        "chunk-1",
        "chunk-2",
    ]

    assert package.total_characters == 80


def test_truncates_single_oversized_chunk_to_budget() -> None:
    """A single oversized chunk is deterministically truncated."""

    chunk = make_chunk(
        stable_id="chunk-1",
        content="x" * 100,
    )

    package = ContextAssembly(
        chunks=[chunk],
        max_context_characters=30,
        neighbor_limit=0,
    ).assemble(
        query="Explain the code.",
        evidence=[make_evidence(chunk)],
    )

    assert package.total_characters == 30
    assert len(
        package.context_blocks[0].chunk.content
    ) == 30


def test_rejects_mixed_revisions() -> None:
    """Known chunks from incompatible revisions must not be combined."""

    first = make_chunk(
        stable_id="chunk-1",
        commit_sha="commit-a",
    )

    second = make_chunk(
        stable_id="chunk-2",
        commit_sha="commit-b",
    )

    package = ContextAssembly(
        chunks=[first, second],
        neighbor_limit=0,
    ).assemble(
        query="Explain the code.",
        evidence=[
            make_evidence(first),
            make_evidence(second),
        ],
    )

    assert package.sufficient is False
    assert "incompatible repository revisions" in (
        package.limitation or ""
    )


def test_empty_evidence_is_insufficient() -> None:
    """No retrieved evidence must result in insufficiency."""

    package = ContextAssembly(
        chunks=[],
    ).assemble(
        query="Explain the code.",
        evidence=[],
    )

    assert package.sufficient is False
    assert package.context_blocks == []

    assert package.limitation == (
        "No repository evidence was retrieved."
    )


def test_exact_symbol_without_matching_evidence_is_insufficient() -> None:
    """An exact symbol query needs matching symbol evidence."""

    chunk = make_chunk(
        stable_id="chunk-1",
        symbol="Response",
    )

    package = ContextAssembly(
        chunks=[chunk],
        neighbor_limit=0,
    ).assemble(
        query="Explain Request.",
        evidence=[make_evidence(chunk)],
        query_analysis=make_query_analysis(
            "Explain Request.",
            intent="exact_symbol",
            symbols=["Request"],
        ),
    )

    assert package.sufficient is False
    assert "Request" in (
        package.limitation or ""
    )


def test_matching_parent_symbol_satisfies_exact_symbol_query() -> None:
    """Parent-symbol evidence can satisfy an exact-symbol request."""

    parent = make_chunk(
        stable_id="parent",
        symbol="Request",
    )

    child = make_chunk(
        stable_id="child",
        symbol="__init__",
        parent_symbol="Request",
    )

    package = ContextAssembly(
        chunks=[parent, child],
        neighbor_limit=0,
    ).assemble(
        query="Explain Request initialization.",
        evidence=[make_evidence(child)],
        query_analysis=make_query_analysis(
            "Explain Request initialization.",
            intent="exact_symbol",
            symbols=["Request"],
        ),
    )

    assert package.sufficient is True


def test_path_query_requires_matching_path() -> None:
    """A path query needs evidence from the requested path."""

    chunk = make_chunk(
        stable_id="chunk-1",
        source_path="scrapy/http/response.py",
    )

    package = ContextAssembly(
        chunks=[chunk],
        neighbor_limit=0).assemble(
        query="What is implemented in scrapy/http/request.py?",
        evidence=[make_evidence(chunk)],
        query_analysis=make_query_analysis(
            "What is implemented in scrapy/http/request.py?",
            intent="path",
            paths=["scrapy/http/request.py"],
        ),
    )

    assert package.sufficient is False


def test_historical_query_without_history_is_insufficient() -> None:
    """Historical questions need historical repository artifacts."""

    chunk = make_chunk(
        stable_id="chunk-1",
        artifact_type="code",
    )

    package = ContextAssembly(
        chunks=[chunk],
        neighbor_limit=0,
    ).assemble(
        query="When was this behavior introduced?",
        evidence=[make_evidence(chunk)],
        query_analysis=make_query_analysis(
            "When was this behavior introduced?",
            intent="historical",
        ),
    )

    assert package.sufficient is False

    assert "Historical evidence" in (
        package.limitation or ""
    )


def test_historical_query_with_commit_evidence_is_sufficient() -> None:
    """An ingested commit can satisfy a historical query."""

    chunk = make_chunk(
        stable_id="commit-1",
        artifact_type="commit",
        content="Introduced Request behavior.",
    )

    package = ContextAssembly(
        chunks=[chunk],
        neighbor_limit=0,
    ).assemble(
        query="When was this behavior introduced?",
        evidence=[make_evidence(chunk)],
        query_analysis=make_query_analysis(
            "When was this behavior introduced?",
            intent="historical",
        ),
    )

    assert package.sufficient is True


def test_preserves_relevance_metadata_and_provenance() -> None:
    """Relevance and source provenance survive context assembly."""

    chunk = make_chunk(
        stable_id="chunk-1",
        symbol="Request",
        source_path="scrapy/http/request.py",
    )

    evidence = make_evidence(
        chunk,
        score=2.5,
        retrieval_method="hybrid+reranker",
    )

    package = ContextAssembly(
        chunks=[chunk],
        neighbor_limit=0,
    ).assemble(
        query="Explain Request.",
        evidence=[evidence],
    )

    block = package.context_blocks[0]

    assert block.score == 2.5
    assert block.retrieval_method == (
        "hybrid+reranker"
    )

    assert block.dense_score == 0.8
    assert block.bm25_score == 1.2
    assert block.rerank_score == 0.9

    assert (
        block.chunk.source_path_or_object_id
        == "scrapy/http/request.py"
    )

    assert block.chunk.commit_sha == "abc123"


def test_invalid_configuration_is_rejected() -> None:
    """Invalid context-assembly configuration must fail early."""

    try:
        ContextAssembly(
            chunks=[],
            max_context_characters=0,
        )
    except ValueError as exc:
        assert "max_context_characters" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError."
        )

    try:
        ContextAssembly(
            chunks=[],
            neighbor_limit=-1,
        )
    except ValueError as exc:
        assert "neighbor_limit" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError."
        )



# Task 27 — Evidence Sufficiency, Confidence & Limitations


def test_confidence_is_zero_when_evidence_is_missing() -> None:
    """Missing evidence must have zero confidence."""

    package = ContextAssembly(
        chunks=[],
    ).assemble(
        query="Explain the code.",
        evidence=[],
    )

    assert package.sufficient is False
    assert package.confidence == 0.0
    assert package.refinement_required is False


def test_sufficient_exact_symbol_has_high_confidence() -> None:
    """Direct symbol evidence should produce strong confidence."""

    chunk = make_chunk(
        stable_id="chunk-1",
        content="class Request: pass",
        symbol="Request",
    )

    package = ContextAssembly(
        chunks=[chunk],
        neighbor_limit=0,
    ).assemble(
        query="Explain Request.",
        evidence=[make_evidence(chunk)],
        query_analysis=make_query_analysis(
            "Explain Request.",
            intent="exact_symbol",
            symbols=["Request"],
        ),
    )

    assert package.sufficient is True
    assert package.confidence == 0.75
    assert package.refinement_required is False


def test_sufficient_path_evidence_increases_confidence() -> None:
    """Evidence from the requested path should strengthen confidence."""

    chunk = make_chunk(
        stable_id="chunk-1",
        content="class Request: pass",
        source_path="scrapy/http/request.py",
    )

    package = ContextAssembly(
        chunks=[chunk],
        neighbor_limit=0,
    ).assemble(
        query="What is implemented in scrapy/http/request.py?",
        evidence=[make_evidence(chunk)],
        query_analysis=make_query_analysis(
            "What is implemented in scrapy/http/request.py?",
            intent="path",
            paths=["scrapy/http/request.py"],
        ),
    )

    assert package.sufficient is True
    assert package.confidence == 0.65
    assert package.refinement_required is False


def test_insufficient_symbol_evidence_requires_refinement() -> None:
    """Missing requested symbol should trigger a refinement signal."""

    chunk = make_chunk(
        stable_id="chunk-1",
        content="class Response: pass",
        symbol="Response",
    )

    package = ContextAssembly(
        chunks=[chunk],
        neighbor_limit=0,
    ).assemble(
        query="Explain Request.",
        evidence=[make_evidence(chunk)],
        query_analysis=make_query_analysis(
            "Explain Request.",
            intent="exact_symbol",
            symbols=["Request"],
        ),
    )

    assert package.sufficient is False
    assert package.confidence == 0.0
    assert package.refinement_required is True
    assert package.limitation is not None


def test_mixed_revisions_require_refinement() -> None:
    """Conflicting revisions must block generation and request refinement."""

    first = make_chunk(
        stable_id="chunk-1",
        commit_sha="commit-a",
    )

    second = make_chunk(
        stable_id="chunk-2",
        commit_sha="commit-b",
    )

    package = ContextAssembly(
        chunks=[first, second],
        neighbor_limit=0,
    ).assemble(
        query="Explain the code.",
        evidence=[
            make_evidence(first),
            make_evidence(second),
        ],
    )

    assert package.sufficient is False
    assert package.confidence == 0.0
    assert package.refinement_required is True
    assert package.repository_revision is None


def test_historical_evidence_has_strong_confidence() -> None:
    """Historical artifact evidence should satisfy historical questions."""

    chunk = make_chunk(
        stable_id="commit-1",
        artifact_type="commit",
        content="Introduced Request behavior.",
    )

    package = ContextAssembly(
        chunks=[chunk],
        neighbor_limit=0,
    ).assemble(
        query="When was this behavior introduced?",
        evidence=[make_evidence(chunk)],
        query_analysis=make_query_analysis(
            "When was this behavior introduced?",
            intent="historical",
        ),
    )

    assert package.sufficient is True
    assert package.confidence == 0.60
    assert package.refinement_required is False