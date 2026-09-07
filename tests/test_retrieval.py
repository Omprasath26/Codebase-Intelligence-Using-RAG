from unittest.mock import MagicMock, patch

import pytest

from src.code_processing import KnowledgeChunk
from src.query_analysis import QueryAnalysis
from src.retrieval import Retrieval


def create_chunk(
    stable_id: str,
    symbol: str,
    path: str,
    content: str,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        stable_id=stable_id,
        artifact_id=f"artifact-{stable_id}",
        repository="scrapy/scrapy",
        artifact_type="code",
        content=content,
        source_path_or_object_id=path,
        source_url=f"https://github.com/scrapy/scrapy/blob/master/{path}",
        commit_sha="abc123",
        ref="master",
        language="python",
        start_line=1,
        end_line=5,
        module=path.replace("/", ".").removesuffix(".py"),
        symbol=symbol,
        parent_symbol=None,
        metadata={},
    )


def create_retrieval(
    chunks: list[KnowledgeChunk],
) -> Retrieval:
    faiss_index = MagicMock()
    faiss_index.ntotal = len(chunks)

    bm25_index = MagicMock()

    index_state = {
        "faiss": faiss_index,
        "bm25": bm25_index,
        "chunks": chunks,
        "chunk_metadata": {
            index: {
                "stable_id": chunk.stable_id,
                "artifact_id": chunk.artifact_id,
                "repository": chunk.repository,
                "artifact_type": chunk.artifact_type,
                "source_path_or_object_id": (
                    chunk.source_path_or_object_id
                ),
                "source_url": chunk.source_url,
                "commit_sha": chunk.commit_sha,
                "ref": chunk.ref,
                "language": chunk.language,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "module": chunk.module,
                "symbol": chunk.symbol,
            }
            for index, chunk in enumerate(chunks)
        },
    }

    with patch("src.retrieval.SentenceTransformer") as model_class:
        model_class.return_value = MagicMock()

        return Retrieval(
            index_state=index_state,
            embedding_model="test-model",
        )


def create_query(
    query: str = "parse response",
) -> QueryAnalysis:
    return QueryAnalysis(
        original_query=query,
        normalized_query=query,
        intent="general",
        symbols=[],
        paths=[],
    )


def test_rejects_empty_index_state() -> None:
    with pytest.raises(
        ValueError,
        match="Index state cannot be empty",
    ):
        Retrieval(index_state={})


def test_rejects_inconsistent_index_size() -> None:
    chunks = [
        create_chunk(
            "chunk-1",
            "Request",
            "scrapy/http/request.py",
            "class Request:",
        )
    ]

    faiss_index = MagicMock()
    faiss_index.ntotal = 0

    bm25_index = MagicMock()

    with patch("src.retrieval.SentenceTransformer") as model_class:
        model_class.return_value = MagicMock()

        with pytest.raises(
            ValueError,
            match="FAISS index size does not match chunk count",
        ):
            Retrieval(
                index_state={
                    "faiss": faiss_index,
                    "bm25": bm25_index,
                    "chunks": chunks,
                    "chunk_metadata": {},
                },
                embedding_model="test-model",
            )


def test_fuses_dense_and_bm25_results() -> None:
    chunks = [
        create_chunk(
            "chunk-1",
            "Request",
            "scrapy/http/request.py",
            "class Request:",
        ),
        create_chunk(
            "chunk-2",
            "Response",
            "scrapy/http/response.py",
            "class Response:",
        ),
        create_chunk(
            "chunk-3",
            "Crawler",
            "scrapy/core/crawler.py",
            "class Crawler:",
        ),
    ]

    retrieval = create_retrieval(chunks)

    dense_results = [
        (0, 0.90),
        (1, 0.80),
    ]

    lexical_results = [
        (1, 4.0),
        (0, 3.0),
    ]

    results = retrieval._fuse_results(
        dense_results,
        lexical_results,
    )

    assert len(results) == 2
    assert results[0].retrieval_method == "bm25+dense"


def test_returns_top_k_results() -> None:
    chunks = [
        create_chunk(
            f"chunk-{index}",
            f"Symbol{index}",
            f"module{index}.py",
            f"content {index}",
        )
        for index in range(4)
    ]

    retrieval = create_retrieval(chunks)

    retrieval._dense_search = MagicMock(
        return_value=[
            (0, 0.9),
            (1, 0.8),
            (2, 0.7),
            (3, 0.6),
        ]
    )

    retrieval._bm25_search = MagicMock(
        return_value=[
            (0, 4.0),
            (1, 3.0),
            (2, 2.0),
            (3, 1.0),
        ]
    )

    results = retrieval.retrieve(
        create_query(),
        top_k=2,
    )

    assert len(results) == 2


def test_applies_metadata_filter() -> None:
    chunks = [
        create_chunk(
            "chunk-1",
            "Request",
            "scrapy/http/request.py",
            "class Request:",
        ),
        create_chunk(
            "chunk-2",
            "Response",
            "scrapy/http/response.py",
            "class Response:",
        ),
    ]

    retrieval = create_retrieval(chunks)

    retrieval._dense_search = MagicMock(
        return_value=[
            (0, 0.9),
            (1, 0.8),
        ]
    )

    retrieval._bm25_search = MagicMock(
        return_value=[
            (0, 4.0),
            (1, 3.0),
        ]
    )

    results = retrieval.retrieve(
        create_query(),
        top_k=5,
        metadata_filters={
            "symbol": "Request",
        },
    )

    assert len(results) == 1
    assert results[0].chunk.symbol == "Request"


def test_preserves_provenance() -> None:
    chunks = [
        create_chunk(
            "chunk-1",
            "Request",
            "scrapy/http/request.py",
            "class Request:",
        )
    ]

    retrieval = create_retrieval(chunks)

    retrieval._dense_search = MagicMock(
        return_value=[(0, 0.9)]
    )

    retrieval._bm25_search = MagicMock(
        return_value=[(0, 4.0)]
    )

    results = retrieval.retrieve(
        create_query(),
        top_k=1,
    )

    evidence = results[0]

    assert evidence.chunk.stable_id == "chunk-1"
    assert evidence.chunk.artifact_id == "artifact-chunk-1"
    assert evidence.chunk.source_path_or_object_id == (
        "scrapy/http/request.py"
    )
    assert evidence.chunk.commit_sha == "abc123"
    assert evidence.chunk.ref == "master"
    assert evidence.chunk.start_line == 1
    assert evidence.chunk.end_line == 5


def test_returns_empty_for_empty_chunks() -> None:
    retrieval = create_retrieval([])

    results = retrieval.retrieve(
        create_query(),
        top_k=5,
    )

    assert results == []


def test_rejects_invalid_top_k() -> None:
    chunks = [
        create_chunk(
            "chunk-1",
            "Request",
            "scrapy/http/request.py",
            "class Request:",
        )
    ]

    retrieval = create_retrieval(chunks)

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        retrieval.retrieve(
            create_query(),
            top_k=0,
        )