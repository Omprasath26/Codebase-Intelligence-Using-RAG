import numpy as np
import pytest

from src.code_processing import KnowledgeChunk
from src.indexing import Indexing


def make_chunk(stable_id: str,content: str,symbol: str | None = None,metadata: dict | None = None) -> KnowledgeChunk:
    return KnowledgeChunk(
        stable_id=stable_id,
        artifact_id="artifact-123",
        repository="scrapy/scrapy",
        artifact_type="code",
        content=content,
        source_path_or_object_id="scrapy/example.py",
        source_url=(
            "https://github.com/scrapy/scrapy/"
            "blob/master/scrapy/example.py"
        ),
        commit_sha="abc123",
        ref="master",
        language="Python",
        start_line=1,
        end_line=3,
        module="scrapy.example",
        symbol=symbol,
        parent_symbol=None,
        metadata=metadata or {},
    )


def test_builds_faiss_and_bm25_indexes() -> None:
    indexer = Indexing()

    chunks = [
        make_chunk(
            stable_id="chunk-1",
            content="def hello(): return True",
            symbol="hello",
        ),
        make_chunk(
            stable_id="chunk-2",
            content="def crawl(): return response",
            symbol="crawl",
        ),
    ]

    state = indexer.build(chunks)

    assert state["faiss"].ntotal == 2
    assert state["bm25"] is not None
    assert state["embeddings"].shape[0] == 2


def test_embedding_vectors_are_normalized() -> None:
    indexer = Indexing()

    chunks = [
        make_chunk(
            stable_id="chunk-1",
            content="def hello(): return True",
            symbol="hello",
        )
    ]

    state = indexer.build(chunks)

    embeddings = state["embeddings"]

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    assert np.allclose(
        norms,
        1.0,
        atol=1e-5,
    )


def test_preserves_chunk_metadata() -> None:
    indexer = Indexing()

    chunk = make_chunk(
        stable_id="chunk-1",
        content="def hello(): return True",
        symbol="hello",
    )

    state = indexer.build([chunk])

    metadata = state["chunk_metadata"][0]

    assert metadata["stable_id"] == chunk.stable_id
    assert metadata["artifact_id"] == chunk.artifact_id
    assert (
        metadata["source_path_or_object_id"]
        == chunk.source_path_or_object_id
    )
    assert metadata["symbol"] == "hello"
    assert metadata["commit_sha"] == "abc123"
    assert metadata["ref"] == "master"


def test_preserves_enriched_chunk_metadata() -> None:
    indexer = Indexing()

    chunk_metadata = {
        "symbol_type": "function",
        "package": "scrapy.example",
        "imports": [
            "scrapy",
            "typing",
        ],
        "parent_child_relationship": {
            "parent": None,
            "children": [],
        },
    }

    chunk = make_chunk(
        stable_id="chunk-1",
        content="def hello(): return True",
        symbol="hello",
        metadata=chunk_metadata,
    )

    state = indexer.build([chunk])

    metadata = state["chunk_metadata"][0]

    assert metadata["metadata"] == chunk_metadata


def test_index_metadata_contains_version_information() -> None:
    indexer = Indexing(
        index_version="v1"
    )

    chunk = make_chunk(
        stable_id="chunk-1",
        content="def hello(): return True",
    )

    state = indexer.build([chunk])

    metadata = state["metadata"]

    assert metadata.index_version == "v1"
    assert (
        metadata.embedding_model
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert metadata.chunk_count == 1
    assert metadata.embedding_dimension > 0


def test_build_is_deterministic_for_same_chunks() -> None:
    indexer = Indexing()

    chunks = [
        make_chunk(
            stable_id="chunk-2",
            content="def crawl(): return response",
            symbol="crawl",
        ),
        make_chunk(
            stable_id="chunk-1",
            content="def hello(): return True",
            symbol="hello",
        ),
    ]

    first = indexer.build(chunks)
    second = indexer.build(
        list(reversed(chunks))
    )

    assert (
        indexer.build_id(chunks)
        == indexer.build_id(
            list(reversed(chunks))
        )
    )

    assert (
        first["metadata"].chunk_count
        == second["metadata"].chunk_count
    )

    assert np.array_equal(
        first["embeddings"],
        second["embeddings"],
    )

    assert (
        first["chunk_id_by_position"]
        == second["chunk_id_by_position"]
    )


def test_chunk_position_mapping_matches_sorted_chunks() -> None:
    indexer = Indexing()

    chunks = [
        make_chunk(
            stable_id="chunk-2",
            content="def crawl(): return response",
            symbol="crawl",
        ),
        make_chunk(
            stable_id="chunk-1",
            content="def hello(): return True",
            symbol="hello",
        ),
    ]

    state = indexer.build(chunks)

    assert (
        state["chunks"][0].stable_id
        == "chunk-1"
    )

    assert (
        state["chunks"][1].stable_id
        == "chunk-2"
    )

    assert state["chunk_id_by_position"] == {
        0: "chunk-1",
        1: "chunk-2",
    }


def test_bm25_contains_all_chunk_documents() -> None:
    indexer = Indexing()

    chunks = [
        make_chunk(
            stable_id="chunk-1",
            content="hello hello hello function",
            symbol="hello",
        ),
        make_chunk(
            stable_id="chunk-2",
            content="crawl response function",
            symbol="crawl",
        ),
        make_chunk(
            stable_id="chunk-3",
            content="parse request function",
            symbol="parse",
        ),
        make_chunk(
            stable_id="chunk-4",
            content="index document function",
            symbol="index",
        ),
    ]

    state = indexer.build(chunks)

    scores = state["bm25"].get_scores(
        indexer._tokenize("hello")
    )

    assert len(scores) == 4
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    assert scores[0] > scores[3]


def test_bm25_includes_enriched_metadata() -> None:
    indexer = Indexing()

    chunks = [
        make_chunk(
            stable_id="chunk-1",
            content="return response",
            symbol="crawl",
            metadata={
                "api_name": "HtmlResponse",
                "symbol_type": "function",
            },
        ),
        make_chunk(
            stable_id="chunk-2",
            content="return request",
            symbol="parse",
            metadata={
                "api_name": "Request",
                "symbol_type": "function",
            },
        ),
        make_chunk(
            stable_id="chunk-3",
            content="return item",
            symbol="process",
            metadata={
                "api_name": "Item",
                "symbol_type": "class",
            },
        ),
    ]

    state = indexer.build(chunks)

    scores = state["bm25"].get_scores(
        indexer._tokenize("HtmlResponse")
    )

    assert len(scores) == 3
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_build_rejects_empty_chunks() -> None:
    indexer = Indexing()

    try:
        indexer.build([])
    except ValueError as exc:
        assert (
            str(exc)
            == "Cannot build indexes from empty chunks."
        )
    else:
        raise AssertionError(
            "Expected ValueError for empty chunks."
        )


class _FakeEmbeddingModel:
    """Deterministic embedding model used by incremental tests."""

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False):
        vectors = []

        for text in texts:
            value = (
                float(
                    sum(
                        ord(char)
                        for char in text
                    )
                    % 1000
                )
                + 1.0
            )

            vector = np.array(
                [
                    value,
                    value + 1.0,
                    value + 2.0,
                ],
                dtype=np.float32,
            )

            vector /= np.linalg.norm(
                vector
            )

            vectors.append(vector)

        return np.asarray(
            vectors,
            dtype=np.float32,
        )


def test_index_metadata_tracks_chunking_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model",
        index_version="test-v1",
        chunking_version="chunk-v2",
    )

    state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return response",
            )
        ]
    )

    assert (
        state["metadata"].chunking_version
        == "chunk-v2"
    )

    assert state["pipeline_version"] == {
        "index_version": "test-v1",
        "embedding_model": "test-model",
        "chunking_version": "chunk-v2",
    }


def test_incremental_index_reembeds_only_affected_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model",
        index_version="test-v1",
        chunking_version="chunk-v1",
    )

    chunks = [
        make_chunk(
            "chunk-1",
            "return response",
        ),
        make_chunk(
            "chunk-2",
            "return request",
        ),
    ]

    previous_state = indexer.build(
        chunks
    )

    updated_chunks = [
        make_chunk(
            "chunk-1",
            "return updated response",
        ),
        chunks[1],
    ]

    original = indexer._generate_embeddings
    calls = []

    def wrapped(texts):
        calls.append(list(texts))
        return original(texts)

    monkeypatch.setattr(
        indexer,
        "_generate_embeddings",
        wrapped,
    )

    state = indexer.prepare_incremental(
        previous_index_state=previous_state,
        chunks=updated_chunks,
        affected_chunk_ids={"chunk-1"},
    )

    assert len(calls) == 1
    assert len(calls[0]) == 1
    assert (
        "return updated response"
        in calls[0][0]
    )

    assert state["faiss"].ntotal == 2

    assert state[
        "reembedded_chunk_ids"
    ] == ["chunk-1"]

    assert np.array_equal(
        state["embeddings"][1],
        previous_state["embeddings"][1],
    )


def test_incremental_index_removes_deleted_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model"
    )

    chunks = [
        make_chunk(
            "chunk-1",
            "return response",
        ),
        make_chunk(
            "chunk-2",
            "return request",
        ),
    ]

    previous_state = indexer.build(
        chunks
    )

    state = indexer.prepare_incremental(
        previous_index_state=previous_state,
        chunks=[chunks[0]],
        affected_chunk_ids=set(),
        removed_chunk_ids={"chunk-2"},
    )

    assert state["faiss"].ntotal == 1

    assert [
        chunk.stable_id
        for chunk in state["chunks"]
    ] == ["chunk-1"]

    assert state[
        "removed_chunk_ids"
    ] == ["chunk-2"]


def test_incremental_index_rejects_incompatible_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    previous = Indexing(
        embedding_model="model-a",
        index_version="v1",
        chunking_version="chunk-v1",
    )

    state = previous.build(
        [
            make_chunk(
                "chunk-1",
                "return response",
            )
        ]
    )

    current = Indexing(
        embedding_model="model-b",
        index_version="v1",
        chunking_version="chunk-v1",
    )

    with pytest.raises(
        ValueError,
        match="Embedding model"):
        current.prepare_incremental(
            previous_index_state=state,
            chunks=state["chunks"],
            affected_chunk_ids=set(),
        )


def test_build_id_includes_chunking_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    chunks = [
        make_chunk(
            "chunk-1",
            "return response",
        )
    ]

    first = Indexing(
        chunking_version="chunk-v1"
    )

    second = Indexing(
        chunking_version="chunk-v2"
    )

    assert (
        first.build_id(chunks)
        != second.build_id(chunks)
    )



# Task36 — Atomic Index Update & Last-Known-Good Recovery



def test_active_index_state_is_none_before_first_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model"
    )

    assert indexer.active_index_state is None
    assert (
        indexer.last_known_good_index_state
        is None
    )


def test_commit_activates_valid_index_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model"
    )

    state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return response",
            )
        ]
    )

    committed = indexer.commit_index_state(
        state
    )

    assert committed is state
    assert (
        indexer.active_index_state
        is state
    )
    assert (
        indexer.last_known_good_index_state
        is None
    )


def test_commit_preserves_previous_active_as_last_known_good(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model"
    )

    first_state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return response",
            )
        ]
    )

    indexer.commit_index_state(
        first_state
    )

    second_state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return updated response",
            )
        ]
    )

    indexer.commit_index_state(
        second_state
    )

    assert (
        indexer.active_index_state
        is second_state
    )

    assert (
        indexer.last_known_good_index_state
        is first_state
    )


def test_invalid_candidate_does_not_replace_active_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model"
    )

    valid_state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return response",
            )
        ]
    )

    indexer.commit_index_state(
        valid_state
    )

    invalid_state = dict(
        valid_state
    )

    invalid_state["chunks"] = []

    with pytest.raises(
        ValueError,
        match="at least one chunk",
    ):
        indexer.commit_index_state(
            invalid_state
        )

    assert (
        indexer.active_index_state
        is valid_state
    )

    assert (
        indexer.last_known_good_index_state
        is None
    )


def test_validation_rejects_faiss_count_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model"
    )

    state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return response",
            ),
            make_chunk(
                "chunk-2",
                "return request",
            ),
        ]
    )

    invalid_state = dict(
        state
    )

    invalid_state[
        "chunk_id_by_position"
    ] = {
        0: "chunk-1",
    }

    with pytest.raises(
        ValueError,
        match="Chunk-position mapping",
    ):
        indexer.validate_index_state(
            invalid_state
        )


def test_validation_rejects_embedding_count_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model"
    )

    state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return response",
            ),
            make_chunk(
                "chunk-2",
                "return request",
            ),
        ]
    )

    invalid_state = dict(
        state
    )

    invalid_state["embeddings"] = (
        state["embeddings"][:-1]
    )

    with pytest.raises(
        ValueError,
        match="Embedding count",
    ):
        indexer.validate_index_state(
            invalid_state
        )


def test_validation_rejects_bm25_count_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model"
    )

    state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return response",
            ),
            make_chunk(
                "chunk-2",
                "return request",
            ),
        ]
    )

    invalid_state = dict(
        state
    )

    # Build a BM25 index containing only one document
    # while the rest of the candidate still contains two.
    invalid_state["bm25"] = (
        indexer._build_bm25_index(
            ["replacement"]
        )
    )

    with pytest.raises(
        ValueError,
        match="BM25 corpus count"):
        indexer.validate_index_state(
            invalid_state
        )


def test_validation_rejects_pipeline_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model",
        index_version="test-v1",
        chunking_version="chunk-v1",
    )

    state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return response",
            )
        ]
    )

    invalid_state = dict(
        state
    )

    invalid_state[
        "pipeline_version"
    ] = {
        "index_version": "different-v1",
        "embedding_model": "test-model",
        "chunking_version": "chunk-v1",
    }

    with pytest.raises(
        ValueError,
        match="pipeline version"):
        indexer.validate_index_state(
            invalid_state
        )


def test_validation_rejects_duplicate_chunk_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model"
    )

    first = make_chunk(
        "chunk-1",
        "return response",
    )

    second = make_chunk(
        "chunk-2",
        "return request",
    )

    state = indexer.build(
        [first, second]
    )

    invalid_state = dict(
        state
    )

    invalid_state["chunks"] = [
        first,
        first,
    ]

    with pytest.raises(
        ValueError,
        match="duplicate chunk IDs",
    ):
        indexer.validate_index_state(
            invalid_state
        )


def test_recovery_requires_last_known_good_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model"
    )

    with pytest.raises(
        RuntimeError,
        match="No last-known-good",
    ):
        indexer.recover_last_known_good()


def test_recover_last_known_good_restores_previous_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model"
    )

    first_state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return response",
            )
        ]
    )

    indexer.commit_index_state(
        first_state
    )

    second_state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return updated response",
            )
        ]
    )

    indexer.commit_index_state(
        second_state
    )

    recovered = (
        indexer.recover_last_known_good()
    )

    assert recovered is first_state

    assert (
        indexer.active_index_state
        is first_state
    )

    assert (
        indexer.last_known_good_index_state
        is first_state
    )


def test_prepare_and_commit_incremental_activates_candidate(monkeypatch: pytest.MonkeyPatch,) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model",
        index_version="test-v1",
        chunking_version="chunk-v1",
    )

    initial_state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return response",
            ),
            make_chunk(
                "chunk-2",
                "return request",
            ),
        ]
    )

    indexer.commit_index_state(
        initial_state
    )

    updated_chunks = [
        make_chunk(
            "chunk-1",
            "return updated response",
        ),
        make_chunk(
            "chunk-2",
            "return request",
        ),
    ]

    committed = (
        indexer.prepare_and_commit_incremental(
            previous_index_state=initial_state,
            chunks=updated_chunks,
            affected_chunk_ids={
                "chunk-1"
            },
        )
    )

    assert (
        indexer.active_index_state
        is committed
    )

    assert (
        indexer.last_known_good_index_state
        is initial_state
    )

    assert committed[
        "reembedded_chunk_ids"
    ] == ["chunk-1"]


def test_rebuild_and_commit_activates_full_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model"
    )

    rebuilt_state = (
        indexer.rebuild_and_commit(
            [
                make_chunk(
                    "chunk-1",
                    "return response",
                )
            ]
        )
    )

    assert (
        indexer.active_index_state
        is rebuilt_state
    )

    assert (
        indexer.last_known_good_index_state
        is None
    )


def test_failed_incremental_prepare_does_not_change_active_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model"
    )

    initial_state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return response",
            )
        ]
    )

    indexer.commit_index_state(
        initial_state
    )

    with pytest.raises(
        ValueError,
        match="Embedding model"):
        incompatible_indexer = Indexing(
            embedding_model="different-model"
        )

        incompatible_indexer.prepare_and_commit_incremental(
            previous_index_state=initial_state,
            chunks=initial_state["chunks"],
            affected_chunk_ids=set(),
        )

    assert (
        indexer.active_index_state
        is initial_state
    )

    assert (
        indexer.last_known_good_index_state
        is None
    )


def test_failed_candidate_validation_preserves_last_known_good(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.indexing.SentenceTransformer",
        lambda _: _FakeEmbeddingModel(),
    )

    indexer = Indexing(
        embedding_model="test-model"
    )

    first_state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return response",
            )
        ]
    )

    indexer.commit_index_state(
        first_state
    )

    second_state = indexer.build(
        [
            make_chunk(
                "chunk-1",
                "return updated response",
            )
        ]
    )

    indexer.commit_index_state(
        second_state
    )

    invalid_state = dict(
        second_state
    )

    invalid_state[
        "pipeline_version"
    ] = {
        "index_version": "invalid",
        "embedding_model": "test-model",
        "chunking_version": "v1",
    }

    with pytest.raises(
        ValueError,
        match="pipeline version"):
        indexer.commit_index_state(
            invalid_state
        )

    assert (
        indexer.active_index_state
        is second_state
    )

    assert (
        indexer.last_known_good_index_state
        is first_state
    )