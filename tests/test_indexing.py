import numpy as np

from src.code_processing import KnowledgeChunk
from src.indexing import Indexing


def make_chunk(
    stable_id: str,
    content: str,
    symbol: str | None = None,
) -> KnowledgeChunk:
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
        metadata={},
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
    second = indexer.build(list(reversed(chunks)))

    assert indexer.build_id(chunks) == indexer.build_id(
        list(reversed(chunks))
    )

    assert first["metadata"].chunk_count == (
        second["metadata"].chunk_count
    )

    assert np.array_equal(
        first["embeddings"],
        second["embeddings"],
    )

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