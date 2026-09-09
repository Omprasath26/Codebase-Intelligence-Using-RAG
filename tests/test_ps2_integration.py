from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

from src.code_processing import CodeProcessing, KnowledgeChunk
from src.evaluation import EvaluationQuery, RetrievalEvaluation
from src.indexing import Indexing
from src.ingestion_control import RepositoryArtifact
from src.query_analysis import QueryAnalyzer
from src.retrieval import Retrieval


class DeterministicEmbeddingModel:
    """Small deterministic embedding model for integration tests."""

    def encode(self,texts: list[str],convert_to_numpy: bool = True,normalize_embeddings: bool = True,show_progress_bar: bool = False,) -> np.ndarray:
        """Create deterministic vectors from repository concepts."""
        vectors = []

        for text in texts:
            lowered = text.lower()

            vector = np.array(
                [
                    1.0 if "request" in lowered else 0.0,
                    1.0 if "scheduler" in lowered else 0.0,
                    1.0 if "downloader" in lowered else 0.0,
                    1.0 if "spider" in lowered else 0.0,
                ],
                dtype=np.float32,
            )

            norm = np.linalg.norm(vector)

            if normalize_embeddings and norm > 0:
                vector = vector / norm

            vectors.append(vector)

        return np.asarray(vectors, dtype=np.float32)


def make_artifact(*,stable_id: str,path: str,content: str) -> RepositoryArtifact:
    """Create a repository artifact for integration testing."""
    return RepositoryArtifact(
        stable_id=stable_id,
        repository="scrapy/scrapy",
        artifact_type="code",
        content=content,
        source_path_or_object_id=path,
        source_url=(
            "https://github.com/scrapy/scrapy/blob/master/"
            + path
        ),
        language="python",
        commit_sha="integration-commit",
        ref="master",
        ingestion_timestamp=datetime.now(timezone.utc),
        metadata={
            "source": "integration-test",
        },
    )


def build_chunks() -> list[KnowledgeChunk]:
    """Run the real processing stage."""
    processor = CodeProcessing()

    artifacts = [
        make_artifact(
            stable_id="artifact-request",
            path="scrapy/http/request.py",
            content=(
                "class Request:\n"
                "    def __init__(self, url):\n"
                "        self.url = url\n"
                "\n"
                "    def replace(self, **kwargs):\n"
                "        return Request(self.url)\n"
            ),
        ),
        make_artifact(
            stable_id="artifact-scheduler",
            path="scrapy/core/scheduler.py",
            content=(
                "class Scheduler:\n"
                "    def enqueue_request(self, request):\n"
                "        return request\n"
            ),
        ),
        make_artifact(
            stable_id="artifact-downloader",
            path="scrapy/core/downloader/__init__.py",
            content=(
                "class Downloader:\n"
                "    def fetch(self, request):\n"
                "        return request\n"
            ),
        ),
        make_artifact(
            stable_id="artifact-spider",
            path="scrapy/spiders/__init__.py",
            content=(
                "class Spider:\n"
                "    def start_requests(self):\n"
                "        return []\n"
            ),
        ),
    ]

    chunks: list[KnowledgeChunk] = []

    for artifact in artifacts:
        chunks.extend(
            processor.process(artifact)
        )

    return chunks


def build_index(chunks: list[KnowledgeChunk]) -> dict:
    """Build the real indexing stage."""
    with patch(
        "src.indexing.SentenceTransformer",
        return_value=DeterministicEmbeddingModel(),
    ):
        indexer = Indexing(
            embedding_model="integration-test-model",
            index_version="integration-v1",
        )

        return indexer.build(chunks)


def create_retrieval(index_state: dict) -> Retrieval:
    """Create the real hybrid retrieval stage."""
    with patch(
        "src.retrieval.SentenceTransformer",
        return_value=DeterministicEmbeddingModel(),
    ):
        return Retrieval(
            index_state=index_state,
            embedding_model="integration-test-model",
        )


def create_evaluation_query(query_id: str,query: str,relevant_chunk_ids: list[str]) -> EvaluationQuery:
    """Create an evaluation query using the actual evaluation contract."""
    return EvaluationQuery(
        query_id=query_id,
        query=query,
        relevant_chunk_ids=relevant_chunk_ids,
    )


def test_ps2_processing_to_indexing_integration() -> None:
    """Verify processing produces chunks consumed by indexing."""
    chunks = build_chunks()

    assert chunks

    symbols = {
        chunk.symbol
        for chunk in chunks
        if chunk.symbol is not None
    }

    assert "Request" in symbols
    assert "Scheduler" in symbols
    assert "Downloader" in symbols
    assert "Spider" in symbols

    for chunk in chunks:
        assert chunk.stable_id
        assert chunk.artifact_id
        assert chunk.source_path_or_object_id
        assert chunk.commit_sha == "integration-commit"

    index_state = build_index(chunks)

    assert (
        index_state["faiss"].ntotal
        == len(chunks)
    )

    assert (
        len(index_state["chunks"])
        == len(chunks)
    )

    assert (
        len(index_state["chunk_metadata"])
        == len(chunks)
    )

    assert (
        index_state["metadata"].chunk_count
        == len(chunks)
    )


def test_ps2_query_to_hybrid_retrieval_integration() -> None:
    """Verify query analysis feeds hybrid retrieval."""
    chunks = build_chunks()

    index_state = build_index(chunks)

    retrieval = create_retrieval(
        index_state
    )

    analyzer = QueryAnalyzer()

    query_analysis = analyzer.analyze(
        "Where is the Request class implemented?"
    )

    assert "Request" in query_analysis.symbols

    assert query_analysis.intent == "location"

    results = retrieval.retrieve(
        query_analysis=query_analysis,
        top_k=3,
        candidate_k=4,
    )

    assert results

    assert all(
        result.chunk.stable_id
        for result in results
    )

    assert all(
        result.chunk.source_path_or_object_id
        for result in results
    )

    assert all(
        result.chunk.source_url
        for result in results
    )

    retrieved_symbols = [
        result.chunk.symbol
        for result in results
    ]

    assert "Request" in retrieved_symbols


def test_ps2_retrieval_provenance_survives_end_to_end() -> None:
    """Verify retrieval preserves repository provenance."""
    chunks = build_chunks()

    index_state = build_index(chunks)

    retrieval = create_retrieval(
        index_state
    )

    analyzer = QueryAnalyzer()

    query_analysis = analyzer.analyze(
        "Where is the Scheduler class implemented?"
    )

    assert "Scheduler" in query_analysis.symbols

    results = retrieval.retrieve(
        query_analysis=query_analysis,
        top_k=2,
        candidate_k=4,
    )

    scheduler_result = next(
        result
        for result in results
        if result.chunk.symbol == "Scheduler"
    )

    assert (
        scheduler_result.chunk.repository
        == "scrapy/scrapy"
    )

    assert (
        scheduler_result.chunk.source_path_or_object_id
        == "scrapy/core/scheduler.py"
    )

    assert (
        scheduler_result.chunk.commit_sha
        == "integration-commit"
    )

    assert (
        scheduler_result.chunk.ref
        == "master"
    )

    assert scheduler_result.chunk.start_line is not None
    assert scheduler_result.chunk.end_line is not None

    assert scheduler_result.score > 0
    assert scheduler_result.retrieval_method


def test_ps2_retrieval_to_evaluation_integration() -> None:
    """Verify retrieved stable IDs can be evaluated directly."""
    chunks = build_chunks()

    index_state = build_index(chunks)

    retrieval = create_retrieval(
        index_state
    )

    analyzer = QueryAnalyzer()

    query = (
        "Where is the Request class implemented?"
    )

    query_analysis = analyzer.analyze(query)

    results = retrieval.retrieve(
        query_analysis=query_analysis,
        top_k=3,
        candidate_k=4,
    )

    request_chunk = next(
        chunk
        for chunk in chunks
        if chunk.symbol == "Request"
    )

    evaluation_query = create_evaluation_query(
        query_id="q001",
        query=query,
        relevant_chunk_ids=[
            request_chunk.stable_id,
        ],
    )

    evaluator = RetrievalEvaluation()

    evaluation = evaluator.evaluate_query(
        evaluation_query,
        results,
    )

    assert evaluation.query_id == "q001"

    assert (
        evaluation.relevant_chunk_ids
        == [request_chunk.stable_id]
    )

    assert evaluation.retrieved_chunk_ids

    assert evaluation.recall_at_k[1] == 1.0
    assert evaluation.recall_at_k[3] == 1.0

    assert evaluation.precision_at_k[1] == 1.0
    assert evaluation.precision_at_k[3] > 0.0

    assert evaluation.reciprocal_rank == 1.0


def test_ps2_evaluation_summary_is_deterministic() -> None:
    """Verify repeated evaluation produces the same summary."""
    chunks = build_chunks()

    index_state = build_index(chunks)

    retrieval = create_retrieval(
        index_state
    )

    analyzer = QueryAnalyzer()

    query = (
        "Where is the Request class implemented?"
    )

    query_analysis = analyzer.analyze(query)

    results = retrieval.retrieve(
        query_analysis=query_analysis,
        top_k=3,
        candidate_k=4,
    )

    request_chunk = next(
        chunk
        for chunk in chunks
        if chunk.symbol == "Request"
    )

    evaluation_query = create_evaluation_query(
        query_id="q001",
        query=query,
        relevant_chunk_ids=[
            request_chunk.stable_id,
        ],
    )

    evaluator = RetrievalEvaluation()

    evaluation = evaluator.evaluate_query(
        evaluation_query,
        results,
    )

    first = evaluator.summarize(
        [evaluation],
    )

    second = evaluator.summarize(
        [evaluation],
    )

    # EvaluationSummary is deterministic for identical evaluation input.
    # Compare the actual summary object instead of assuming field names
    # that are not part of the implemented contract.
    assert first == second

    assert first.query_count == 1
    assert first.recall_at_k[1] == 1.0
    assert first.recall_at_k[3] == 1.0
