from __future__ import annotations

import json

import pytest

from src.code_processing import KnowledgeChunk
from src.evaluation import (
    EvaluationQuery,
    RetrievalEvaluation,
)
from src.retrieval import RetrievedEvidence


def make_chunk(
    stable_id: str,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        stable_id=stable_id,
        artifact_id=f"artifact-{stable_id}",
        repository="scrapy/scrapy",
        artifact_type="code",
        content=f"content for {stable_id}",
        source_path_or_object_id="scrapy/http/request.py",
        source_url=None,
        commit_sha="abc123",
        ref="main",
        language="python",
        start_line=1,
        end_line=3,
        module="scrapy.http.request",
        symbol="Request",
        parent_symbol=None,
        metadata={},
    )


def make_evidence(
    stable_id: str,
    score: float = 1.0,
    retrieval_method: str = "hybrid",
) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk=make_chunk(stable_id),
        score=score,
        retrieval_method=retrieval_method,
    )


def write_dataset(
    tmp_path,
    questions: list[dict],
):
    dataset_path = (
        tmp_path / "evaluation_dataset.json"
    )

    dataset_path.write_text(
        json.dumps(
            {"questions": questions},
            indent=2,
        ),
        encoding="utf-8",
    )

    return dataset_path


# ----------------------------------------------------------------------
# Ground truth loading
# ----------------------------------------------------------------------


def test_load_ground_truth(tmp_path) -> None:
    dataset_path = write_dataset(
        tmp_path,
        [
            {
                "id": "q1",
                "query": "Where is Request implemented?",
                "relevant_chunk_ids": [
                    "chunk-request",
                    "chunk-module",
                ],
            }
        ],
    )

    evaluator = RetrievalEvaluation(
        dataset_path
    )

    result = evaluator.load_ground_truth()

    assert len(result) == 1
    assert result[0].query_id == "q1"
    assert (
        result[0].query
        == "Where is Request implemented?"
    )
    assert result[0].relevant_chunk_ids == {
        "chunk-request",
        "chunk-module",
    }


def test_missing_dataset_is_rejected(tmp_path) -> None:
    evaluator = RetrievalEvaluation(
        tmp_path / "missing.json"
    )

    with pytest.raises(FileNotFoundError):
        evaluator.load_ground_truth()


def test_invalid_json_is_rejected(tmp_path) -> None:
    dataset_path = (
        tmp_path / "evaluation_dataset.json"
    )

    dataset_path.write_text(
        "{invalid",
        encoding="utf-8",
    )

    evaluator = RetrievalEvaluation(
        dataset_path
    )

    with pytest.raises(ValueError):
        evaluator.load_ground_truth()


def test_invalid_question_schema_is_rejected(
    tmp_path,
) -> None:
    dataset_path = write_dataset(
        tmp_path,
        [
            {
                "id": "q1",
                "query": "Find Request",
                "relevant_chunk_ids": "chunk-request",
            }
        ],
    )

    evaluator = RetrievalEvaluation(
        dataset_path
    )

    with pytest.raises(ValueError):
        evaluator.load_ground_truth()


# ----------------------------------------------------------------------
# Recall
# ----------------------------------------------------------------------


def test_recall_at_k() -> None:
    result = RetrievalEvaluation.recall_at_k(
        retrieved_chunk_ids=[
            "a",
            "b",
            "c",
        ],
        relevant_chunk_ids=[
            "b",
            "c",
        ],
        k=2,
    )

    assert result == 0.5


def test_recall_at_k_full_recall() -> None:
    result = RetrievalEvaluation.recall_at_k(
        retrieved_chunk_ids=[
            "a",
            "b",
            "c",
        ],
        relevant_chunk_ids=[
            "b",
            "c",
        ],
        k=3,
    )

    assert result == 1.0


def test_recall_at_k_invalid_k() -> None:
    with pytest.raises(ValueError):
        RetrievalEvaluation.recall_at_k(
            ["a"],
            ["a"],
            0,
        )


def test_recall_with_no_relevant_items() -> None:
    result = RetrievalEvaluation.recall_at_k(
        ["a", "b"],
        [],
        2,
    )

    assert result == 0.0


# ----------------------------------------------------------------------
# Precision
# ----------------------------------------------------------------------


def test_precision_at_k() -> None:
    result = RetrievalEvaluation.precision_at_k(
        retrieved_chunk_ids=[
            "a",
            "b",
            "c",
        ],
        relevant_chunk_ids=[
            "b",
            "c",
        ],
        k=2,
    )

    assert result == 0.5


def test_precision_at_k_full_precision() -> None:
    result = RetrievalEvaluation.precision_at_k(
        retrieved_chunk_ids=[
            "b",
            "c",
            "a",
        ],
        relevant_chunk_ids=[
            "b",
            "c",
        ],
        k=2,
    )

    assert result == 1.0


def test_precision_with_empty_results() -> None:
    result = RetrievalEvaluation.precision_at_k(
        [],
        ["a"],
        5,
    )

    assert result == 0.0


def test_precision_when_fewer_than_k_results_exist() -> None:
    result = RetrievalEvaluation.precision_at_k(
        retrieved_chunk_ids=[
            "other",
            "b",
        ],
        relevant_chunk_ids=[
            "b",
        ],
        k=3,
    )

    assert result == 0.5


# ----------------------------------------------------------------------
# Reciprocal Rank
# ----------------------------------------------------------------------


def test_reciprocal_rank_first_result() -> None:
    result = RetrievalEvaluation.reciprocal_rank(
        ["a", "b", "c"],
        ["a"],
    )

    assert result == 1.0


def test_reciprocal_rank_second_result() -> None:
    result = RetrievalEvaluation.reciprocal_rank(
        ["a", "b", "c"],
        ["b"],
    )

    assert result == 0.5


def test_reciprocal_rank_missing_relevant_result() -> None:
    result = RetrievalEvaluation.reciprocal_rank(
        ["a", "b", "c"],
        ["x"],
    )

    assert result == 0.0


# ----------------------------------------------------------------------
# Query evaluation
# ----------------------------------------------------------------------


def test_evaluate_query() -> None:
    evaluator = RetrievalEvaluation()

    query = EvaluationQuery(
        query_id="q1",
        query="Where is Request implemented?",
        relevant_chunk_ids=frozenset(
            {
                "chunk-request",
                "chunk-module",
            }
        ),
    )

    results = [
        make_evidence("chunk-other"),
        make_evidence("chunk-request"),
        make_evidence("chunk-module"),
    ]

    evaluation = evaluator.evaluate_query(
        query,
        results,
        k_values=(1, 2, 3),
    )

    assert evaluation.query_id == "q1"

    assert evaluation.retrieved_chunk_ids == (
        "chunk-other",
        "chunk-request",
        "chunk-module",
    )

    assert evaluation.recall_at_k[1] == 0.0
    assert evaluation.recall_at_k[2] == 0.5
    assert evaluation.recall_at_k[3] == 1.0

    assert evaluation.precision_at_k[1] == 0.0
    assert evaluation.precision_at_k[2] == 0.5
    assert evaluation.precision_at_k[3] == pytest.approx(
        2 / 3
    )

    assert evaluation.reciprocal_rank == 0.5


def test_duplicate_retrieved_chunks_are_deduplicated() -> None:
    evaluator = RetrievalEvaluation()

    query = EvaluationQuery(
        query_id="q1",
        query="Find Request",
        relevant_chunk_ids=frozenset(
            {"request"}
        ),
    )

    results = [
        make_evidence("request"),
        make_evidence("request"),
        make_evidence("other"),
    ]

    evaluation = evaluator.evaluate_query(
        query,
        results,
        k_values=(1, 2),
    )

    assert evaluation.retrieved_chunk_ids == (
        "request",
        "other",
    )

    assert evaluation.recall_at_k[1] == 1.0


# ----------------------------------------------------------------------
# Aggregate summary
# ----------------------------------------------------------------------


def test_summarize_evaluations() -> None:
    evaluator = RetrievalEvaluation()

    query_one = EvaluationQuery(
        query_id="q1",
        query="Find A",
        relevant_chunk_ids=frozenset({"a"}),
    )

    query_two = EvaluationQuery(
        query_id="q2",
        query="Find B",
        relevant_chunk_ids=frozenset({"b"}),
    )

    evaluation_one = evaluator.evaluate_query(
        query_one,
        [
            make_evidence("a"),
        ],
        k_values=(1, 3),
    )

    evaluation_two = evaluator.evaluate_query(
        query_two,
        [
            make_evidence("other"),
            make_evidence("b"),
        ],
        k_values=(1, 3),
    )

    summary = evaluator.summarize(
        [
            evaluation_one,
            evaluation_two,
        ],
        k_values=(1, 3),
    )

    assert summary.query_count == 2

    assert summary.recall_at_k[1] == 0.5
    assert summary.recall_at_k[3] == 1.0

    assert summary.precision_at_k[1] == 0.5

    # q1 precision@3 = 1/1 = 1.0
    # q2 precision@3 = 1/2 = 0.5
    # Mean precision@3 = 0.75
    assert summary.precision_at_k[3] == 0.75

    assert summary.mean_reciprocal_rank == 0.75


def test_summary_serialization() -> None:
    evaluator = RetrievalEvaluation()

    query = EvaluationQuery(
        query_id="q1",
        query="Find A",
        relevant_chunk_ids=frozenset({"a"}),
    )

    evaluation = evaluator.evaluate_query(
        query,
        [make_evidence("a")],
        k_values=(1, 3),
    )

    summary = evaluator.summarize(
        [evaluation],
        k_values=(1, 3),
    )

    serialized = evaluator.to_dict(summary)

    assert serialized["query_count"] == 1
    assert serialized["recall_at_k"]["1"] == 1.0
    assert serialized["recall_at_k"]["3"] == 1.0
    assert serialized["precision_at_k"]["1"] == 1.0
    assert serialized["precision_at_k"]["3"] == 1.0
    assert serialized["mean_reciprocal_rank"] == 1.0