"""Tests for PS3 grounding and regression evaluation."""

from __future__ import annotations
from datetime import datetime, timezone
import pytest
from src.code_processing import KnowledgeChunk
from src.evaluation import (EvaluationQuery,RAGEvaluation,RegressionComparison,RetrievalEvaluation)
from src.generation import RAGResponse, SourceReference
from src.retrieval import RetrievedEvidence


def make_chunk(
    *,
    stable_id: str = "chunk-request",
    content: str = (
        "class Request:\n"
        "    def __init__(self, url):\n"
        "        self.url = url\n"
    ),
    symbol: str | None = "Request",
    path: str = "scrapy/http/request.py") -> KnowledgeChunk:
    """Create deterministic repository evidence for evaluation tests."""

    return KnowledgeChunk(
        stable_id=stable_id,
        artifact_id=f"artifact-{stable_id}",
        repository="scrapy/scrapy",
        artifact_type="code",
        content=content,
        source_path_or_object_id=path,
        source_url=(
            "https://github.com/scrapy/scrapy/blob/master/"
            + path
        ),
        commit_sha="evaluation-commit",
        ref="master",
        language="python",
        start_line=1,
        end_line=5,
        module="scrapy.http.request",
        symbol=symbol,
        parent_symbol=None,
        metadata={},
    )


def make_result(*,stable_id: str = "chunk-request",score: float = 1.0,retrieval_method: str = "hybrid") -> RetrievedEvidence:
    """Create deterministic retrieved evidence."""

    return RetrievedEvidence(
        chunk=make_chunk(stable_id=stable_id),
        score=score,
        retrieval_method=retrieval_method,
    )


def make_response(
    *,
    answer: str = (
        "The Request class is implemented in "
        "scrapy/http/request.py."),
    source_ids: tuple[str, ...] = ("chunk-request",)) -> RAGResponse:
    """Create a deterministic grounded response."""

    sources = [
        SourceReference(
            stable_id=source_id,
            artifact_id=f"artifact-{source_id}",
            repository="scrapy/scrapy",
            artifact_type="code",
            source_path_or_object_id=(
                "scrapy/http/request.py"
            ),
            source_url=(
                "https://github.com/scrapy/scrapy/blob/"
                "master/scrapy/http/request.py"
            ),
            commit_sha="evaluation-commit",
            ref="master",
            language="python",
            start_line=1,
            end_line=5,
            symbol="Request",
            parent_symbol=None,
        )
        for source_id in source_ids
    ]

    return RAGResponse(
        answer=answer,
        sources=sources,
        confidence=0.9,
        limitation=None,
        repository_revision="evaluation-commit",
        evidence_metadata={},
        validated=True,
    )



# Retrieval evaluation



def test_source_correctness_returns_full_score_for_expected_sources() -> None:
    """Expected retrieved sources receive full source correctness."""

    evaluator = RetrievalEvaluation()

    score = evaluator.source_correctness(
        [
            "chunk-request",
            "chunk-scheduler",
        ],
        [
            "chunk-request",
            "chunk-scheduler",
        ],
    )

    assert score == 1.0


def test_source_correctness_counts_unexpected_sources() -> None:
    """Unexpected retrieved sources reduce source correctness."""

    evaluator = RetrievalEvaluation()

    score = evaluator.source_correctness(
        [
            "chunk-request",
            "chunk-unrelated",
        ],
        [
            "chunk-request",
        ],
    )

    assert score == 0.5


def test_evaluate_query_includes_source_correctness() -> None:
    """Per-query evaluation reports source correctness."""

    evaluator = RetrievalEvaluation()

    query = EvaluationQuery(
        query_id="q001",
        query="Where is Request implemented?",
        relevant_chunk_ids=frozenset(
            {"chunk-request"}
        ),
        expected_source_ids=frozenset(
            {"chunk-request"}
        ),
    )

    evaluation = evaluator.evaluate_query(
        query,
        [
            make_result(
                stable_id="chunk-request"
            )
        ],
        k_values=(1, 3),
    )

    assert evaluation.recall_at_k[1] == 1.0
    assert evaluation.precision_at_k[1] == 1.0
    assert evaluation.reciprocal_rank == 1.0
    assert evaluation.source_correctness == 1.0


def test_summary_includes_mean_source_correctness() -> None:
    """Aggregate evaluation reports mean source correctness."""

    evaluator = RetrievalEvaluation()

    query = EvaluationQuery(
        query_id="q001",
        query="Where is Request implemented?",
        relevant_chunk_ids=frozenset(
            {"chunk-request"}
        ),
        expected_source_ids=frozenset(
            {"chunk-request"}
        ),
    )

    evaluation = evaluator.evaluate_query(query,[make_result()], k_values=(1,))

    summary = evaluator.summarize( [evaluation],k_values=(1,))

    assert summary.query_count == 1
    assert summary.recall_at_k[1] == 1.0
    assert summary.mean_source_correctness == 1.0



# Hybrid versus vector-only



def test_compare_retrieval_evaluates_hybrid_and_vector_only() -> None:
    """Hybrid and vector-only results use the same ground truth."""

    query = EvaluationQuery(
        query_id="q001",
        query="Where is Request implemented?",
        relevant_chunk_ids=frozenset(
            {"chunk-request"}
        ),
    )

    hybrid_results = [
        make_result(
            stable_id="chunk-request",
            retrieval_method="hybrid",
        ),
        make_result(
            stable_id="chunk-unrelated",
            retrieval_method="hybrid",
        ),
    ]

    vector_results = [
        make_result(
            stable_id="chunk-unrelated",
            retrieval_method="dense",
        ),
        make_result(
            stable_id="chunk-request",
            retrieval_method="dense",
        ),
    ]

    comparison = RetrievalEvaluation.compare_retrieval(
        hybrid_results,
        vector_results,
        query,
        k_values=(1, 2),
    )

    assert comparison["query_id"] == "q001"
    assert (comparison["hybrid"]["recall_at_k"][1]== 1.0)
    assert (comparison["vector_only"]["recall_at_k"][1]== 0.0)
    assert (comparison["hybrid"]["reciprocal_rank"]== 1.0)
    assert (comparison["vector_only"]["reciprocal_rank"]== 0.5)



# Context relevance



def test_context_relevance_detects_query_supported_blocks() -> None:
    """Context relevance recognizes lexical query overlap."""

    score = RetrievalEvaluation.context_relevance(
        "Where is the Request class implemented?",
        [
            (
                "The Request class is implemented in "
                "scrapy/http/request.py."
            ),
            "The Scheduler handles queued requests.",
        ],
    )

    assert score == 1.0


def test_context_relevance_returns_zero_without_context() -> None:
    """No context means no measurable context relevance."""

    score = RetrievalEvaluation.context_relevance(
        "Where is Request implemented?",
        [],
    )

    assert score == 0.0



# Faithfulness / grounding



def test_faithfulness_measures_answer_context_overlap() -> None:
    """Faithfulness measures lexical support in supplied context."""

    score = RetrievalEvaluation.faithfulness(
        (
            "The Request class is implemented in "
            "scrapy/http/request.py."
        ),
        [
            (
                "The Request class is implemented in "
                "scrapy/http/request.py."
            )
        ],
    )

    assert score == 1.0


def test_faithfulness_returns_zero_without_context() -> None:
    """An answer without context receives no grounding support."""

    score = RetrievalEvaluation.faithfulness(
        "The Request class is implemented here.",
        [],
    )

    assert score == 0.0


def test_faithfulness_is_not_claimed_as_semantic_truth() -> None:
    """
    The metric only measures lexical support.

    This test intentionally verifies that unrelated context can still
    produce a non-zero lexical signal when terms overlap.
    """

    score = RetrievalEvaluation.faithfulness(
        "Request is implemented.",
        [
            "Requests are discussed elsewhere."
        ],
    )

    assert score > 0.0



# Answer correctness



def test_answer_correctness_uses_reference_answer() -> None:
    """Reference-answer overlap produces an answer correctness score."""

    score = RetrievalEvaluation.answer_correctness(
        (
            "The Request class is implemented in "
            "scrapy/http/request.py."
        ),
        (
            "Request is implemented in "
            "scrapy/http/request.py."
        ),
    )

    assert score > 0.0
    assert score <= 1.0


def test_answer_correctness_returns_none_without_reference() -> None:
    """No reference answer means correctness is unavailable."""

    score = RetrievalEvaluation.answer_correctness(
        "The Request class is implemented.",
        None,
    )

    assert score is None



# Citation coverage



def test_citation_coverage_is_full_when_expected_source_is_cited() -> None:
    """Expected repository source receives complete citation coverage."""

    response = make_response(
        source_ids=("chunk-request",)
    )

    score = RetrievalEvaluation.citation_coverage(
        response,
        ["chunk-request"],
    )

    assert score == 1.0


def test_citation_coverage_detects_missing_source() -> None:
    """Missing expected citations reduce coverage."""

    response = make_response(
        source_ids=("chunk-request",)
    )

    score = RetrievalEvaluation.citation_coverage(
        response,
        [
            "chunk-request",
            "chunk-scheduler",
        ],
    )

    assert score == 0.5



# Complete RAG evaluation



def test_evaluate_rag_response_returns_grounding_metrics() -> None:
    """A response can be evaluated across the PS3 RAG metrics."""

    evaluator = RetrievalEvaluation()

    query = EvaluationQuery(
        query_id="q001",
        query="Where is Request implemented?",
        relevant_chunk_ids=frozenset(
            {"chunk-request"}
        ),
        reference_answer=(
            "Request is implemented in "
            "scrapy/http/request.py."
        ),
        expected_source_ids=frozenset(
            {"chunk-request"}
        ),
    )

    response = make_response()

    evaluation = evaluator.evaluate_rag_response(
        response=response,
        query=query,
        context_blocks=[
            (
                "The Request class is implemented in "
                "scrapy/http/request.py."
            )
        ],
        latency_seconds=0.25,
        failed=False,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        estimated_cost=0.001,
    )

    assert isinstance(
        evaluation,
        RAGEvaluation,
    )

    assert evaluation.context_relevance > 0.0
    assert evaluation.faithfulness > 0.0
    assert evaluation.answer_correctness is not None
    assert evaluation.answer_correctness > 0.0
    assert evaluation.citation_coverage == 1.0
    assert evaluation.latency_seconds == 0.25
    assert evaluation.failed is False
    assert evaluation.input_tokens == 100
    assert evaluation.output_tokens == 50
    assert evaluation.total_tokens == 150
    assert evaluation.estimated_cost == 0.001


def test_rag_evaluation_supports_failure_metadata() -> None:
    """Operational failure state is preserved in evaluation."""

    evaluator = RetrievalEvaluation()

    query = EvaluationQuery(
        query_id="q002",
        query="Where is Request implemented?",
        relevant_chunk_ids=frozenset(
            {"chunk-request"}
        ),
    )

    response = make_response()

    evaluation = evaluator.evaluate_rag_response(
        response=response,
        query=query,
        context_blocks=[],
        latency_seconds=2.0,
        failed=True,
    )

    assert evaluation.failed is True
    assert evaluation.latency_seconds == 2.0
    assert evaluation.context_relevance == 0.0
    assert evaluation.faithfulness == 0.0



# Regression comparison



def test_regression_comparison_detects_metric_drop() -> None:
    """A meaningful metric drop is reported as a regression."""

    result = RetrievalEvaluation.compare_regression(
        baseline={
            "recall_at_3": 0.90,
            "mrr": 0.80,
        },
        candidate={
            "recall_at_3": 0.75,
            "mrr": 0.85,
        },
        minimum_change=0.05,
    )

    assert isinstance(
        result,
        RegressionComparison)

    assert result.passed is False
    assert "recall_at_3" in result.regressions
    assert "mrr" in result.improved


def test_regression_comparison_passes_when_no_metric_regresses() -> None:
    """Candidate passes when no metric drops beyond tolerance."""

    result = RetrievalEvaluation.compare_regression(
        baseline={
            "recall_at_3": 0.90,
            "mrr": 0.80,
        },
        candidate={
            "recall_at_3": 0.90,
            "mrr": 0.82,
        },
        minimum_change=0.05,
    )

    assert result.passed is True
    assert result.regressions == ()
    assert "mrr" in result.unchanged


def test_regression_comparison_is_deterministic() -> None:
    """Identical inputs produce identical regression results."""

    baseline = {
        "recall_at_3": 0.90,
        "mrr": 0.80,
    }

    candidate = {
        "recall_at_3": 0.85,
        "mrr": 0.82,
    }

    first = RetrievalEvaluation.compare_regression(
        baseline,
        candidate,
        minimum_change=0.01,
    )

    second = RetrievalEvaluation.compare_regression(
        baseline,
        candidate,
        minimum_change=0.01,
    )

    assert first == second



# Dataset integrity



def test_ps3_evaluation_dataset_contains_30_questions() -> None:
    """
    Verify the supplied PS3 question set is preserved.

    The current dataset intentionally has no manually invented
    relevant chunk IDs.
    """

    evaluator = RetrievalEvaluation()

    questions = evaluator.load_ground_truth()

    assert len(questions) == 30
    assert questions[0].query_id == "q001"
    assert questions[0].query == (
        "Where is the Request class implemented?")


def test_ps3_dataset_questions_have_unique_ids() -> None:
    """Evaluation question IDs must remain unique."""

    evaluator = RetrievalEvaluation()

    questions = evaluator.load_ground_truth()

    ids = [
        question.query_id
        for question in questions
    ]

    assert len(ids) == len(set(ids))



# Serialization



def test_retrieval_summary_serialization_includes_source_correctness() -> None:
    """Serialized retrieval summaries contain PS3 source correctness."""

    evaluator = RetrievalEvaluation()

    query = EvaluationQuery(
        query_id="q001",
        query="Where is Request implemented?",
        relevant_chunk_ids=frozenset(
            {"chunk-request"}
        ),
        expected_source_ids=frozenset(
            {"chunk-request"}
        ),
    )

    evaluation = evaluator.evaluate_query(
        query,
        [make_result()],
        k_values=(1,),
    )

    summary = evaluator.summarize(
        [evaluation],
        k_values=(1,),
    )

    serialized = evaluator.to_dict(summary)

    assert serialized["query_count"] == 1
    assert serialized["recall_at_k"]["1"] == 1.0
    assert serialized[
        "mean_source_correctness"
    ] == 1.0


def test_rag_evaluation_serialization_is_json_compatible() -> None:
    """RAG evaluation metrics serialize without custom objects."""

    evaluation = RAGEvaluation(
        context_relevance=1.0,
        faithfulness=0.8,
        answer_correctness=0.9,
        citation_coverage=1.0,
        latency_seconds=0.5,
        failed=False,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        estimated_cost=0.001,
    )

    serialized = RetrievalEvaluation.rag_to_dict(evaluation)

    assert serialized == {
        "context_relevance": 1.0,
        "faithfulness": 0.8,
        "answer_correctness": 0.9,
        "citation_coverage": 1.0,
        "latency_seconds": 0.5,
        "failed": False,
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "estimated_cost": 0.001,
    }