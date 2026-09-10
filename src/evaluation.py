"""Evaluation utilities for retrieval quality and grounded RAG quality."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.generation import RAGResponse
from src.retrieval import RetrievedEvidence


@dataclass(frozen=True)
class EvaluationQuery:
    """One evaluation query and its expected evidence."""

    query_id: str
    query: str
    relevant_chunk_ids: frozenset[str]
    reference_answer: str | None = None
    expected_source_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class QueryEvaluation:
    """Retrieval metrics for one evaluation query."""

    query_id: str
    query: str
    retrieved_chunk_ids: tuple[str, ...]
    relevant_chunk_ids: frozenset[str]
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    reciprocal_rank: float
    source_correctness: float = 0.0


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate retrieval evaluation metrics."""

    query_count: int
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mean_reciprocal_rank: float
    mean_source_correctness: float = 0.0


@dataclass(frozen=True)
class RAGEvaluation:
    """Grounded RAG and operational evaluation metrics."""

    context_relevance: float
    faithfulness: float
    answer_correctness: float | None
    citation_coverage: float

    latency_seconds: float | None = None
    failed: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost: float | None = None

    query_id: str | None = None
    query: str | None = None


@dataclass(frozen=True)
class RegressionComparison:
    """Comparison between baseline and candidate metrics."""

    metric_deltas: dict[str, float]
    regressions: tuple[str, ...]
    improved: tuple[str, ...]
    unchanged: tuple[str, ...]
    passed: bool


class RetrievalEvaluation:
    """Evaluate retrieval and grounded RAG results deterministically."""

    def __init__(self,dataset_path: str | Path | None = None) -> None:
        self.dataset_path = (
            Path(dataset_path)
            if dataset_path is not None
            else Path(
                "data/evaluation/evaluation_dataset.json"
            )
        )

    
    # Ground-truth loading
    

    def load_ground_truth(self) -> list[EvaluationQuery]:
        """Load and validate the evaluation dataset."""

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: {self.dataset_path}"
            )

        try:
            with self.dataset_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                dataset = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid evaluation dataset JSON: {exc}"
            ) from exc

        if not isinstance(dataset, dict):
            raise ValueError(
                "Evaluation dataset must be a JSON object."
            )

        questions = dataset.get("questions")

        if not isinstance(questions, list):
            raise ValueError(
                "Evaluation dataset must contain a "
                "'questions' list."
            )

        if not questions:
            raise ValueError(
                "Evaluation dataset must contain at least "
                "one question."
            )

        evaluation_queries: list[EvaluationQuery] = []
        seen_query_ids: set[str] = set()

        for index, record in enumerate(questions):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Question at index {index} must be an object."
                )

            query_id = record.get("id")
            query = record.get("query")
            relevant_chunk_ids = record.get(
                "relevant_chunk_ids"
            )

            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(
                    f"Question at index {index} has an invalid 'id'."
                )

            query_id = query_id.strip()

            if query_id in seen_query_ids:
                raise ValueError(
                    f"Duplicate evaluation query ID: {query_id}"
                )

            seen_query_ids.add(query_id)

            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"{query_id}: 'query' must be a "
                    "non-empty string."
                )

            if not isinstance(relevant_chunk_ids, list):
                raise ValueError(
                    f"{query_id}: 'relevant_chunk_ids' "
                    "must be a list."
                )

            normalized_ids: list[str] = []

            for chunk_id in relevant_chunk_ids:
                if not isinstance(chunk_id, str):
                    raise ValueError(
                        f"{query_id}: every relevant chunk ID "
                        "must be a string."
                    )

                chunk_id = chunk_id.strip()

                if not chunk_id:
                    raise ValueError(
                        f"{query_id}: relevant chunk IDs "
                        "cannot be empty."
                    )

                normalized_ids.append(chunk_id)

            if len(set(normalized_ids)) != len(normalized_ids):
                raise ValueError(
                    f"{query_id}: duplicate relevant chunk "
                    "IDs are not allowed."
                )

            reference_answer = record.get(
                "reference_answer"
            )

            if reference_answer is not None:
                if (
                    not isinstance(reference_answer, str)
                    or not reference_answer.strip()):
                    raise ValueError(
                        f"{query_id}: 'reference_answer' must "
                        "be a non-empty string when provided."
                    )

                reference_answer = reference_answer.strip()

            expected_source_ids = record.get(
                "expected_source_ids",
                [],
            )

            if not isinstance(expected_source_ids,list):
                raise ValueError(
                    f"{query_id}: 'expected_source_ids' "
                    "must be a list."
                )

            normalized_source_ids: list[str] = []

            for source_id in expected_source_ids:
                if not isinstance(source_id, str):
                    raise ValueError(
                        f"{query_id}: every expected source ID "
                        "must be a string."
                    )

                source_id = source_id.strip()

                if not source_id:
                    raise ValueError(
                        f"{query_id}: expected source IDs "
                        "cannot be empty."
                    )

                normalized_source_ids.append(
                    source_id
                )

            if len(set(normalized_source_ids)) != len(normalized_source_ids):
                raise ValueError(
                    f"{query_id}: duplicate expected source "
                    "IDs are not allowed."
                )

            evaluation_queries.append(
                EvaluationQuery(
                    query_id=query_id,
                    query=query.strip(),
                    relevant_chunk_ids=frozenset(
                        normalized_ids
                    ),
                    reference_answer=reference_answer,
                    expected_source_ids=frozenset(
                        normalized_source_ids
                    ),
                )
            )

        return evaluation_queries

    
    # Retrieval result normalization
    

    @staticmethod
    def _extract_chunk_ids(results: Iterable[RetrievedEvidence]) -> tuple[str, ...]:
        """Extract stable chunk IDs preserving ranking order."""

        chunk_ids: list[str] = []
        seen: set[str] = set()

        for result in results:
            if not isinstance(result,RetrievedEvidence):
                raise TypeError(
                    "Retrieval results must contain "
                    "RetrievedEvidence."
                )

            chunk_id = result.chunk.stable_id

            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            chunk_ids.append(chunk_id)

        return tuple(chunk_ids)

    
    # Retrieval metrics
    
    @staticmethod
    def recall_at_k(retrieved_chunk_ids: Iterable[str],relevant_chunk_ids: Iterable[str],k: int,) -> float:
        """Calculate Recall@K."""

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        relevant = set(relevant_chunk_ids)

        if not relevant:
            return 0.0

        retrieved = set(list(retrieved_chunk_ids)[:k])

        return len(
            retrieved.intersection(relevant)
        ) / len(relevant)

    @staticmethod
    def precision_at_k(retrieved_chunk_ids: Iterable[str],relevant_chunk_ids: Iterable[str],k: int) -> float:
        """Calculate Precision@K."""

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        relevant = set(
            relevant_chunk_ids
        )

        retrieved = list(
            retrieved_chunk_ids
        )[:k]

        if not retrieved:
            return 0.0

        relevant_count = sum(
            chunk_id in relevant
            for chunk_id in retrieved
        )

        return relevant_count / len(
            retrieved
        )

    @staticmethod
    def reciprocal_rank(retrieved_chunk_ids: Iterable[str],relevant_chunk_ids: Iterable[str]) -> float:
        """Calculate reciprocal rank."""

        relevant = set(
            relevant_chunk_ids
        )

        if not relevant:
            return 0.0

        for rank, chunk_id in enumerate(retrieved_chunk_ids,start=1):
            if chunk_id in relevant:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def source_correctness(retrieved_source_ids: Iterable[str],expected_source_ids: Iterable[str],relevant_chunk_ids: Iterable[str] = ()) -> float:
        """
        Measure source correctness.

        Unexpected retrieved sources reduce the score.
        """

        expected = set(
            expected_source_ids
        )

        if not expected:
            expected = set(
                relevant_chunk_ids
            )

        retrieved = list(
            dict.fromkeys(
                retrieved_source_ids
            )
        )

        if not retrieved:
            return 0.0

        if not expected:
            return 0.0

        correct = sum(
            source_id in expected
            for source_id in retrieved
        )

        return correct / len(
            retrieved
        )

    
    # One-query retrieval evaluation
    

    def evaluate_query(self,
        evaluation_query: EvaluationQuery,
        results: Iterable[RetrievedEvidence],
        k_values: Iterable[int] = (1,3,5,10),) -> QueryEvaluation:
        """Evaluate retrieval results for one query."""

        if not isinstance(evaluation_query,EvaluationQuery):
            raise TypeError(
                "evaluation_query must be "
                "an EvaluationQuery."
            )

        k_values = tuple(k_values)

        if not k_values:
            raise ValueError(
                "At least one k value is required."
            )

        if any(k <= 0 for k in k_values):
            raise ValueError(
                "All k values must be greater than zero."
            )

        retrieved_chunk_ids = (
            self._extract_chunk_ids(results)
        )

        recall = {
            k: self.recall_at_k(
                retrieved_chunk_ids,
                evaluation_query.relevant_chunk_ids,
                k,
            )
            for k in k_values
        }

        precision = {
            k: self.precision_at_k(
                retrieved_chunk_ids,
                evaluation_query.relevant_chunk_ids,
                k,
            )
            for k in k_values
        }

        reciprocal_rank = (
            self.reciprocal_rank(
                retrieved_chunk_ids,
                evaluation_query.relevant_chunk_ids,
            )
        )

        source_correctness = (
            self.source_correctness(
                retrieved_chunk_ids,
                evaluation_query.expected_source_ids,
                evaluation_query.relevant_chunk_ids,
            )
        )

        return QueryEvaluation(
            query_id=evaluation_query.query_id,
            query=evaluation_query.query,
            retrieved_chunk_ids=(
                retrieved_chunk_ids
            ),
            relevant_chunk_ids=(
                evaluation_query.relevant_chunk_ids
            ),
            recall_at_k=recall,
            precision_at_k=precision,
            reciprocal_rank=(
                reciprocal_rank
            ),
            source_correctness=(
                source_correctness
            ),
        )

    
    # Aggregate retrieval evaluation
    

    @staticmethod
    def summarize(
        evaluations: Iterable[QueryEvaluation],
        k_values: Iterable[int] = (1,3,5,10),) -> EvaluationSummary:
        """Aggregate per-query retrieval metrics."""

        evaluations = list(
            evaluations
        )

        k_values = tuple(
            k_values
        )

        if not k_values:
            raise ValueError(
                "At least one k value is required."
            )

        if not evaluations:
            raise ValueError(
                "At least one query evaluation "
                "is required."
            )

        if any(k <= 0 for k in k_values):
            raise ValueError(
                "All k values must be greater than zero."
            )

        recall_at_k = {
            k: sum(
                evaluation.recall_at_k[k]
                for evaluation in evaluations
            ) / len(evaluations)
            for k in k_values
        }

        precision_at_k = {
            k: sum(
                evaluation.precision_at_k[k]
                for evaluation in evaluations
            ) / len(evaluations)
            for k in k_values
        }

        mean_reciprocal_rank = (
            sum(
                evaluation.reciprocal_rank
                for evaluation in evaluations
            ) / len(evaluations)
        )

        mean_source_correctness = (
            sum(
                evaluation.source_correctness
                for evaluation in evaluations
            ) / len(evaluations)
        )

        return EvaluationSummary(
            query_count=len(evaluations),
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mean_reciprocal_rank=(
                mean_reciprocal_rank
            ),
            mean_source_correctness=(
                mean_source_correctness
            ),
        )

    
    # Hybrid versus vector-only
    

    @staticmethod
    def _query_evaluation_to_dict(evaluation: QueryEvaluation) -> dict[str, Any]:
        """Serialize a query evaluation for comparison output."""

        return {
            "query_id": evaluation.query_id,
            "query": evaluation.query,
            "retrieved_chunk_ids": list(
                evaluation.retrieved_chunk_ids
            ),
            "relevant_chunk_ids": list(
                evaluation.relevant_chunk_ids
            ),
            "recall_at_k": dict(
                evaluation.recall_at_k
            ),
            "precision_at_k": dict(
                evaluation.precision_at_k
            ),
            "reciprocal_rank": (
                evaluation.reciprocal_rank
            ),
            "source_correctness": (
                evaluation.source_correctness
            ),
        }

    @staticmethod
    def compare_retrieval(
        hybrid_results: Iterable[RetrievedEvidence],
        vector_results: Iterable[RetrievedEvidence],
        evaluation_query: EvaluationQuery,
        k_values: Iterable[int] = (1,3,5,10,),) -> dict[str, Any]:
        """
        Compare hybrid and vector-only retrieval.

        The returned structure is intentionally JSON-friendly and exposes
        each evaluation as a dictionary so callers can directly access
        metrics such as ``comparison["hybrid"]["recall_at_k"][1]``.
        """

        evaluator = RetrievalEvaluation()

        hybrid_evaluation = evaluator.evaluate_query(
            evaluation_query,
            hybrid_results,
            k_values,
        )

        vector_evaluation = evaluator.evaluate_query(
            evaluation_query,
            vector_results,
            k_values,
        )

        return {
            "query_id": evaluation_query.query_id,
            "query": evaluation_query.query,
            "hybrid": (
                RetrievalEvaluation._query_evaluation_to_dict(
                    hybrid_evaluation
                )
            ),
            "vector_only": (
                RetrievalEvaluation._query_evaluation_to_dict(
                    vector_evaluation
                )
            ),
        }

    
    # Lexical helpers
    

    _LEXICAL_STOPWORDS = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "be",
            "by",
            "does",
            "for",
            "from",
            "how",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "the",
            "this",
            "that",
            "these",
            "those",
            "to",
            "was",
            "what",
            "where",
            "which",
            "who",
            "why",
            "with",
        }
    )

    @staticmethod
    def _normalize_token(token: str) -> str:
        """Apply lightweight deterministic lexical normalization."""

        token = token.lower()

        if "_" in token or any(
            character.isdigit()
            for character in token):
            return token

        if (len(token) > 4 and token.endswith("ies")):
            return token[:-3] + "y"

        if (len(token) > 4 and token.endswith("ses")):
            return token[:-2]

        if (len(token) > 3 and token.endswith("s")):
            return token[:-1]

        return token

    @classmethod
    def _normalize_text(cls,text: str,) -> tuple[str, ...]:
        """
        Normalize text into deterministic lexical tokens.

        Common natural-language stopwords are excluded while programming
        identifiers remain preserved.
        """

        if not isinstance(text, str):
            return ()

        raw_tokens = re.findall(
            r"[A-Za-z0-9_]+",
            text.lower(),
        )

        normalized: list[str] = []

        for token in raw_tokens:
            token = cls._normalize_token(
                token
            )

            if (
                token in cls._LEXICAL_STOPWORDS
                and "_" not in token
                and not any(
                    character.isdigit()
                    for character in token
                )
            ):
                continue

            normalized.append(token)

        return tuple(normalized)

    @classmethod
    def _token_set(cls,text: str) -> set[str]:
        """Return normalized lexical tokens."""

        return set(cls._normalize_text(text))

    @classmethod
    def _lexical_overlap(cls,left: str,right: str) -> float:
        """Calculate directional lexical overlap."""

        left_tokens = cls._token_set(left)
        right_tokens = cls._token_set(right)

        if not left_tokens or not right_tokens:
            return 0.0

        return len(
            left_tokens.intersection(
                right_tokens
            )
        ) / len(left_tokens)

    
    # RAG metrics
    

    @classmethod
    def context_relevance( cls,query: str,contexts: Iterable[str]) -> float:
        """
        Estimate whether at least one context supports the query.

        The strongest context block is used.

        This is a lexical heuristic and not semantic evaluation.
        """

        contexts = list(
            contexts
        )

        if not contexts:
            return 0.0

        scores = [
            cls._lexical_overlap(
                query,
                context,
            )
            for context in contexts
        ]

        return max(
            scores,
            default=0.0,
        )

    @classmethod
    def faithfulness(cls,answer: str,contexts: Iterable[str]) -> float:
        """
        Estimate lexical support for an answer.

        This does NOT establish semantic truth.
        """

        contexts = list(
            contexts
        )

        if not contexts:
            return 0.0

        answer_tokens = cls._token_set(answer)

        if not answer_tokens:
            return 0.0

        context_tokens: set[str] = set()

        for context in contexts:
            context_tokens.update(
                cls._token_set(context)
            )

        if not context_tokens:
            return 0.0

        return len(
            answer_tokens.intersection(
                context_tokens
            )
        ) / len(answer_tokens)

    @classmethod
    def answer_correctness(cls,answer: str, reference_answer: str | None) -> float | None:
        """
        Estimate answer correctness through lexical overlap.

        Returns None when reference evidence is unavailable.
        """

        if reference_answer is None:
            return None

        if not reference_answer.strip():
            return None

        return cls._lexical_overlap(answer,reference_answer)

    @staticmethod
    def citation_coverage(response: RAGResponse,expected_source_ids: Iterable[str]) -> float:
        """Calculate expected-source citation coverage."""

        if not isinstance(response,RAGResponse):
            raise TypeError("response must be a RAGResponse.")

        expected = set(expected_source_ids)

        if not expected:
            return 0.0

        cited = {
            source.stable_id
            for source in response.sources
        }

        return len(
            cited.intersection(expected)
        ) / len(expected)

    
    # Complete RAG evaluation
    

    def evaluate_rag_response(self,response: RAGResponse,query: EvaluationQuery,context_blocks: Iterable[str],latency_seconds: float | None = None,failed: bool = False,input_tokens: int | None = None,output_tokens: int | None = None,total_tokens: int | None = None,estimated_cost: float | None = None) -> RAGEvaluation:
        """
        Evaluate a generated RAG response.

        Quality and operational metadata are captured together for
        regression evaluation.
        """

        if not isinstance(response,RAGResponse):
            raise TypeError(
                "response must be a RAGResponse."
            )

        if not isinstance(query,EvaluationQuery):
            raise TypeError(
                "query must be an EvaluationQuery."
            )

        if (latency_seconds is not None and latency_seconds < 0):
            raise ValueError(
                "latency_seconds cannot be negative."
            )

        if (input_tokens is not None and input_tokens < 0):
            raise ValueError(
                "input_tokens cannot be negative."
            )

        if (output_tokens is not None and output_tokens < 0):
            raise ValueError(
                "output_tokens cannot be negative."
            )

        if (total_tokens is not None and total_tokens < 0):
            raise ValueError(
                "total_tokens cannot be negative."
            )

        if (estimated_cost is not None and estimated_cost < 0):
            raise ValueError(
                "estimated_cost cannot be negative."
            )

        context_blocks = list(
            context_blocks
        )

        return RAGEvaluation(
            query_id=query.query_id,
            query=query.query,
            context_relevance=(
                self.context_relevance(
                    query.query,
                    context_blocks,
                )
            ),
            faithfulness=self.faithfulness(
                response.answer,
                context_blocks,
            ),
            answer_correctness=(
                self.answer_correctness(
                    response.answer,
                    query.reference_answer,
                )
            ),
            citation_coverage=(
                self.citation_coverage(
                    response,
                    query.expected_source_ids,
                )
            ),
            latency_seconds=latency_seconds,
            failed=failed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
        )

    
    # Regression comparison
    

    @staticmethod
    def compare_regression(baseline: dict[str, float],candidate: dict[str, float],minimum_change: float = 0.05) -> RegressionComparison:
        """
        Compare candidate metrics against baseline metrics.

        A metric is a regression when it drops by at least the configured
        minimum change, improved when it rises by at least that amount,
        and unchanged otherwise.
        """

        if minimum_change < 0:
            raise ValueError(
                "minimum_change cannot be negative.")

        if set(baseline) != set(candidate):
            raise ValueError(
                "Baseline and candidate must contain "
                "the same metric keys.")

        epsilon = 1e-12

        metric_deltas = {
            key: candidate[key] - baseline[key]
            for key in sorted(baseline)
        }

        regressions: list[str] = []
        improved: list[str] = []
        unchanged: list[str] = []

        for key, delta in metric_deltas.items():
            if delta <= (-minimum_change + epsilon):
                regressions.append(key)

            elif delta >= (minimum_change - epsilon):
                improved.append(key)

            else:
                unchanged.append(key)

        return RegressionComparison(
            metric_deltas=metric_deltas,
            regressions=tuple(regressions),
            improved=tuple(improved),
            unchanged=tuple(unchanged),
            passed=not regressions,
        )

    
    # Serialization
    

    @staticmethod
    def to_dict(summary: EvaluationSummary) -> dict[str, Any]:
        """Convert retrieval summary to JSON-compatible data."""

        return {
            "query_count": summary.query_count,
            "recall_at_k": {
                str(k): value
                for k, value in summary.recall_at_k.items()
            },
            "precision_at_k": {
                str(k): value
                for k, value in summary.precision_at_k.items()
            },
            "mean_reciprocal_rank": (
                summary.mean_reciprocal_rank
            ),
            "mean_source_correctness": (
                summary.mean_source_correctness
            ),
        }

    @staticmethod
    def rag_to_dict(evaluation: RAGEvaluation,) -> dict[str, Any]:
        """Convert RAG evaluation to JSON-compatible data."""

        result: dict[str, Any] = {
            "context_relevance": (
                evaluation.context_relevance
            ),
            "faithfulness": (
                evaluation.faithfulness
            ),
            "answer_correctness": (
                evaluation.answer_correctness
            ),
            "citation_coverage": (
                evaluation.citation_coverage
            ),
            "latency_seconds": (
                evaluation.latency_seconds
            ),
            "failed": evaluation.failed,
            "input_tokens": evaluation.input_tokens,
            "output_tokens": evaluation.output_tokens,
            "total_tokens": evaluation.total_tokens,
            "estimated_cost": evaluation.estimated_cost,
        }

        if evaluation.query_id is not None:
            result["query_id"] = evaluation.query_id

        if evaluation.query is not None:
            result["query"] = evaluation.query

        return result

    @staticmethod
    def regression_to_dict(comparison: RegressionComparison) -> dict[str, Any]:
        """Convert regression comparison to JSON-compatible data."""

        return {
            "metric_deltas": dict(
                comparison.metric_deltas
            ),
            "regressions": list(
                comparison.regressions
            ),
            "improved": list(
                comparison.improved
            ),
            "unchanged": list(
                comparison.unchanged
            ),
            "passed": comparison.passed,
        }