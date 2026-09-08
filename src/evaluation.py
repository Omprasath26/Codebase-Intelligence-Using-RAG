"""Evaluation utilities for retrieval quality and ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.retrieval import RetrievedEvidence


@dataclass(frozen=True)
class EvaluationQuery:
    """One retrieval evaluation query and its relevant chunk IDs."""

    query_id: str
    query: str
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class QueryEvaluation:
    """Evaluation metrics for one query."""

    query_id: str
    query: str
    retrieved_chunk_ids: tuple[str, ...]
    relevant_chunk_ids: frozenset[str]
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate retrieval evaluation metrics."""

    query_count: int
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mean_reciprocal_rank: float


class RetrievalEvaluation:
    """Evaluate retrieval results against deterministic ground truth."""

    def __init__(
        self,
        dataset_path: str | Path | None = None,
    ) -> None:
        self.dataset_path = (
            Path(dataset_path)
            if dataset_path is not None
            else Path("data/evaluation/evaluation_dataset.json")
        )

    # ------------------------------------------------------------------
    # Ground-truth loading
    # ------------------------------------------------------------------

    def load_ground_truth(self) -> list[EvaluationQuery]:
        """Load and validate the retrieval ground-truth dataset."""

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
                "Evaluation dataset must contain a 'questions' list."
            )

        if not questions:
            raise ValueError(
                "Evaluation dataset must contain at least one question."
            )

        evaluation_queries: list[EvaluationQuery] = []

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

            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"{query_id}: 'query' must be a non-empty string."
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
                    f"{query_id}: duplicate relevant chunk IDs "
                    "are not allowed."
                )

            evaluation_queries.append(
                EvaluationQuery(
                    query_id=query_id.strip(),
                    query=query.strip(),
                    relevant_chunk_ids=frozenset(
                        normalized_ids
                    ),
                )
            )

        return evaluation_queries

    # ------------------------------------------------------------------
    # Result normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_chunk_ids(
        results: Iterable[RetrievedEvidence],
    ) -> tuple[str, ...]:
        """Extract stable chunk IDs in retrieval ranking order."""

        chunk_ids: list[str] = []
        seen: set[str] = set()

        for result in results:
            if not isinstance(result, RetrievedEvidence):
                raise TypeError(
                    "Retrieval results must contain RetrievedEvidence."
                )

            chunk_id = result.chunk.stable_id

            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            chunk_ids.append(chunk_id)

        return tuple(chunk_ids)

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    @staticmethod
    def recall_at_k(
        retrieved_chunk_ids: Iterable[str],
        relevant_chunk_ids: Iterable[str],
        k: int,
    ) -> float:
        """Calculate Recall@K."""

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        relevant = set(relevant_chunk_ids)

        if not relevant:
            return 0.0

        retrieved = set(
            list(retrieved_chunk_ids)[:k]
        )

        return len(
            retrieved.intersection(relevant)
        ) / len(relevant)

    # ------------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------------

    @staticmethod
    def precision_at_k(
        retrieved_chunk_ids: Iterable[str],
        relevant_chunk_ids: Iterable[str],
        k: int,
    ) -> float:
        """Calculate Precision@K."""

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        relevant = set(relevant_chunk_ids)
        retrieved = list(retrieved_chunk_ids)[:k]

        if not retrieved:
            return 0.0

        relevant_count = sum(
            chunk_id in relevant
            for chunk_id in retrieved
        )

        return relevant_count / len(retrieved)

    # ------------------------------------------------------------------
    # Mean Reciprocal Rank
    # ------------------------------------------------------------------

    @staticmethod
    def reciprocal_rank(
        retrieved_chunk_ids: Iterable[str],
        relevant_chunk_ids: Iterable[str],
    ) -> float:
        """Calculate Reciprocal Rank of the first relevant result."""

        relevant = set(relevant_chunk_ids)

        if not relevant:
            return 0.0

        for rank, chunk_id in enumerate(
            retrieved_chunk_ids,
            start=1,
        ):
            if chunk_id in relevant:
                return 1.0 / rank

        return 0.0

    # ------------------------------------------------------------------
    # One-query evaluation
    # ------------------------------------------------------------------

    def evaluate_query(
        self,
        evaluation_query: EvaluationQuery,
        results: Iterable[RetrievedEvidence],
        k_values: Iterable[int] = (1, 3, 5, 10),
    ) -> QueryEvaluation:
        """Evaluate one retrieval query."""

        k_values = tuple(k_values)

        if not k_values:
            raise ValueError(
                "At least one k value is required."
            )

        if any(k <= 0 for k in k_values):
            raise ValueError(
                "All k values must be greater than zero."
            )

        retrieved_chunk_ids = self._extract_chunk_ids(
            results
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

        reciprocal_rank = self.reciprocal_rank(
            retrieved_chunk_ids,
            evaluation_query.relevant_chunk_ids,
        )

        return QueryEvaluation(
            query_id=evaluation_query.query_id,
            query=evaluation_query.query,
            retrieved_chunk_ids=retrieved_chunk_ids,
            relevant_chunk_ids=evaluation_query.relevant_chunk_ids,
            recall_at_k=recall,
            precision_at_k=precision,
            reciprocal_rank=reciprocal_rank,
        )

    # ------------------------------------------------------------------
    # Aggregate evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(
        evaluations: Iterable[QueryEvaluation],
        k_values: Iterable[int] = (1, 3, 5, 10),
    ) -> EvaluationSummary:
        """Aggregate per-query retrieval metrics."""

        evaluations = list(evaluations)
        k_values = tuple(k_values)

        if not k_values:
            raise ValueError(
                "At least one k value is required."
            )

        if not evaluations:
            raise ValueError(
                "At least one query evaluation is required."
            )

        recall_at_k = {
            k: sum(
                evaluation.recall_at_k[k]
                for evaluation in evaluations
            )
            / len(evaluations)
            for k in k_values
        }

        precision_at_k = {
            k: sum(
                evaluation.precision_at_k[k]
                for evaluation in evaluations
            )
            / len(evaluations)
            for k in k_values
        }

        mean_reciprocal_rank = sum(
            evaluation.reciprocal_rank
            for evaluation in evaluations
        ) / len(evaluations)

        return EvaluationSummary(
            query_count=len(evaluations),
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mean_reciprocal_rank=mean_reciprocal_rank,
        )

    # ------------------------------------------------------------------
    # Evaluation result serialization
    # ------------------------------------------------------------------

    @staticmethod
    def to_dict(
        summary: EvaluationSummary,
    ) -> dict[str, Any]:
        """Convert an evaluation summary to JSON-compatible data."""

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
        }