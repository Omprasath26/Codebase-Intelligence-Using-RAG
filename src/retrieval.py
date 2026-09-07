"""Hybrid retrieval for the codebase intelligence system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.code_processing import KnowledgeChunk
from src.query_analysis import QueryAnalysis


@dataclass(frozen=True)
class RetrievedEvidence:
    """Retrieved knowledge chunk with ranking and provenance metadata."""

    chunk: KnowledgeChunk
    score: float
    retrieval_method: str


class Retrieval:
    """Retrieve knowledge chunks using dense and lexical search."""

    def __init__(
        self,
        index_state: dict[str, Any],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rrf_k: int = 60,
    ) -> None:
        """Initialize retrieval from an existing indexing state."""

        if not index_state:
            raise ValueError("Index state cannot be empty.")

        self.faiss_index = index_state["faiss"]
        self.bm25_index: BM25Okapi = index_state["bm25"]
        self.chunks: list[KnowledgeChunk] = index_state["chunks"]

        self.chunk_metadata: dict[int, dict[str, Any]] = (
            index_state["chunk_metadata"]
        )

        self.embedding_model = SentenceTransformer(
            embedding_model
        )

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        self.rrf_k = rrf_k

        if len(self.chunks) != self.faiss_index.ntotal:
            raise ValueError(
                "FAISS index size does not match chunk count."
            )

    def retrieve(
        self,
        query_analysis: QueryAnalysis,
        top_k: int = 5,
        candidate_k: int | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[RetrievedEvidence]:
        """Retrieve and rank evidence for an analyzed query."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not self.chunks:
            return []

        candidate_count = candidate_k or max(top_k * 3, 10)
        candidate_count = min(
            candidate_count,
            len(self.chunks),
        )

        dense_results = self._dense_search(
            query_analysis.normalized_query,
            candidate_count,
        )

        lexical_results = self._bm25_search(
            query_analysis.normalized_query,
            candidate_count,
        )

        fused_results = self._fuse_results(
            dense_results,
            lexical_results,
        )

        filtered_results = self._apply_metadata_filters(
            fused_results,
            metadata_filters,
        )

        return filtered_results[:top_k]

    def _dense_search(
        self,
        query: str,
        candidate_k: int,
    ) -> list[tuple[int, float]]:
        """Search the FAISS index using a normalized query embedding."""

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_vector = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        scores, indices = self.faiss_index.search(
            query_vector,
            candidate_k,
        )

        results: list[tuple[int, float]] = []

        for index, score in zip(
            indices[0],
            scores[0],
        ):
            if index < 0:
                continue

            results.append(
                (
                    int(index),
                    float(score),
                )
            )

        return results

    def _bm25_search(
        self,
        query: str,
        candidate_k: int,
    ) -> list[tuple[int, float]]:
        """Search the BM25 index using the same lexical tokenization as indexing."""

        tokens = self._tokenize(query)

        if not tokens:
            return []

        scores = self.bm25_index.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (
                -float(scores[index]),
                index,
            ),
        )

        return [
            (
                index,
                float(scores[index]),
            )
            for index in ranked_indices[:candidate_k]
        ]

    def _fuse_results(
        self,
        dense_results: list[tuple[int, float]],
        lexical_results: list[tuple[int, float]],
    ) -> list[RetrievedEvidence]:
        """Fuse dense and lexical rankings using Reciprocal Rank Fusion."""

        fused_scores: dict[int, float] = {}
        methods: dict[int, set[str]] = {}

        for rank, (index, _) in enumerate(
            dense_results,
            start=1,
        ):
            fused_scores[index] = (
                fused_scores.get(index, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )
            methods.setdefault(index, set()).add("dense")

        for rank, (index, _) in enumerate(
            lexical_results,
            start=1,
        ):
            fused_scores[index] = (
                fused_scores.get(index, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )
            methods.setdefault(index, set()).add("bm25")

        ranked_indices = sorted(
            fused_scores,
            key=lambda index: (
                -fused_scores[index],
                index,
            ),
        )

        results: list[RetrievedEvidence] = []

        for index in ranked_indices:
            method = "+".join(
                sorted(methods[index])
            )

            results.append(
                RetrievedEvidence(
                    chunk=self.chunks[index],
                    score=fused_scores[index],
                    retrieval_method=method,
                )
            )

        return results

    def _apply_metadata_filters(
        self,
        results: list[RetrievedEvidence],
        metadata_filters: dict[str, Any] | None,
    ) -> list[RetrievedEvidence]:
        """Apply exact metadata filters to retrieved evidence."""

        if not metadata_filters:
            return results

        filtered: list[RetrievedEvidence] = []

        for result in results:
            metadata = self.chunk_metadata.get(
                self.chunks.index(result.chunk),
                {},
            )

            if self._matches_filters(
                metadata,
                metadata_filters,
            ):
                filtered.append(result)

        return filtered

    def _matches_filters(
        self,
        metadata: dict[str, Any],
        metadata_filters: dict[str, Any],
    ) -> bool:
        """Determine whether metadata satisfies all requested filters."""

        for key, expected in metadata_filters.items():
            actual = metadata.get(key)

            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False

        return True

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize query text consistently with the indexing stage."""

        import re

        return re.findall(
            r"[A-Za-z0-9_./:-]+",
            text.lower(),
        )