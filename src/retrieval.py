"""Hybrid retrieval and optional reranking for the codebase intelligence system."""

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
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


class Retrieval:
    """Retrieve knowledge chunks using dense, lexical, and optional reranked search."""

    def __init__(self,index_state: dict[str, Any],embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",rrf_k: int = 60,reranker_model: str | None = None,reranker: Any | None = None) -> None:
        """Initialize retrieval from an existing indexing state."""

        if not index_state:
            raise ValueError("Index state cannot be empty.")

        required_keys = {
            "faiss",
            "bm25",
            "chunks",
            "chunk_metadata",
        }

        missing_keys = required_keys.difference(index_state)

        if missing_keys:
            raise ValueError(
                "Index state is missing required keys: "
                + ", ".join(sorted(missing_keys))
            )

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

        if reranker is not None and reranker_model is not None:
            raise ValueError(
                "Provide either reranker or reranker_model, not both."
            )

        self.reranker = reranker

        if reranker_model is not None:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(reranker_model)

        if len(self.chunks) != self.faiss_index.ntotal:
            raise ValueError(
                "FAISS index size does not match chunk count."
            )

        self._validate_metadata_positions()

    def retrieve( self,query_analysis: QueryAnalysis,top_k: int = 5,candidate_k: int | None = None,metadata_filters: dict[str, Any] | None = None,rerank: bool = True,rerank_k: int | None = None) -> list[RetrievedEvidence]:
        """
        Retrieve evidence using dense + BM25 RRF fusion and optional reranking.

        Retrieval strategy:
        1. Retrieve candidate_k results independently from FAISS and BM25.
        2. Fuse the rankings using Reciprocal Rank Fusion.
        3. Apply metadata filters.
        4. Optionally rerank the merged candidate set.
        5. Return the deterministic top_k results.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if candidate_k is not None and candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than zero."
            )

        if rerank_k is not None and rerank_k <= 0:
            raise ValueError(
                "rerank_k must be greater than zero."
            )

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

        if not filtered_results:
            return []

        if rerank and self.reranker is not None:
            filtered_results = self._rerank_results(
                query_analysis.normalized_query,
                filtered_results,
                rerank_k=rerank_k,
            )

        return filtered_results[:top_k]

    def _dense_search(self,query: str,candidate_k: int) -> list[tuple[int, float]]:
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

    def _bm25_search(self,query: str,candidate_k: int) -> list[tuple[int, float]]:
        """Search the BM25 index using the indexing-stage tokenization."""

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

    def _fuse_results(self,dense_results: list[tuple[int, float]],lexical_results: list[tuple[int, float]]) -> list[RetrievedEvidence]:
        """
        Fuse dense and lexical rankings using Reciprocal Rank Fusion.

        RRF deliberately combines ranking positions rather than raw
        FAISS and BM25 scores because those scores have different scales.
        """

        fused_scores: dict[int, float] = {}
        methods: dict[int, set[str]] = {}
        dense_scores: dict[int, float] = {}
        bm25_scores: dict[int, float] = {}

        for rank, (index, score) in enumerate(
            dense_results,
            start=1,
        ):
            fused_scores[index] = (
                fused_scores.get(index, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

            dense_scores[index] = score
            methods.setdefault(index, set()).add("dense")

        for rank, (index, score) in enumerate(
            lexical_results,
            start=1,
        ):
            fused_scores[index] = (
                fused_scores.get(index, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

            bm25_scores[index] = score
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
                    dense_score=dense_scores.get(index),
                    bm25_score=bm25_scores.get(index),
                )
            )

        return results

    def _rerank_results(self, query: str,results: list[RetrievedEvidence],rerank_k: int | None = None) -> list[RetrievedEvidence]:
        """
        Rerank the merged candidate set with an optional CrossEncoder.

        Only the first rerank_k fused candidates are sent to the reranker.
        Remaining candidates retain their deterministic fused ordering.
        """

        if self.reranker is None:
            return results

        if not results:
            return []

        effective_rerank_k = rerank_k or len(results)

        effective_rerank_k = min(
            effective_rerank_k,
            len(results),
        )

        rerank_candidates = results[:effective_rerank_k]
        remaining_results = results[effective_rerank_k:]

        pairs = [
            (
                query,
                result.chunk.content,
            )
            for result in rerank_candidates
        ]

        scores = self.reranker.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked: list[RetrievedEvidence] = []

        scored_results = list(
            zip(
                rerank_candidates,
                scores,
            )
        )

        scored_results.sort(
            key=lambda item: (
                -float(item[1]),
                item[0].chunk.stable_id,
            )
        )

        for result, rerank_score in scored_results:
            reranked.append(
                RetrievedEvidence(
                    chunk=result.chunk,
                    score=float(rerank_score),
                    retrieval_method=(
                        result.retrieval_method
                        + "+reranker"
                    ),
                    dense_score=result.dense_score,
                    bm25_score=result.bm25_score,
                    rerank_score=float(rerank_score),
                )
            )

        return reranked + remaining_results

    def _apply_metadata_filters(self,results: list[RetrievedEvidence],metadata_filters: dict[str, Any] | None) -> list[RetrievedEvidence]:
        """Apply exact metadata filters to retrieved evidence."""

        if not metadata_filters:
            return results

        filtered: list[RetrievedEvidence] = []

        for result in results:
            index = self._chunk_index(result.chunk)

            metadata = self.chunk_metadata.get(
                index,
                {},
            )

            if self._matches_filters(
                metadata,
                metadata_filters,
            ):
                filtered.append(result)

        return filtered

    def _matches_filters(self,metadata: dict[str, Any],metadata_filters: dict[str, Any]) -> bool:
        """Determine whether metadata satisfies all requested filters."""

        for key, expected in metadata_filters.items():
            actual = metadata.get(key)

            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False

            elif actual != expected:
                return False

        return True

    def _chunk_index(self,chunk: KnowledgeChunk) -> int:
        """Return the indexed position for a knowledge chunk."""

        for index, candidate in enumerate(self.chunks):
            if candidate.stable_id == chunk.stable_id:
                return index

        raise ValueError(
            f"Chunk '{chunk.stable_id}' is not present in the index."
        )

    def _validate_metadata_positions(self) -> None:
        """Validate that metadata exists for every indexed chunk."""

        missing_positions = [
            index
            for index in range(len(self.chunks))
            if index not in self.chunk_metadata
        ]

        if missing_positions:
            raise ValueError(
                "Chunk metadata is missing for index positions: "
                + ", ".join(
                    str(index)
                    for index in missing_positions
                )
            )

    def _tokenize(self,text: str) -> list[str]:
        """Tokenize query text consistently with the indexing stage."""

        import re

        return re.findall(
            r"[A-Za-z0-9_./:-]+",
            text.lower(),
        )