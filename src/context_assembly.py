"""Context assembly and evidence sufficiency for grounded generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.code_processing import KnowledgeChunk
from src.retrieval import RetrievedEvidence


@dataclass(frozen=True)
class ContextBlock:
    """A selected context block with attached source provenance."""

    chunk: KnowledgeChunk
    score: float
    retrieval_method: str
    dense_score: float | None
    bm25_score: float | None
    rerank_score: float | None
    expansion_type: str


@dataclass(frozen=True)
class EvidencePackage:
    """Final evidence package passed to the generation stage."""

    query: str
    context_blocks: list[ContextBlock]
    sufficient: bool
    limitation: str | None
    repository_revision: str | None
    total_characters: int


class ContextAssembly:
    """Expand, deduplicate, budget, and validate retrieved evidence."""

    def __init__(self,chunks: list[KnowledgeChunk],max_context_characters: int = 12000,neighbor_limit: int = 1) -> None:
        """Initialize context assembly from the processed knowledge chunks."""

        if max_context_characters <= 0:
            raise ValueError(
                "max_context_characters must be greater than zero."
            )

        if neighbor_limit < 0:
            raise ValueError(
                "neighbor_limit cannot be negative."
            )

        self.chunks = list(chunks)
        self.max_context_characters = max_context_characters
        self.neighbor_limit = neighbor_limit

        self._chunks_by_id = {
            chunk.stable_id: chunk
            for chunk in self.chunks
        }

        self._chunks_by_artifact: dict[str, list[KnowledgeChunk]] = {}

        for chunk in self.chunks:
            self._chunks_by_artifact.setdefault(
                chunk.artifact_id,
                [],
            ).append(chunk)

        for artifact_chunks in self._chunks_by_artifact.values():
            artifact_chunks.sort(
                key=lambda chunk: (
                    chunk.start_line
                    if chunk.start_line is not None
                    else float("inf"),
                    chunk.end_line
                    if chunk.end_line is not None
                    else float("inf"),
                    chunk.stable_id,
                )
            )

    def assemble(self,query: str,evidence: list[RetrievedEvidence], query_analysis: Any | None = None) -> EvidencePackage:
        """Build the final evidence package deterministically."""

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        if not query.strip():
            raise ValueError("query cannot be empty.")

        if not evidence:
            return EvidencePackage(
                query=query,
                context_blocks=[],
                sufficient=False,
                limitation="No repository evidence was retrieved.",
                repository_revision=None,
                total_characters=0,
            )

        expanded = self._expand_evidence(evidence)
        deduplicated = self._deduplicate(expanded)

        revision, revision_error = self._check_revision_consistency(
            deduplicated
        )

        if revision_error is not None:
            budgeted = self._apply_budget(deduplicated)

            return EvidencePackage(
                query=query,
                context_blocks=budgeted,
                sufficient=False,
                limitation=revision_error,
                repository_revision=None,
                total_characters=self._total_characters(budgeted),
            )

        budgeted = self._apply_budget(deduplicated)

        limitation = self._determine_sufficiency_limitation(
            query_analysis=query_analysis,
            context_blocks=budgeted,
        )

        return EvidencePackage(
            query=query,
            context_blocks=budgeted,
            sufficient=limitation is None,
            limitation=limitation,
            repository_revision=revision,
            total_characters=self._total_characters(budgeted),
        )

    def _expand_evidence(self,evidence: list[RetrievedEvidence]) -> list[ContextBlock]:
        """Expand retrieved evidence with justified contextual chunks."""

        blocks: list[ContextBlock] = []

        for item in evidence:
            blocks.append(
                self._to_context_block(
                    item,
                    expansion_type="retrieved",
                )
            )

            chunk = item.chunk

            parent = self._find_parent_chunk(chunk)

            if parent is not None:
                blocks.append(
                    ContextBlock(
                        chunk=parent,
                        score=item.score,
                        retrieval_method=item.retrieval_method,
                        dense_score=item.dense_score,
                        bm25_score=item.bm25_score,
                        rerank_score=item.rerank_score,
                        expansion_type="parent",
                    )
                )

            for neighbor in self._find_neighbors(chunk):
                blocks.append(
                    ContextBlock(
                        chunk=neighbor,
                        score=item.score,
                        retrieval_method=item.retrieval_method,
                        dense_score=item.dense_score,
                        bm25_score=item.bm25_score,
                        rerank_score=item.rerank_score,
                        expansion_type="neighbor",
                    )
                )

            for related in self._find_related_chunks(chunk):
                blocks.append(
                    ContextBlock(
                        chunk=related,
                        score=item.score,
                        retrieval_method=item.retrieval_method,
                        dense_score=item.dense_score,
                        bm25_score=item.bm25_score,
                        rerank_score=item.rerank_score,
                        expansion_type="related",
                    )
                )

        return blocks

    def _to_context_block(self,evidence: RetrievedEvidence,expansion_type: str) -> ContextBlock:
        """Convert retrieved evidence into a provenance-preserving block."""

        return ContextBlock(
            chunk=evidence.chunk,
            score=evidence.score,
            retrieval_method=evidence.retrieval_method,
            dense_score=evidence.dense_score,
            bm25_score=evidence.bm25_score,
            rerank_score=evidence.rerank_score,
            expansion_type=expansion_type,
        )

    def _find_parent_chunk(self,chunk: KnowledgeChunk) -> KnowledgeChunk | None:
        """Find a parent symbol chunk from the same artifact."""

        if not chunk.parent_symbol:
            return None

        candidates = self._chunks_by_artifact.get(
            chunk.artifact_id,
            [],
        )

        matches = [
            candidate
            for candidate in candidates
            if candidate.symbol == chunk.parent_symbol
            and candidate.parent_symbol is None
        ]

        if not matches:
            return None

        return sorted(
            matches,
            key=lambda candidate: candidate.stable_id,
        )[0]

    def _find_neighbors( self, chunk: KnowledgeChunk) -> list[KnowledgeChunk]:
        """Find nearby structural chunks from the same artifact."""

        if self.neighbor_limit == 0:
            return []

        candidates = self._chunks_by_artifact.get(
            chunk.artifact_id,
            [],
        )

        try:
            position = next(
                index
                for index, candidate in enumerate(candidates)
                if candidate.stable_id == chunk.stable_id
            )
        except StopIteration:
            return []

        neighbors: list[KnowledgeChunk] = []

        for offset in range(1, self.neighbor_limit + 1):
            for neighbor_position in (
                position - offset,
                position + offset,
            ):
                if 0 <= neighbor_position < len(candidates):
                    neighbor = candidates[neighbor_position]

                    if neighbor.stable_id != chunk.stable_id:
                        neighbors.append(neighbor)

        return sorted(
            neighbors,
            key=lambda candidate: (
                candidate.start_line
                if candidate.start_line is not None
                else float("inf"),
                candidate.stable_id,
            ),
        )

    def _find_related_chunks(self,chunk: KnowledgeChunk) -> list[KnowledgeChunk]:
        """Follow only explicit relationships stored in chunk metadata."""

        related_ids = self._extract_known_chunk_ids(
            chunk.metadata
        )

        related_chunks = [
            self._chunks_by_id[chunk_id]
            for chunk_id in related_ids
            if chunk_id in self._chunks_by_id
            and chunk_id != chunk.stable_id
        ]

        return sorted(
            related_chunks,
            key=lambda candidate: candidate.stable_id,
        )

    def _extract_known_chunk_ids(self,value: Any) -> set[str]:
        """Extract IDs only when metadata points to known chunks."""

        found: set[str] = set()

        if isinstance(value, str):
            if value in self._chunks_by_id:
                found.add(value)

            return found

        if isinstance(value, dict):
            for nested_value in value.values():
                found.update(
                    self._extract_known_chunk_ids(nested_value)
                )

            return found

        if isinstance(value, (list, tuple, set)):
            for nested_value in value:
                found.update(
                    self._extract_known_chunk_ids(nested_value)
                )

        return found

    def _deduplicate(self,blocks: list[ContextBlock]) -> list[ContextBlock]:
        """Remove duplicate chunks while preserving best evidence order."""

        selected: dict[str, ContextBlock] = {}

        for block in blocks:
            chunk_id = block.chunk.stable_id

            if chunk_id not in selected:
                selected[chunk_id] = block
                continue

            existing = selected[chunk_id]

            if self._block_priority(block) < self._block_priority(
                existing
            ):
                selected[chunk_id] = block

        return list(selected.values())

    def _block_priority(self,block: ContextBlock) -> tuple[int, float, str]:
        """Return deterministic priority for duplicate blocks."""

        expansion_priority = {
            "retrieved": 0,
            "parent": 1,
            "related": 2,
            "neighbor": 3,
        }

        return (
            expansion_priority.get(
                block.expansion_type,
                99,
            ),
            -block.score,
            block.chunk.stable_id,
        )

    def _check_revision_consistency( self,blocks: list[ContextBlock]) -> tuple[str | None, str | None]:
        """Reject context that silently mixes known repository revisions."""

        revisions = sorted(
            {
                block.chunk.commit_sha
                for block in blocks
                if block.chunk.commit_sha
            }
        )

        if len(revisions) > 1:
            return (
                None,
                "Retrieved evidence contains incompatible repository revisions.",
            )

        if len(revisions) == 1:
            return revisions[0], None

        return None, None

    def _apply_budget(self,blocks: list[ContextBlock]) -> list[ContextBlock]:
        """Select context deterministically without exceeding the budget."""

        selected: list[ContextBlock] = []
        total = 0

        for block in blocks:
            block_size = len(block.chunk.content)

            if not selected and block_size > self.max_context_characters:
                truncated_chunk = block.chunk.model_copy(
                    update={
                        "content": block.chunk.content[
                            : self.max_context_characters
                        ]
                    }
                )

                selected.append(
                    ContextBlock(
                        chunk=truncated_chunk,
                        score=block.score,
                        retrieval_method=block.retrieval_method,
                        dense_score=block.dense_score,
                        bm25_score=block.bm25_score,
                        rerank_score=block.rerank_score,
                        expansion_type=block.expansion_type,
                    )
                )

                total = self.max_context_characters
                break

            if total + block_size > self.max_context_characters:
                continue

            selected.append(block)
            total += block_size

        return selected

    def _determine_sufficiency_limitation(self,query_analysis: Any | None,context_blocks: list[ContextBlock]) -> str | None:
        """Determine whether the assembled evidence is sufficient."""

        if not context_blocks:
            return "No usable repository evidence remains after context assembly."

        if query_analysis is None:
            return None

        symbols = getattr(
            query_analysis,
            "symbols",
            [],
        )

        paths = getattr(
            query_analysis,
            "paths",
            [],
        )

        intent = getattr(
            query_analysis,
            "intent",
            "",
        )

        if symbols:
            symbol_matches = any(
                block.chunk.symbol in symbols
                or block.chunk.parent_symbol in symbols
                for block in context_blocks
            )

            if not symbol_matches:
                return (
                    "The retrieved context does not contain evidence "
                    f"for the requested symbol(s): {', '.join(symbols)}."
                )

        if paths:
            path_matches = any(
                any(
                    path in block.chunk.source_path_or_object_id
                    for path in paths
                )
                for block in context_blocks
            )

            if not path_matches:
                return (
                    "The retrieved context does not contain evidence "
                    f"for the requested path(s): {', '.join(paths)}."
                )

        if intent == "historical":
            historical_types = {
                "issue",
                "pull_request",
                "pr",
                "commit",
                "review",
            }

            has_history = any(
                block.chunk.artifact_type in historical_types
                for block in context_blocks
            )

            if not has_history:
                return (
                    "Historical evidence was requested, but no ingested "
                    "issue, pull request, commit, or review evidence "
                    "is available in the assembled context."
                )

        return None

    def _total_characters( self,blocks: list[ContextBlock]) -> int:
        """Return total selected context characters."""

        return sum(
            len(block.chunk.content)
            for block in blocks
        )