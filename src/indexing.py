"""Indexing pipeline for knowledge chunks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.code_processing import KnowledgeChunk
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class IndexMetadata:
    """Metadata describing a deterministic index build."""

    index_version: str
    embedding_model: str
    embedding_dimension: int
    chunk_count: int
    chunking_version: str = "v1"


class Indexing:
    """Generate embeddings and build FAISS and BM25 indexes."""

    def __init__(self,embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",index_version: str = "v1",chunking_version: str = "v1") -> None:
        self.embedding_model_name = embedding_model
        self.index_version = index_version
        self.chunking_version = chunking_version
        self.embedding_model = SentenceTransformer(
            embedding_model
        )

    def build(self,chunks: list[KnowledgeChunk]) -> dict[str, Any]:
        """Build deterministic vector and lexical indexes."""

        if not chunks:
            raise ValueError("Cannot build indexes from empty chunks.")

        ordered_chunks = sorted(
            chunks,
            key=lambda chunk: chunk.stable_id,
        )

        texts = [
            self._build_index_text(chunk)
            for chunk in ordered_chunks
        ]

        embeddings = self._generate_embeddings(texts)

        faiss_index = self._build_faiss_index(
            embeddings
        )

        bm25_index = self._build_bm25_index(
            texts
        )

        chunk_metadata = {
            position: self._chunk_metadata(chunk)
            for position, chunk in enumerate(ordered_chunks)
        }

        chunk_id_by_position = {
            position: chunk.stable_id
            for position, chunk in enumerate(ordered_chunks)
        }

        metadata = IndexMetadata(
            index_version=self.index_version,
            embedding_model=self.embedding_model_name,
            embedding_dimension=int(
                embeddings.shape[1]
            ),
            chunk_count=len(ordered_chunks),
            chunking_version=self.chunking_version,
        )

        logger.info(
            "Built indexes for %d knowledge chunks.",
            len(ordered_chunks),
        )

        return {
            "faiss": faiss_index,
            "bm25": bm25_index,
            "embeddings": embeddings,
            "chunks": ordered_chunks,
            "chunk_metadata": chunk_metadata,
            "chunk_id_by_position": chunk_id_by_position,
            "metadata": metadata,
            "pipeline_version": {
                "index_version": self.index_version,
                "embedding_model": self.embedding_model_name,
                "chunking_version": self.chunking_version,
            },
        }

    def prepare_incremental(self, previous_index_state: dict[str, Any],chunks: list[KnowledgeChunk],affected_chunk_ids: set[str],removed_chunk_ids: set[str] | None = None) -> dict[str, Any]:
        """Prepare a replacement index state for an incremental update."""
        if not previous_index_state:
            raise ValueError("previous_index_state cannot be empty.")
        if not chunks:
            raise ValueError(
                "Cannot prepare an incremental index from empty chunks."
            )

        removed_ids = set(removed_chunk_ids or set())
        affected_ids = set(affected_chunk_ids)

        previous_metadata = previous_index_state.get("metadata")
        if previous_metadata is None:
            raise ValueError("Previous index state is missing metadata.")

        if previous_metadata.index_version != self.index_version:
            raise ValueError(
                "Index version is incompatible with the previous index."
            )
        if previous_metadata.embedding_model != self.embedding_model_name:
            raise ValueError(
                "Embedding model is incompatible with the previous index."
            )
        if getattr(previous_metadata, "chunking_version", "v1") != self.chunking_version:
            raise ValueError(
                "Chunking version is incompatible with the previous index."
            )

        previous_chunks = previous_index_state.get("chunks", [])
        previous_embeddings = previous_index_state.get("embeddings")
        if previous_embeddings is None:
            raise ValueError("Previous index state is missing embeddings.")
        if len(previous_chunks) != len(previous_embeddings):
            raise ValueError(
                "Previous index chunk count does not match embeddings."
            )

        previous_positions = {
            chunk.stable_id: position
            for position, chunk in enumerate(previous_chunks)
        }
        ordered_chunks = sorted(
            chunks,
            key=lambda chunk: chunk.stable_id,
        )
        texts = [
            self._build_index_text(chunk)
            for chunk in ordered_chunks
        ]

        reembed_positions: list[int] = []
        reembed_texts: list[str] = []
        for position, chunk in enumerate(ordered_chunks):
            previous_position = previous_positions.get(chunk.stable_id)
            if (
                chunk.stable_id in affected_ids
                or previous_position is None
                or chunk.stable_id in removed_ids
            ):
                reembed_positions.append(position)
                reembed_texts.append(texts[position])

        if reembed_texts:
            generated_embeddings = self._generate_embeddings(reembed_texts)
            if generated_embeddings.ndim != 2:
                raise ValueError("Generated embeddings must be a 2D array.")
            if generated_embeddings.shape[1] != previous_embeddings.shape[1]:
                raise ValueError(
                    "Embedding dimension is incompatible with the previous index."
                )
        else:
            generated_embeddings = np.empty(
                (0, previous_embeddings.shape[1]),
                dtype=np.float32,
            )

        embeddings = np.empty(
            (len(ordered_chunks), previous_embeddings.shape[1]),
            dtype=np.float32,
        )

        generated_position = 0
        for position, chunk in enumerate(ordered_chunks):
            if position in reembed_positions:
                embeddings[position] = generated_embeddings[generated_position]
                generated_position += 1
                continue

            previous_position = previous_positions.get(chunk.stable_id)
            if previous_position is None:
                raise ValueError(
                    "Unable to resolve embedding for unchanged chunk "
                    f"{chunk.stable_id}."
                )
            embeddings[position] = previous_embeddings[previous_position]

        faiss_index = self._build_faiss_index(embeddings)
        bm25_index = self._build_bm25_index(texts)
        chunk_metadata = {
            position: self._chunk_metadata(chunk)
            for position, chunk in enumerate(ordered_chunks)
        }
        chunk_id_by_position = {
            position: chunk.stable_id
            for position, chunk in enumerate(ordered_chunks)
        }
        metadata = IndexMetadata(
            index_version=self.index_version,
            embedding_model=self.embedding_model_name,
            embedding_dimension=int(embeddings.shape[1]),
            chunk_count=len(ordered_chunks),
            chunking_version=self.chunking_version,
        )

        logger.info(
            "Prepared incremental indexes for %d knowledge chunks; "
            "re-embedded %d chunks.",
            len(ordered_chunks),
            len(reembed_positions),
        )

        return {
            "faiss": faiss_index,
            "bm25": bm25_index,
            "embeddings": embeddings,
            "chunks": ordered_chunks,
            "chunk_metadata": chunk_metadata,
            "chunk_id_by_position": chunk_id_by_position,
            "metadata": metadata,
            "pipeline_version": {
                "index_version": self.index_version,
                "embedding_model": self.embedding_model_name,
                "chunking_version": self.chunking_version,
            },
            "affected_chunk_ids": sorted(affected_ids),
            "removed_chunk_ids": sorted(removed_ids),
            "reembedded_chunk_ids": [
                chunk.stable_id
                for chunk in ordered_chunks
                if chunk.stable_id in affected_ids
                or chunk.stable_id not in previous_positions
            ],
        }

    def _generate_embeddings(self,texts: list[str]) -> np.ndarray:
        """Generate normalized embeddings for index construction."""

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def _build_faiss_index(self,embeddings: np.ndarray) -> faiss.Index:
        """Build a cosine-similarity FAISS index."""

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        return index

    def _build_bm25_index(self,texts: list[str]) -> BM25Okapi:
        """Build a BM25 lexical index."""

        tokenized_documents = [
            self._tokenize(text)
            for text in texts
        ]

        return BM25Okapi(tokenized_documents)

    def _build_index_text(self,chunk: KnowledgeChunk) -> str:
        """Create deterministic searchable text from a knowledge chunk."""

        searchable_metadata = [
            chunk.source_path_or_object_id,
            chunk.symbol or "",
            chunk.parent_symbol or "",
            chunk.module or "",
            chunk.artifact_type,
            chunk.language or "",
        ]

        enriched_metadata = self._metadata_search_text(
            chunk.metadata
        )

        return "\n".join(
            [
                *searchable_metadata,
                enriched_metadata,
                chunk.content,
            ]
        )

    def _metadata_search_text(self,metadata: dict[str, Any]) -> str:
        """Convert enriched metadata into deterministic searchable text."""

        values: list[str] = []

        for key in sorted(metadata):
            value = metadata[key]

            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                values.append(str(value))
                continue

            if isinstance(value, (list, tuple, set)):
                normalized_values = sorted(
                    str(item)
                    for item in value
                    if item is not None
                )
                values.extend(normalized_values)
                continue

            if isinstance(value, dict):
                nested_values = self._metadata_search_text(value)
                if nested_values:
                    values.append(nested_values)
                continue

            values.append(str(value))

        return "\n".join(values)

    def _tokenize(self,text: str) -> list[str]:
        """Tokenize text for lexical retrieval."""

        return re.findall(
            r"[A-Za-z0-9_./:-]+",
            text.lower(),
        )

    def _chunk_metadata(self,chunk: KnowledgeChunk) -> dict[str, Any]:
        """Preserve chunk provenance and searchable metadata."""

        return {
            "stable_id": chunk.stable_id,
            "artifact_id": chunk.artifact_id,
            "repository": chunk.repository,
            "artifact_type": chunk.artifact_type,
            "source_path_or_object_id": (
                chunk.source_path_or_object_id
            ),
            "source_url": chunk.source_url,
            "commit_sha": chunk.commit_sha,
            "ref": chunk.ref,
            "language": chunk.language,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "module": chunk.module,
            "symbol": chunk.symbol,
            "parent_symbol": chunk.parent_symbol,
            "metadata": dict(chunk.metadata),
        }

    def build_id(self,chunks: list[KnowledgeChunk]) -> str:
        """Generate a deterministic identifier for an index build."""

        ordered_ids = sorted(
            chunk.stable_id
            for chunk in chunks
        )

        identity = "|".join(
            [
                self.index_version,
                self.embedding_model_name,
                self.chunking_version,
                *ordered_ids,
            ]
        )

        return hashlib.sha256(
            identity.encode("utf-8")).hexdigest()