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

    def __init__(
        self,
        embedding_model: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        index_version: str = "v1",
        chunking_version: str = "v1") -> None:
        self.embedding_model_name = embedding_model
        self.index_version = index_version
        self.chunking_version = chunking_version

        self.embedding_model = SentenceTransformer(
            embedding_model
        )

        # The currently active, validated index state.
        self._active_index_state: (
            dict[str, Any] | None
        ) = None

        # The previous successfully active state.
        # It is retained as the last-known-good state.
        self._last_known_good_index_state: (
            dict[str, Any] | None
        ) = None

    @property
    def active_index_state(self) -> dict[str, Any] | None:
        """Return the currently active index state."""

        return self._active_index_state

    @property
    def last_known_good_index_state(self) -> dict[str, Any] | None:
        """Return the last successfully active index state."""

        return self._last_known_good_index_state

    def build(self,chunks: list[KnowledgeChunk]) -> dict[str, Any]:
        """Build deterministic vector and lexical indexes."""

        if not chunks:
            raise ValueError(
                "Cannot build indexes from empty chunks."
            )

        ordered_chunks = sorted(
            chunks,
            key=lambda chunk: chunk.stable_id,
        )

        texts = [
            self._build_index_text(chunk)
            for chunk in ordered_chunks
        ]

        embeddings = self._generate_embeddings(
            texts
        )

        faiss_index = self._build_faiss_index(
            embeddings
        )

        bm25_index = self._build_bm25_index(
            texts
        )

        chunk_metadata = {
            position: self._chunk_metadata(chunk)
            for position, chunk in enumerate(
                ordered_chunks
            )
        }

        chunk_id_by_position = {
            position: chunk.stable_id
            for position, chunk in enumerate(
                ordered_chunks
            )
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
            "chunk_id_by_position": (
                chunk_id_by_position
            ),
            "metadata": metadata,
            "pipeline_version": {
                "index_version": self.index_version,
                "embedding_model": (
                    self.embedding_model_name
                ),
                "chunking_version": (
                    self.chunking_version
                ),
            },
        }

    def prepare_incremental(self,previous_index_state: dict[str, Any],chunks: list[KnowledgeChunk],affected_chunk_ids: set[str],removed_chunk_ids: set[str] | None = None) -> dict[str, Any]:
        """Prepare a replacement index state for an incremental update."""

        if not previous_index_state:
            raise ValueError(
                "previous_index_state cannot be empty."
            )

        if not chunks:
            raise ValueError(
                "Cannot prepare an incremental index "
                "from empty chunks."
            )

        removed_ids = set(
            removed_chunk_ids or set()
        )

        affected_ids = set(
            affected_chunk_ids
        )

        previous_metadata = (
            previous_index_state.get(
                "metadata"
            )
        )

        if previous_metadata is None:
            raise ValueError(
                "Previous index state is missing metadata."
            )

        if (
            previous_metadata.index_version
            != self.index_version
        ):
            raise ValueError(
                "Index version is incompatible "
                "with the previous index."
            )

        if (
            previous_metadata.embedding_model
            != self.embedding_model_name
        ):
            raise ValueError(
                "Embedding model is incompatible "
                "with the previous index."
            )

        if (
            getattr(
                previous_metadata,
                "chunking_version",
                "v1",
            )
            != self.chunking_version
        ):
            raise ValueError(
                "Chunking version is incompatible "
                "with the previous index."
            )

        previous_chunks = (
            previous_index_state.get(
                "chunks",
                [],
            )
        )

        previous_embeddings = (
            previous_index_state.get(
                "embeddings"
            )
        )

        if previous_embeddings is None:
            raise ValueError(
                "Previous index state is missing embeddings."
            )

        if (
            len(previous_chunks)
            != len(previous_embeddings)):
            raise ValueError(
                "Previous index chunk count "
                "does not match embeddings."
            )

        previous_positions = {
            chunk.stable_id: position
            for position, chunk in enumerate(
                previous_chunks
            )
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

        for position, chunk in enumerate(
            ordered_chunks
        ):
            previous_position = (
                previous_positions.get(
                    chunk.stable_id
                )
            )

            if (
                chunk.stable_id in affected_ids
                or previous_position is None
            ):
                reembed_positions.append(
                    position
                )
                reembed_texts.append(
                    texts[position]
                )

        if reembed_texts:
            generated_embeddings = (
                self._generate_embeddings(
                    reembed_texts
                )
            )

            if generated_embeddings.ndim != 2:
                raise ValueError(
                    "Generated embeddings must "
                    "be a 2D array."
                )

            if (
                generated_embeddings.shape[1]
                != previous_embeddings.shape[1]
            ):
                raise ValueError(
                    "Embedding dimension is incompatible "
                    "with the previous index."
                )
        else:
            generated_embeddings = np.empty(
                (
                    0,
                    previous_embeddings.shape[1],
                ),
                dtype=np.float32,
            )

        embeddings = np.empty(
            (
                len(ordered_chunks),
                previous_embeddings.shape[1],
            ),
            dtype=np.float32,
        )

        generated_position = 0

        for position, chunk in enumerate( ordered_chunks):
            if position in reembed_positions:
                embeddings[position] = (
                    generated_embeddings[
                        generated_position
                    ]
                )

                generated_position += 1
                continue

            previous_position = (
                previous_positions.get(
                    chunk.stable_id
                )
            )

            if previous_position is None:
                raise ValueError(
                    "Unable to resolve embedding "
                    "for unchanged chunk "
                    f"{chunk.stable_id}."
                )

            embeddings[position] = (
                previous_embeddings[
                    previous_position
                ]
            )

        faiss_index = self._build_faiss_index(
            embeddings
        )

        bm25_index = self._build_bm25_index(
            texts
        )

        chunk_metadata = {
            position: self._chunk_metadata(chunk)
            for position, chunk in enumerate(
                ordered_chunks
            )
        }

        chunk_id_by_position = {
            position: chunk.stable_id
            for position, chunk in enumerate(
                ordered_chunks
            )
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

        reembedded_chunk_ids = [
            chunk.stable_id
            for position, chunk in enumerate(
                ordered_chunks
            )
            if position in reembed_positions
        ]

        logger.info(
            "Prepared incremental indexes for "
            "%d knowledge chunks; re-embedded "
            "%d chunks.",
            len(ordered_chunks),
            len(reembedded_chunk_ids),
        )

        return {
            "faiss": faiss_index,
            "bm25": bm25_index,
            "embeddings": embeddings,
            "chunks": ordered_chunks,
            "chunk_metadata": chunk_metadata,
            "chunk_id_by_position": (
                chunk_id_by_position
            ),
            "metadata": metadata,
            "pipeline_version": {
                "index_version": self.index_version,
                "embedding_model": (
                    self.embedding_model_name
                ),
                "chunking_version": (
                    self.chunking_version
                ),
            },
            "affected_chunk_ids": sorted(
                affected_ids
            ),
            "removed_chunk_ids": sorted(
                removed_ids
            ),
            "reembedded_chunk_ids": (
                reembedded_chunk_ids
            ),
        }

    def validate_index_state(
        self,
        index_state: dict[str, Any],
        expected_pipeline_version: (
            dict[str, str] | None
        ) = None) -> bool:
        """
        Validate a complete candidate index state.

        Validation happens before activation. Therefore an invalid
        candidate cannot replace the currently active state.
        """

        if not isinstance(index_state,dict):
            raise ValueError(
                "Index state must be a dictionary."
            )

        required_keys = {
            "faiss",
            "bm25",
            "embeddings",
            "chunks",
            "chunk_metadata",
            "chunk_id_by_position",
            "metadata",
            "pipeline_version",
        }

        missing_keys = sorted(
            required_keys.difference(
                index_state
            )
        )

        if missing_keys:
            raise ValueError(
                "Index state is missing required "
                "fields: "
                + ", ".join(missing_keys)
            )

        metadata = index_state[
            "metadata"
        ]

        if not isinstance(metadata,IndexMetadata):
            raise ValueError(
                "Index state metadata must "
                "be IndexMetadata."
            )

        pipeline_version = (
            index_state[
                "pipeline_version"
            ]
        )

        if not isinstance(pipeline_version,dict):
            raise ValueError(
                "Index state pipeline_version "
                "must be a dictionary."
            )

        current_pipeline_version = {
            "index_version": self.index_version,
            "embedding_model": (
                self.embedding_model_name
            ),
            "chunking_version": (
                self.chunking_version
            ),
        }

        if (
            pipeline_version
            != current_pipeline_version):
            raise ValueError(
                "Index state pipeline version "
                "is incompatible with the "
                "current indexing pipeline."
            )

        if (
            expected_pipeline_version is not None
            and pipeline_version
            != expected_pipeline_version):
            raise ValueError(
                "Index state pipeline version "
                "does not match the expected "
                "pipeline version."
            )

        chunks = index_state[
            "chunks"
        ]

        embeddings = index_state[
            "embeddings"
        ]

        faiss_index = index_state[
            "faiss"
        ]

        bm25_index = index_state[
            "bm25"
        ]

        chunk_metadata = index_state[
            "chunk_metadata"
        ]

        chunk_id_by_position = index_state[
            "chunk_id_by_position"
        ]

        if not isinstance(chunks,list):
            raise ValueError(
                "Index state chunks must be a list."
            )

        if not chunks:
            raise ValueError(
                "Index state must contain "
                "at least one chunk."
            )

        if not isinstance(embeddings,np.ndarray):
            raise ValueError(
                "Index state embeddings must "
                "be a numpy array."
            )

        if embeddings.ndim != 2:
            raise ValueError(
                "Index state embeddings must "
                "be a 2D array."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Index state embeddings contain "
                "non-finite values."
            )

        chunk_count = len(chunks)

        if (
            embeddings.shape[0]
            != chunk_count):
            raise ValueError(
                "Embedding count does not "
                "match chunk count."
            )

        if (
            metadata.chunk_count
            != chunk_count
        ):
            raise ValueError(
                "Metadata chunk count does not "
                "match actual chunk count."
            )

        if (
            metadata.embedding_dimension
            != embeddings.shape[1]
        ):
            raise ValueError(
                "Metadata embedding dimension "
                "does not match embeddings."
            )

        chunk_ids = [
            chunk.stable_id
            for chunk in chunks
        ]

        if (
            len(set(chunk_ids))
            != chunk_count):
            raise ValueError(
                "Index state contains "
                "duplicate chunk IDs."
            )

        if not isinstance(faiss_index,faiss.Index,):
            raise ValueError(
                "Index state FAISS object "
                "is invalid."
            )

        if (
            faiss_index.ntotal
            != chunk_count):
            raise ValueError(
                "FAISS vector count does not "
                "match chunk count."
            )

        if (
            faiss_index.d
            != embeddings.shape[1]):
            raise ValueError(
                "FAISS dimension does not "
                "match embedding dimension."
            )

        if not isinstance(bm25_index, BM25Okapi):
            raise ValueError(
                "Index state BM25 object "
                "is invalid."
            )

        # rank_bm25.BM25Okapi does not expose a
        # public `corpus` attribute in the installed
        # version. `doc_len` contains one entry per
        # indexed document and is therefore the correct
        # compatibility-safe document-count check.
        bm25_document_count = len(
            bm25_index.doc_len
        )

        if (
            bm25_document_count
            != chunk_count
        ):
            raise ValueError(
                "BM25 corpus count does not "
                "match chunk count."
            )

        expected_positions = {
            position: chunk.stable_id
            for position, chunk in enumerate(
                chunks
            )
        }

        if (
            chunk_id_by_position
            != expected_positions):
            raise ValueError(
                "Chunk-position mapping does "
                "not match index chunks."
            )

        if set(
            chunk_metadata
        ) != set(
            range(chunk_count)):
            raise ValueError(
                "Chunk metadata positions do "
                "not match chunk positions."
            )

        for position, chunk in enumerate(chunks):
            metadata_entry = (
                chunk_metadata[position]
            )

            if (
                metadata_entry.get(
                    "stable_id"
                )
                != chunk.stable_id
            ):
                raise ValueError(
                    "Chunk metadata does not "
                    "match chunk ID at "
                    f"position {position}."
                )

        logger.info(
            "Validated index state containing "
            "%d knowledge chunks.",
            chunk_count,
        )

        return True

    def commit_index_state(self,candidate_index_state: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and atomically activate a candidate state.

        The current active state is untouched if validation fails.
        Only after successful validation does the state switch occur.
        """

        self.validate_index_state(
            candidate_index_state
        )

        previous_active = (
            self._active_index_state
        )

        # Atomic state switch.
        #
        # No mutation of the active state happens before
        # validation succeeds.
        self._active_index_state = (
            candidate_index_state
        )

        if previous_active is not None:
            self._last_known_good_index_state = (
                previous_active
            )

        logger.info(
            "Committed new active index state."
        )

        return candidate_index_state

    def recover_last_known_good(self) -> dict[str, Any]:
        """
        Restore the last-known-good index state.
        """

        if (
            self._last_known_good_index_state is None):
            raise RuntimeError(
                "No last-known-good index state "
                "is available."
            )

        recovered_state = (
            self._last_known_good_index_state
        )

        self.validate_index_state(
            recovered_state
        )

        self._active_index_state = (
            recovered_state
        )

        # The recovered state is now the active
        # known-good state.
        self._last_known_good_index_state = (
            recovered_state
        )

        logger.warning(
            "Recovered the last-known-good "
            "index state."
        )

        return recovered_state

    def prepare_and_commit_incremental(self, previous_index_state: dict[str, Any],chunks: list[KnowledgeChunk],affected_chunk_ids: set[str],removed_chunk_ids: set[str] | None = None) -> dict[str, Any]:
        """
        Prepare, validate, and atomically activate
        an incremental replacement index.
        """

        candidate = self.prepare_incremental(
            previous_index_state=(
                previous_index_state
            ),
            chunks=chunks,
            affected_chunk_ids=(
                affected_chunk_ids
            ),
            removed_chunk_ids=(
                removed_chunk_ids
            ),
        )

        return self.commit_index_state(
            candidate
        )

    def rebuild_and_commit(self,chunks: list[KnowledgeChunk]) -> dict[str, Any]:
        """
        Build, validate, and atomically activate
        a complete replacement index.
        """

        candidate = self.build(
            chunks
        )

        return self.commit_index_state(
            candidate
        )

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

    def _build_faiss_index(self,embeddings: np.ndarray ) -> faiss.Index:
        """Build a cosine-similarity FAISS index."""

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(
            embeddings
        )

        return index

    def _build_bm25_index(self,texts: list[str]) -> BM25Okapi:
        """Build a BM25 lexical index."""

        tokenized_documents = [
            self._tokenize(text)
            for text in texts
        ]

        return BM25Okapi(
            tokenized_documents
        )

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

        enriched_metadata = (
            self._metadata_search_text(
                chunk.metadata
            )
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

            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            ):
                values.append(
                    str(value)
                )
                continue

            if isinstance(
                value,
                (
                    list,
                    tuple,
                    set,
                ),
            ):
                normalized_values = sorted(
                    str(item)
                    for item in value
                    if item is not None
                )

                values.extend(
                    normalized_values
                )
                continue

            if isinstance(value,dict):
                nested_values = (
                    self._metadata_search_text(
                        value
                    )
                )

                if nested_values:
                    values.append(
                        nested_values
                    )

                continue

            values.append(
                str(value)
            )

        return "\n".join(
            values
        )

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
            "metadata": dict(
                chunk.metadata
            ),
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

        return hashlib.sha256(identity.encode("utf-8")).hexdigest()