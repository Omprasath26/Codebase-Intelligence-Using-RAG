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


class Indexing:
    """Generate embeddings and build FAISS and BM25 indexes."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_version: str = "v1",
    ) -> None:
        self.embedding_model_name = embedding_model
        self.index_version = index_version
        self.embedding_model = SentenceTransformer(
            embedding_model
        )

    def build(
        self,
        chunks: list[KnowledgeChunk],
    ) -> dict[str, Any]:
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

        metadata = IndexMetadata(
            index_version=self.index_version,
            embedding_model=self.embedding_model_name,
            embedding_dimension=int(
                embeddings.shape[1]
            ),
            chunk_count=len(ordered_chunks),
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
            "metadata": metadata,
        }

    def _generate_embeddings(
        self,
        texts: list[str],
    ) -> np.ndarray:
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

    def _build_faiss_index(
        self,
        embeddings: np.ndarray,
    ) -> faiss.Index:
        """Build a cosine-similarity FAISS index."""

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        return index

    def _build_bm25_index(
        self,
        texts: list[str],
    ) -> BM25Okapi:
        """Build a BM25 lexical index."""

        tokenized_documents = [
            self._tokenize(text)
            for text in texts
        ]

        return BM25Okapi(tokenized_documents)

    def _build_index_text(
        self,
        chunk: KnowledgeChunk,
    ) -> str:
        """Create deterministic searchable text from a knowledge chunk."""

        searchable_metadata = [
            chunk.source_path_or_object_id,
            chunk.symbol or "",
            chunk.parent_symbol or "",
            chunk.module or "",
            chunk.artifact_type,
            chunk.language or "",
        ]

        return "\n".join(
            [
                *searchable_metadata,
                chunk.content,
            ]
        )

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:
        """Tokenize text for lexical retrieval."""

        return re.findall(
            r"[A-Za-z0-9_./:-]+",
            text.lower(),
        )

    def _chunk_metadata(
        self,
        chunk: KnowledgeChunk,
    ) -> dict[str, Any]:
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
        }

    def build_id(
        self,
        chunks: list[KnowledgeChunk],
    ) -> str:
        """Generate a deterministic identifier for an index build."""

        ordered_ids = sorted(
            chunk.stable_id
            for chunk in chunks
        )

        identity = "|".join(
            [
                self.index_version,
                self.embedding_model_name,
                *ordered_ids,
            ]
        )

        return hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()