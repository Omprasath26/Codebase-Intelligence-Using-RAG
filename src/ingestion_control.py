"""Ingestion control for filtering and normalizing repository artifacts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from src.config.settings import Settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class ArtifactDecision:
    """Represents whether a raw repository item is accepted or excluded."""

    accepted: bool
    reason: str


class RepositoryArtifact(BaseModel):
    """Canonical normalized repository artifact."""

    stable_id: str
    repository: str
    artifact_type: str
    source_path_or_object_id: str
    source_url: str | None = None
    content: str
    language: str | None = None

    # Repository revision used for this ingestion.
    commit_sha: str | None = None

    # Exact GitHub file/blob/object revision when available.
    source_sha: str | None = None

    ref: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    ingestion_timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionControl:
    """Filters, classifies, and normalizes raw repository artifacts."""

    _SECRET_PATTERNS = (
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        ),
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)"
            r"\s*[:=]\s*['\"]?[\w\-\/+=]{16,}"
        ),
        re.compile(r"(?i)\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    )

    _CODE_EXTENSIONS = {
        ".py",
    }

    _DOCUMENTATION_EXTENSIONS = {
        ".md",
        ".rst",
        ".txt",
    }

    _CONFIGURATION_EXTENSIONS = {
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".ini",
        ".cfg",
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def classify_artifact(
        self,
        raw_artifact: dict[str, Any],
    ) -> str:
        """Classify a raw repository artifact."""

        artifact_type = raw_artifact.get("artifact_type")

        if artifact_type:
            return artifact_type

        path = str(raw_artifact.get("path", "")).lower()

        if path:
            filename = path.rsplit("/", 1)[-1]
            extension = ""

            if "." in filename:
                extension = "." + filename.rsplit(".", 1)[-1]

            if self._is_test_path(path):
                return "test"

            if extension in self._CODE_EXTENSIONS:
                return "code"

            if extension in self._DOCUMENTATION_EXTENSIONS:
                return "documentation"

            if extension in self._CONFIGURATION_EXTENSIONS:
                return "configuration"

        return "other"

    def _is_test_path(self, path: str) -> bool:
        """Determine whether a repository path represents a test."""

        parts = path.lower().split("/")
        filename = parts[-1] if parts else ""

        return (
            "tests" in parts
            or "test" in parts
            or filename.startswith("test_")
            or filename.endswith("_test.py")
        )

    def evaluate_file(
        self,
        raw_artifact: dict[str, Any],
    ) -> ArtifactDecision:
        """Apply repository file eligibility rules."""

        path = str(raw_artifact.get("path", ""))

        if not path:
            return ArtifactDecision(
                accepted=False,
                reason="missing_source_path",
            )

        if self._is_excluded_directory(path):
            return ArtifactDecision(
                accepted=False,
                reason="excluded_directory",
            )

        extension = self._get_extension(path)

        if extension not in {
            item.lower()
            for item in self.settings.ingestion.allowed_extensions
        }:
            return ArtifactDecision(
                accepted=False,
                reason="unsupported_file_type",
            )

        size = raw_artifact.get("size")

        if isinstance(size, int):
            if size > self.settings.ingestion.max_file_size_bytes:
                return ArtifactDecision(
                    accepted=False,
                    reason="file_size_exceeds_limit",
                )

        content = raw_artifact.get("content")

        if not isinstance(content, str):
            return ArtifactDecision(
                accepted=False,
                reason="unsupported_or_invalid_content",
            )

        if self._contains_secret(content):
            return ArtifactDecision(
                accepted=False,
                reason="potential_secret_detected",
            )

        return ArtifactDecision(
            accepted=True,
            reason="accepted",
        )

    def _is_excluded_directory(
        self,
        path: str,
    ) -> bool:
        """Check whether a path contains a configured excluded directory."""

        path_parts = {
            part.lower()
            for part in path.replace("\\", "/").split("/")
            if part
        }

        excluded_directories = {
            item.lower().strip("/")
            for item in self.settings.ingestion.excluded_directories
        }

        return bool(path_parts.intersection(excluded_directories))

    def _get_extension(
        self,
        path: str,
    ) -> str:
        """Return the lowercase file extension."""

        filename = path.rsplit("/", 1)[-1]

        if "." not in filename:
            return ""

        return "." + filename.rsplit(".", 1)[-1].lower()

    def _contains_secret(
        self,
        content: str,
    ) -> bool:
        """Detect common secret or credential patterns."""

        return any(
            pattern.search(content)
            for pattern in self._SECRET_PATTERNS
        )

    def normalize_file(
        self,
        raw_artifact: dict[str, Any],
        repository: str,
        ref: str,
        commit_sha: str,
    ) -> RepositoryArtifact:
        """Normalize an accepted repository file."""

        decision = self.evaluate_file(raw_artifact)

        if not decision.accepted:
            raise ValueError(
                f"Artifact excluded: {decision.reason}"
            )

        path = str(raw_artifact["path"])
        content = str(raw_artifact["content"])

        artifact_type = self.classify_artifact(
            raw_artifact
        )

        stable_id = self._generate_stable_id(
            repository=repository,
            artifact_type=artifact_type,
            source_path_or_object_id=path,
        )

        ingestion_timestamp = datetime.now(
            timezone.utc
        )

        source_sha = raw_artifact.get("sha")

        return RepositoryArtifact(
            stable_id=stable_id,
            repository=repository,
            artifact_type=artifact_type,
            source_path_or_object_id=path,
            source_url=raw_artifact.get("source_url"),
            content=content,
            language=raw_artifact.get("language"),
            commit_sha=commit_sha,
            source_sha=source_sha,
            ref=ref,
            created_at=self._parse_timestamp(
                raw_artifact.get("created_at")
            ),
            updated_at=self._parse_timestamp(
                raw_artifact.get("updated_at")
            ),
            ingestion_timestamp=ingestion_timestamp,
            metadata={
                "size": raw_artifact.get("size"),
            },
        )

    def normalize_github_artifact(
        self,
        raw_artifact: dict[str, Any],
        repository: str,
        ref: str,
        artifact_type: str,
        object_id: str,
    ) -> RepositoryArtifact:
        """Normalize a non-file GitHub artifact such as an issue or PR."""

        content = str(
            raw_artifact.get("content")
            or raw_artifact.get("body")
            or ""
        )

        source_url = (
            raw_artifact.get("source_url")
            or raw_artifact.get("html_url")
        )

        stable_id = self._generate_stable_id(
            repository=repository,
            artifact_type=artifact_type,
            source_path_or_object_id=object_id,
        )

        return RepositoryArtifact(
            stable_id=stable_id,
            repository=repository,
            artifact_type=artifact_type,
            source_path_or_object_id=object_id,
            source_url=source_url,
            content=content,
            language=None,
            commit_sha=raw_artifact.get("commit_sha"),
            source_sha=raw_artifact.get("sha"),
            ref=ref,
            created_at=self._parse_timestamp(
                raw_artifact.get("created_at")
            ),
            updated_at=self._parse_timestamp(
                raw_artifact.get("updated_at")
            ),
            ingestion_timestamp=datetime.now(
                timezone.utc
            ),
            metadata={
                "github_object_id": raw_artifact.get("id"),
            },
        )

    def _generate_stable_id(
        self,
        repository: str,
        artifact_type: str,
        source_path_or_object_id: str,
    ) -> str:
        """Generate a deterministic identifier from stable source identity."""

        identity = (
            f"{repository}|"
            f"{artifact_type}|"
            f"{source_path_or_object_id}"
        )

        return hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

    def _parse_timestamp(
        self,
        value: Any,
    ) -> datetime | None:
        """Convert an ISO timestamp into a datetime when available."""

        if not value:
            return None

        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            try:
                return datetime.fromisoformat(
                    value.replace("Z", "+00:00")
                )
            except ValueError:
                logger.warning(
                    "Unable to parse timestamp: %s",
                    value,
                )

        return None