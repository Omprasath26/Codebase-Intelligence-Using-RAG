"""Persistent ingestion manifest and checkpoint state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ManifestArtifact(BaseModel):
    """Persisted state for one repository artifact."""

    stable_id: str
    source_path_or_object_id: str
    artifact_type: str
    status: str
    commit_sha: str | None = None
    source_sha: str | None = None
    source_url: str | None = None
    exclusion_reason: str | None = None
    failure_reason: str | None = None


class IngestionCheckpoint(BaseModel):
    """Checkpoint describing resumable ingestion progress."""

    run_id: str
    repository: str
    ref: str
    commit_sha: str
    last_processed_index: int = -1
    status: str = "in_progress"
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class IngestionManifest(BaseModel):
    """Complete persisted manifest for one ingestion run."""

    run_id: str
    repository: str
    ref: str
    commit_sha: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    status: str = "in_progress"
    artifacts: list[ManifestArtifact] = Field(
        default_factory=list
    )
    checkpoint: IngestionCheckpoint | None = None

    def add_artifact(self,artifact: ManifestArtifact) -> None:
        """Add one artifact record to the manifest."""

        self.artifacts.append(artifact)
        self.updated_at = datetime.now(timezone.utc)

    def update_checkpoint(self,last_processed_index: int,status: str = "in_progress") -> None:
        """Update the resumable ingestion checkpoint."""

        if self.checkpoint is None:
            self.checkpoint = IngestionCheckpoint(
                run_id=self.run_id,
                repository=self.repository,
                ref=self.ref,
                commit_sha=self.commit_sha,
            )

        self.checkpoint.last_processed_index = (
            last_processed_index
        )
        self.checkpoint.status = status
        self.checkpoint.updated_at = datetime.now(
            timezone.utc
        )
        self.updated_at = datetime.now(timezone.utc)

    def record_artifact(self,artifact: ManifestArtifact,processed_index: int) -> None:
        """Record an artifact and advance the ingestion checkpoint."""

        self.add_artifact(artifact)

        self.update_checkpoint(
            last_processed_index=processed_index,
            status="in_progress",
        )

    def resume_from_index(self) -> int:
        """Return the next artifact index to process after a checkpoint."""

        if self.checkpoint is None:
            return 0

        return self.checkpoint.last_processed_index + 1

    def mark_completed(self) -> None:
        """Mark the ingestion run as successfully completed."""

        self.status = "completed"

        self.update_checkpoint(
            last_processed_index=len(self.artifacts) - 1,
            status="completed",
        )

    def mark_failed(self,reason: str) -> None:
        """Mark the ingestion run as failed."""

        self.status = "failed"

        self.update_checkpoint(
            last_processed_index=(
                self.checkpoint.last_processed_index
                if self.checkpoint
                else -1
            ),
            status="failed",
        )

        self.add_artifact(
            ManifestArtifact(
                stable_id=f"run-failure:{self.run_id}",
                source_path_or_object_id="__ingestion_run__",
                artifact_type="other",
                status="failed",
                commit_sha=self.commit_sha,
                failure_reason=reason,
            )
        )

    def save(self,path: str | Path) -> None:
        """Persist the manifest as machine-readable JSON."""

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls,path: str | Path) -> "IngestionManifest":
        """Load a previously persisted manifest."""

        input_path = Path(path)

        data: dict[str, Any] = json.loads(
            input_path.read_text(
                encoding="utf-8"
            )
        )

        return cls.model_validate(data)