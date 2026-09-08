"""Incremental repository synchronization for Problem Statement 1."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.change_detection import ArtifactChange, ChangeDetector
from src.ingestion_control import IngestionControl, RepositoryArtifact
from src.ingestion_manifest import (
    IngestionManifest,
    ManifestArtifact,
)
from src.repository_connector import RepositoryConnector


class RepositorySync:
    """Orchestrate repository snapshot acquisition and incremental changes."""

    def __init__(self,connector: RepositoryConnector,ingestion_control: IngestionControl,change_detector: ChangeDetector | None = None) -> None:
        self.connector = connector
        self.ingestion_control = ingestion_control
        self.change_detector = change_detector or ChangeDetector()

    def synchronize(self,manifest_path: str | Path) -> list[ArtifactChange]:
        """Synchronize the repository against the previous manifest."""

        reference = self.connector.validate_reference()

        repository_metadata = (
            self.connector.fetch_repository_metadata(reference)
        )

        commit_sha = repository_metadata["commit_sha"]

        repository = (
            f"{repository_metadata['owner']}/"
            f"{repository_metadata['name']}"
        )

        ref = repository_metadata["ref"]

        previous_manifest = self._load_previous_manifest(
            manifest_path
        )

        current_manifest = IngestionManifest(
            run_id=str(uuid4()),
            repository=repository,
            ref=ref,
            commit_sha=commit_sha,
            status="in_progress",
        )

        tree = self.connector.fetch_tree(
            reference,
            commit_sha,
        )

        previous_by_path = self._previous_processed_by_path(
            previous_manifest
        )

        fetch_entries: list[dict[str, Any]] = []

        for entry in tree:
            if entry.get("type") != "blob":
                continue

            path = str(entry.get("path", ""))
            source_sha = entry.get("sha")

            raw_tree_artifact = {
                "path": path,
                "size": entry.get("size"),
            }

            decision = self.ingestion_control.evaluate_file(
                raw_tree_artifact
            )

            if not decision.accepted:
                current_manifest.record_artifact(
                    ManifestArtifact(
                        stable_id=f"excluded:{repository}:{path}",
                        source_path_or_object_id=path,
                        artifact_type="other",
                        status="excluded",
                        commit_sha=commit_sha,
                        source_sha=source_sha,
                        exclusion_reason=decision.reason,
                    ),
                    processed_index=len(
                        current_manifest.artifacts
                    ),
                )

                current_manifest.save(manifest_path)
                continue

            previous_artifact = previous_by_path.get(path)

            if (
                previous_artifact is not None
                and previous_artifact.source_sha == source_sha
            ):
                current_manifest.record_artifact(
                    self._carry_forward_artifact(
                        previous_artifact,
                        commit_sha=commit_sha,
                    ),
                    processed_index=len(
                        current_manifest.artifacts
                    ),
                )

                current_manifest.save(manifest_path)
                continue

            fetch_entries.append(
                {
                    "path": path,
                    "sha": source_sha,
                    "size": entry.get("size"),
                }
            )

        fetch_paths = [
            entry["path"]
            for entry in fetch_entries
        ]

        fetched_files = self.connector.fetch_files(
            reference,
            fetch_paths,
            commit_sha,
        )

        fetched_by_path = {
            str(item.get("path")): item
            for item in fetched_files
        }

        for entry in fetch_entries:
            path = entry["path"]
            raw_artifact = fetched_by_path.get(path)

            processed_index = len(
                current_manifest.artifacts
            )

            if raw_artifact is None:
                current_manifest.record_artifact(
                    ManifestArtifact(
                        stable_id=f"failed:{repository}:{path}",
                        source_path_or_object_id=path,
                        artifact_type="other",
                        status="failed",
                        commit_sha=commit_sha,
                        source_sha=entry.get("sha"),
                        failure_reason=(
                            "repository_file_fetch_failed"
                        ),
                    ),
                    processed_index=processed_index,
                )

                current_manifest.save(manifest_path)
                continue

            if raw_artifact.get("status") == "failed":
                current_manifest.record_artifact(
                    ManifestArtifact(
                        stable_id=f"failed:{repository}:{path}",
                        source_path_or_object_id=path,
                        artifact_type="other",
                        status="failed",
                        commit_sha=commit_sha,
                        source_sha=raw_artifact.get("sha"),
                        source_url=raw_artifact.get(
                            "source_url"
                        ),
                        failure_reason=raw_artifact.get(
                            "failure_reason",
                            "repository_file_fetch_failed",
                        ),
                    ),
                    processed_index=processed_index,
                )

                current_manifest.save(manifest_path)
                continue

            try:
                normalized = (
                    self.ingestion_control.normalize_file(
                        raw_artifact,
                        repository=repository,
                        ref=ref,
                        commit_sha=commit_sha,
                    )
                )

            except ValueError as exc:
                current_manifest.record_artifact(
                    ManifestArtifact(
                        stable_id=f"failed:{repository}:{path}",
                        source_path_or_object_id=path,
                        artifact_type="other",
                        status="failed",
                        commit_sha=commit_sha,
                        source_sha=raw_artifact.get("sha"),
                        source_url=raw_artifact.get(
                            "source_url"
                        ),
                        failure_reason=str(exc),
                    ),
                    processed_index=processed_index,
                )

                current_manifest.save(manifest_path)
                continue

            current_manifest.record_artifact(
                self._to_manifest_artifact(normalized),
                processed_index=processed_index,
            )

            current_manifest.save(manifest_path)

        current_manifest.mark_completed()
        current_manifest.save(manifest_path)

        previous_artifacts = (
            previous_manifest.artifacts
            if previous_manifest is not None
            else []
        )

        current_artifacts = [
            artifact
            for artifact in current_manifest.artifacts
            if artifact.status == "processed"
        ]

        return self._detect_changes(
            previous_artifacts,
            current_artifacts,
        )

    def _load_previous_manifest(self,manifest_path: str | Path) -> IngestionManifest | None:
        """Load the previous manifest when one exists."""

        path = Path(manifest_path)

        if not path.exists():
            return None

        return IngestionManifest.load(path)

    def _previous_processed_by_path(self,manifest: IngestionManifest | None) -> dict[str, ManifestArtifact]:
        """Return previous processed artifacts indexed by source path."""

        if manifest is None:
            return {}

        return {
            artifact.source_path_or_object_id: artifact
            for artifact in manifest.artifacts
            if artifact.status == "processed"
        }

    def _carry_forward_artifact(self,artifact: ManifestArtifact,commit_sha: str) -> ManifestArtifact:
        """Carry unchanged source state into the current revision."""

        carried = deepcopy(artifact)
        carried.commit_sha = commit_sha

        return carried

    def _detect_changes(self,previous: list[ManifestArtifact],current: list[ManifestArtifact]) -> list[ArtifactChange]:
        """Detect changes using source SHA while preserving revision provenance."""

        comparison_previous = [
            self._comparison_artifact(artifact)
            for artifact in previous
            if artifact.status == "processed"
        ]

        comparison_current = [
            self._comparison_artifact(artifact)
            for artifact in current
            if artifact.status == "processed"
        ]

        changes = self.change_detector.detect(
            previous=comparison_previous,
            current=comparison_current,
        )

        previous_by_id = {
            artifact.stable_id: artifact
            for artifact in previous
            if artifact.status == "processed"
        }

        current_by_id = {
            artifact.stable_id: artifact
            for artifact in current
            if artifact.status == "processed"
        }

        corrected_changes: list[ArtifactChange] = []

        for change in changes:
            previous_artifact = previous_by_id.get(
                change.stable_id
            )

            current_artifact = current_by_id.get(
                change.stable_id
            )

            if change.previous_stable_id is not None:
                previous_artifact = previous_by_id.get(
                    change.previous_stable_id
                )

            corrected_changes.append(
                change.model_copy(
                    update={
                        "previous_sha": (
                            previous_artifact.commit_sha
                            if previous_artifact is not None
                            else change.previous_sha
                        ),
                        "current_sha": (
                            current_artifact.commit_sha
                            if current_artifact is not None
                            else change.current_sha
                        ),
                    }
                )
            )

        return corrected_changes

    def _comparison_artifact(self,artifact: ManifestArtifact) -> ManifestArtifact:
        """Create a change-detection copy using source SHA as identity."""

        comparison = deepcopy(artifact)

        if comparison.source_sha is not None:
            comparison.commit_sha = comparison.source_sha

        return comparison

    def _to_manifest_artifact(self,artifact: RepositoryArtifact) -> ManifestArtifact:
        """Convert a normalized artifact to manifest state."""

        return ManifestArtifact(
            stable_id=artifact.stable_id,
            source_path_or_object_id=(
                artifact.source_path_or_object_id
            ),
            artifact_type=artifact.artifact_type,
            status="processed",
            commit_sha=artifact.commit_sha,
            source_sha=(
                artifact.source_sha
                if hasattr(artifact, "source_sha")
                else artifact.metadata.get("source_sha")
            ),
            source_url=artifact.source_url,
        )