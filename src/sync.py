"""Incremental repository synchronization for Problem Statement 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from src.change_detection import ArtifactChange, ChangeDetector
from src.ingestion_control import (IngestionControl,RepositoryArtifact)
from src.ingestion_manifest import (IngestionManifest,ManifestArtifact)
from src.repository_connector import RepositoryConnector


class RepositorySync:
    """Orchestrate repository snapshot acquisition and incremental changes."""

    def __init__(self,connector: RepositoryConnector,ingestion_control: IngestionControl,change_detector: ChangeDetector | None = None) -> None:
        self.connector = connector
        self.ingestion_control = ingestion_control
        self.change_detector = (
            change_detector
            if change_detector is not None
            else ChangeDetector()
        )

    def synchronize(self,manifest_path: str | Path) -> list[ArtifactChange]:
        """Synchronize the repository against the previous manifest.

        The current repository tree is used to determine the affected set.

        Only added or modified files are fetched. Unchanged files are carried
        forward from the previous manifest, while deleted files are omitted
        from the current processed state.

        This keeps the manifest as the current repository state while
        preserving enough previous provenance for change detection.
        """

        
        # S1 / S2 — Validate repository reference and resolve revision.
        

        reference = self.connector.validate_reference()

        repository_metadata = (
            self.connector.fetch_repository_metadata(
                reference
            )
        )

        commit_sha = repository_metadata["commit_sha"]

        repository = (
            f"{repository_metadata['owner']}/"
            f"{repository_metadata['name']}"
        )

        ref = repository_metadata["ref"]

        
        # Load previous manifest.
        

        previous_manifest = self._load_previous_manifest(manifest_path)

        previous_artifacts = (
            previous_manifest.artifacts
            if previous_manifest is not None
            else []
        )

        previous_processed_by_path = {
            artifact.source_path_or_object_id: artifact
            for artifact in previous_artifacts
            if artifact.status == "processed"
        }

        
        # Create current manifest.
        
        current_manifest = IngestionManifest(
            run_id=str(uuid4()),
            repository=repository,
            ref=ref,
            commit_sha=commit_sha,
            status="in_progress",
        )

        
        # S3 — Fetch current repository tree.
        

        tree = self.connector.fetch_tree(
            reference,
            commit_sha,
        )

        
        # Current accepted tree entries.
        
        # We keep the tree representation separate from the fetched file
        # representation because the tree SHA is sufficient to determine
        # whether the artifact needs to be fetched.
        

        current_tree_by_path: dict[str, dict[str, Any]] = {}

        for entry in tree:
            if entry.get("type") != "blob":
                continue

            path = str(
                entry.get("path", "")
            )

            if not path:
                continue

            tree_source_sha = entry.get("sha")

            # IngestionControl.evaluate_file() expects content to be a
            # string. At tree stage we intentionally use an empty string
            # because actual content is fetched only for affected files.
            raw_tree_artifact = {
                "path": path,
                "size": entry.get("size"),
                "sha": tree_source_sha,
                "content": "",
            }

            decision = (
                self.ingestion_control.evaluate_file(
                    raw_tree_artifact
                )
            )

            if not decision.accepted:
                current_manifest.record_artifact(
                    ManifestArtifact(
                        stable_id=(
                            f"excluded:{repository}:{path}"
                        ),
                        source_path_or_object_id=path,
                        artifact_type="other",
                        status="excluded",
                        commit_sha=commit_sha,
                        source_sha=tree_source_sha,
                        exclusion_reason=decision.reason,
                    ),
                    processed_index=len(
                        current_manifest.artifacts
                    ),
                )
                continue

            current_tree_by_path[path] = {
                "path": path,
                "size": entry.get("size"),
                "sha": tree_source_sha,
            }

        
        # S4 — Determine affected paths.
        
        # Added:
        #   path does not exist in previous processed state.
        
        # Modified:
        #   same path exists but source SHA changed.
        
        # Unchanged:
        #   same path and same source SHA. The previous manifest artifact
        #   is carried forward without fetching content.
        
        # Deleted:
        #   previous processed path no longer exists in current tree.
        #   It is intentionally not added to current processed state.
        
        # Renamed:
        #   handled by ChangeDetector using source SHA/provenance after
        #   the current manifest has been constructed.
        

        affected_paths: list[str] = []

        for path in sorted(current_tree_by_path):
            tree_entry = current_tree_by_path[path]

            current_source_sha = tree_entry.get(
                "sha"
            )

            previous_artifact = (
                previous_processed_by_path.get(path)
            )

            if previous_artifact is None:
                affected_paths.append(path)
                continue

            previous_source_sha = (
                previous_artifact.source_sha
            )

            if previous_source_sha != current_source_sha:
                affected_paths.append(path)
                continue

            
            # Unchanged artifact.
            
            # Preserve the existing artifact identity and provenance,
            # but update its repository revision to the current revision.
            # This is essential: otherwise the previous artifact disappears
            # from the current manifest and ChangeDetector reports it as
            # deleted.
           

            current_manifest.record_artifact(
                self._carry_forward_artifact(
                    previous_artifact=previous_artifact,
                    commit_sha=commit_sha,
                    source_sha=current_source_sha,
                ),
                processed_index=len(
                    current_manifest.artifacts
                ),
            )

        
        # S5 — Fetch only affected files.
        
        # IMPORTANT:
        # Do not call fetch_files() at all when affected_paths is empty.
        # This avoids unnecessary repository calls for deleted-only,
        # excluded-only, and unchanged-only synchronizations.
      

        fetched_files: list[dict[str, Any]] = []

        if affected_paths:
            fetched_files = self.connector.fetch_files(
                reference,
                affected_paths,
                commit_sha,
            )

        fetched_by_path = {
            str(item.get("path")): item
            for item in fetched_files
        }

        
        # S5 / normalization — Process affected files.
        

        for path in affected_paths:
            raw_artifact = fetched_by_path.get(
                path
            )

            if raw_artifact is None:
                current_manifest.record_artifact(
                    ManifestArtifact(
                        stable_id=(
                            f"failed:{repository}:{path}"
                        ),
                        source_path_or_object_id=path,
                        artifact_type="other",
                        status="failed",
                        commit_sha=commit_sha,
                        failure_reason=(
                            "repository_file_fetch_failed"
                        ),
                    ),
                    processed_index=len(
                        current_manifest.artifacts
                    ),
                )
                continue

            if raw_artifact.get("status") == "failed":
                current_manifest.record_artifact(
                    ManifestArtifact(
                        stable_id=(
                            f"failed:{repository}:{path}"
                        ),
                        source_path_or_object_id=path,
                        artifact_type="other",
                        status="failed",
                        commit_sha=commit_sha,
                        source_sha=raw_artifact.get(
                            "sha"
                        ),
                        source_url=raw_artifact.get(
                            "source_url"
                        ),
                        failure_reason=raw_artifact.get(
                            "failure_reason",
                            "repository_file_fetch_failed",
                        ),
                    ),
                    processed_index=len(
                        current_manifest.artifacts
                    ),
                )
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
                        stable_id=(
                            f"failed:{repository}:{path}"
                        ),
                        source_path_or_object_id=path,
                        artifact_type="other",
                        status="failed",
                        commit_sha=commit_sha,
                        source_sha=raw_artifact.get(
                            "sha"
                        ),
                        source_url=raw_artifact.get(
                            "source_url"
                        ),
                        failure_reason=str(exc),
                    ),
                    processed_index=len(
                        current_manifest.artifacts
                    ),
                )
                continue

            current_manifest.record_artifact(
                self._to_manifest_artifact(
                    normalized
                ),
                processed_index=len(
                    current_manifest.artifacts
                ),
            )

        
        # Mark the current manifest complete and persist it.
        

        current_manifest.mark_completed()

        current_manifest.save(
            manifest_path
        )

       
        # S8 — Change detection.
        
        # The detector receives:
        #   previous = previous processed state
        #   current  = current processed state
        
        # Because unchanged artifacts were carried forward above:
        #   unchanged -> UNCHANGED
        #   modified  -> MODIFIED
        #   added     -> ADDED
        #   deleted   -> DELETED
        #   rename    -> RENAMED
        
        # Deleted artifacts are intentionally absent from current state.
        

        current_processed_artifacts = [
            artifact
            for artifact in current_manifest.artifacts
            if artifact.status == "processed"
        ]

        return self.change_detector.detect(
            previous=previous_artifacts,
            current=current_processed_artifacts,
        )

    def _carry_forward_artifact(self,previous_artifact: ManifestArtifact,commit_sha: str,source_sha: str | None) -> ManifestArtifact:
        """Carry an unchanged artifact into the current manifest.

        The stable ID remains unchanged because the artifact itself has not
        changed. Repository revision and source SHA are refreshed from the
        current tree so the manifest accurately describes the current
        snapshot.
        """

        return ManifestArtifact(
            stable_id=previous_artifact.stable_id,
            source_path_or_object_id=(
                previous_artifact.source_path_or_object_id
            ),
            artifact_type=previous_artifact.artifact_type,
            status="processed",
            commit_sha=commit_sha,
            source_sha=source_sha,
            source_url=previous_artifact.source_url,
            exclusion_reason=None,
            failure_reason=None,
        )

    def _load_previous_manifest(self,manifest_path: str | Path) -> IngestionManifest | None:
        """Load the previous manifest when one exists."""

        path = Path(
            manifest_path
        )

        if not path.exists():
            return None

        return IngestionManifest.load(
            path
        )

    def _to_manifest_artifact(self,artifact: RepositoryArtifact,) -> ManifestArtifact:
        """Convert a normalized repository artifact to manifest state."""

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
                if hasattr(
                    artifact,
                    "source_sha",
                )
                else artifact.metadata.get(
                    "source_sha"
                )
            ),
            source_url=artifact.source_url,
        )