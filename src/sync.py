"""Incremental repository synchronization for Problem Statement 1."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.change_detection import ArtifactChange, ChangeDetector
from src.ingestion_control import (IngestionControl,RepositoryArtifact)
from src.ingestion_manifest import (IngestionManifest,ManifestArtifact)
from src.repository_connector import RepositoryConnector


class RepositorySync:
    """
    Orchestrate repository snapshot acquisition and incremental
    synchronization.
    """

    def __init__(self,connector: RepositoryConnector,ingestion_control: IngestionControl,change_detector: ChangeDetector | None = None) -> None:
        self.connector = connector
        self.ingestion_control = ingestion_control
        self.change_detector = (
            change_detector
            if change_detector is not None
            else ChangeDetector()
        )

    def synchronize(self,manifest_path: str | Path) -> list[ArtifactChange]:
        """
        Synchronize the repository against the previous manifest.

        The repository tree determines which files are affected.
        Only new or changed files are fetched and normalized.
        """

        
        # S1 / S2 — Validate repository reference and resolve
        # current repository revision.
        

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
        

        previous_manifest = self._load_previous_manifest(
            manifest_path
        )

        previous_artifacts = (
            previous_manifest.artifacts
            if previous_manifest is not None
            else []
        )

        previous_by_path = {
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

        affected_paths: list[str] = []

        
        # S4 — Determine affected artifacts.
        

        for entry in tree:

            if entry.get("type") != "blob":
                continue

            path = str(
                entry.get("path", "")
            )

            if not path:
                continue

            tree_source_sha = entry.get("sha")

            
            # IMPORTANT:
            
            # IngestionControl.evaluate_file() requires content
            # to be a string.
            
            # At tree stage we intentionally do not have the file
            # content yet, so an empty string is used ONLY for
            # eligibility/secret filtering.
            
            # Actual content is fetched only when the artifact
            # belongs to affected_paths.
            

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

            
            # Excluded artifact.
            

            if not decision.accepted:

                current_manifest.add_artifact(
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
                    )
                )

                continue

            previous_artifact = previous_by_path.get(
                path
            )

            
            # UNCHANGED
            
            # Same path + same repository object SHA.
            
            # Do not download content again.
            

            if (
                previous_artifact is not None
                and previous_artifact.source_sha
                == tree_source_sha):

                current_manifest.add_artifact(
                    ManifestArtifact(
                        stable_id=(
                            previous_artifact.stable_id
                        ),
                        source_path_or_object_id=path,
                        artifact_type=(
                            previous_artifact.artifact_type
                        ),
                        status="processed",
                        commit_sha=commit_sha,
                        source_sha=tree_source_sha,
                        source_url=(
                            previous_artifact.source_url
                        ),
                    )
                )

                continue

            
            # ADDED or MODIFIED.
            
            # The content must be fetched because the file is new
            # or its repository object SHA changed.
            

            affected_paths.append(path)

        
        # Fetch only affected paths.
        
        # Never call fetch_files() with an empty list.
        

        if affected_paths:

            fetched_files = self.connector.fetch_files(
                reference,
                affected_paths,
                commit_sha,
            )

        else:

            fetched_files = []

        fetched_by_path = {
            str(item.get("path")): item
            for item in fetched_files
        }

        
        # S5 — Reprocess affected artifacts.
       

        for path in affected_paths:

            raw_artifact = fetched_by_path.get(
                path
            )

            
            # Missing fetch result.
            

            if raw_artifact is None:

                current_manifest.add_artifact(
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
                    )
                )

                continue

            
            # Explicit connector failure.
            

            if raw_artifact.get("status") == "failed":

                current_manifest.add_artifact(
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
                    )
                )

                continue

            
            # Normalize actual fetched content.
            

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

                current_manifest.add_artifact(
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
                    )
                )

                continue

            
            # The fetched repository object's SHA is authoritative.
            

            current_manifest.add_artifact(
                self._to_manifest_artifact(
                    normalized,
                    source_sha=raw_artifact.get(
                        "sha"
                    ),
                )
            )

        
        # Finalize manifest.
        

        current_manifest.mark_completed()

        current_manifest.save(
            manifest_path
        )

        
        # Only successfully processed artifacts participate in
        # change detection.
        

        current_artifacts = [
            artifact
            for artifact in current_manifest.artifacts
            if artifact.status == "processed"
        ]

        
        # Final change classification.
        

        return self.change_detector.detect(
            previous=previous_artifacts,
            current=current_artifacts,
        )

    def _load_previous_manifest(self,manifest_path: str | Path) -> IngestionManifest | None:
        """Load the previous manifest when it exists."""

        path = Path(manifest_path)

        if not path.exists():
            return None

        return IngestionManifest.load(
            path
        )

    def _to_manifest_artifact(self,artifact: RepositoryArtifact,source_sha: str | None) -> ManifestArtifact:
        """
        Convert a normalized repository artifact to manifest state.

        source_sha comes directly from the repository tree/fetch
        response and is therefore authoritative.
        """

        return ManifestArtifact(
            stable_id=artifact.stable_id,
            source_path_or_object_id=(
                artifact.source_path_or_object_id),
            artifact_type=artifact.artifact_type,
            status="processed",
            commit_sha=artifact.commit_sha,
            source_sha=source_sha,
            source_url=artifact.source_url,
        )