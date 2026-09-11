"""Repository artifact change detection for Problem Statement 1."""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel
from src.ingestion_manifest import ManifestArtifact


class ChangeType(str, Enum):
    """Supported repository artifact change types."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    UNCHANGED = "unchanged"


class ArtifactChange(BaseModel):
    """Represents the detected change for one repository artifact."""

    stable_id: str
    change_type: ChangeType

    # These represent repository revision SHAs.
    previous_sha: str | None = None
    current_sha: str | None = None

    # This represents the actual repository object/file SHA.
    source_sha: str | None = None

    previous_stable_id: str | None = None

    previous_path: str | None = None
    current_path: str | None = None


class ChangeDetector:
    """Compare two artifact snapshots and detect repository changes."""

    @staticmethod
    def _source_sha(artifact: ManifestArtifact) -> str | None:
        """
        Return the artifact-level source SHA.

        source_sha is the preferred file/object identity.

        For older manifest records that do not contain source_sha,
        commit_sha is used as a compatibility fallback.
        """

        if artifact.source_sha is not None:
            return artifact.source_sha

        return artifact.commit_sha

    def detect(self,previous: list[ManifestArtifact],current: list[ManifestArtifact]) -> list[ArtifactChange]:
        """
        Detect added, modified, deleted, renamed, and unchanged artifacts.

        Repository commit SHA is used for provenance fields.
        Source SHA is used for artifact-level change comparison.
        """

        previous_by_id = {
            artifact.stable_id: artifact
            for artifact in previous
        }

        current_by_id = {
            artifact.stable_id: artifact
            for artifact in current
        }

        deleted = {
            stable_id: artifact
            for stable_id, artifact in previous_by_id.items()
            if stable_id not in current_by_id
        }

        added = {
            stable_id: artifact
            for stable_id, artifact in current_by_id.items()
            if stable_id not in previous_by_id
        }

        
        # Rename detection

        previous_by_source_sha: dict[
            str,
            list[tuple[str, ManifestArtifact]],
        ] = {}

        current_by_source_sha: dict[
            str,
            list[tuple[str, ManifestArtifact]],
        ] = {}

        for stable_id, artifact in deleted.items():

            source_sha = self._source_sha(
                artifact
            )

            if source_sha is None:
                continue

            previous_by_source_sha.setdefault(
                source_sha,
                [],
            ).append(
                (stable_id, artifact)
            )

        for stable_id, artifact in added.items():

            source_sha = self._source_sha(
                artifact
            )

            if source_sha is None:
                continue

            current_by_source_sha.setdefault(
                source_sha,
                [],
            ).append(
                (stable_id, artifact)
            )

        rename_matches: dict[
            str,
            tuple[str, ManifestArtifact],
        ] = {}

        for source_sha, previous_items in (
            previous_by_source_sha.items()):

            current_items = current_by_source_sha.get(
                source_sha,
                [],
            )

            # Only unique source-SHA pairs are treated as renames.
            if (
                len(previous_items) != 1
                or len(current_items) != 1):
                continue

            previous_stable_id, previous_artifact = (
                previous_items[0]
            )

            current_stable_id, current_artifact = (
                current_items[0]
            )

            if (
                previous_artifact.artifact_type
                != current_artifact.artifact_type):
                continue

            rename_matches[current_stable_id] = (
                previous_stable_id,
                previous_artifact,
            )

        # The previous stable IDs matched by rename must not also
        # appear as DELETED.
        renamed_previous_ids = {
            previous_stable_id
            for (
                previous_stable_id,
                _previous_artifact,
            ) in rename_matches.values()
        }

        
        # Final change classification
        

        changes: list[ArtifactChange] = []

        all_stable_ids = (
            set(previous_by_id)
            | set(current_by_id)
        )

        for stable_id in sorted(all_stable_ids):

            previous_artifact = previous_by_id.get(
                stable_id
            )

            current_artifact = current_by_id.get(
                stable_id
            )

            
            # RENAMED
            

            if stable_id in rename_matches:

                (
                    previous_stable_id,
                    previous_artifact,
                ) = rename_matches[stable_id]

                if current_artifact is None:
                    continue

                changes.append(
                    ArtifactChange(
                        stable_id=stable_id,
                        change_type=ChangeType.RENAMED,

                        # Provenance remains repository revision.
                        previous_sha=(
                            previous_artifact.commit_sha
                        ),
                        current_sha=(
                            current_artifact.commit_sha
                        ),

                        # Actual file/object identity.
                        source_sha=(
                            current_artifact.source_sha
                            or previous_artifact.source_sha
                        ),

                        previous_stable_id=(
                            previous_stable_id
                        ),

                        previous_path=(
                            previous_artifact
                            .source_path_or_object_id
                        ),

                        current_path=(
                            current_artifact
                            .source_path_or_object_id
                        ),
                    )
                )

                continue

            
            # Suppress old side of a rename.
            

            if stable_id in renamed_previous_ids:
                continue

            
            # ADDED
            

            if previous_artifact is None:

                if current_artifact is None:
                    continue

                changes.append(
                    ArtifactChange(
                        stable_id=stable_id,
                        change_type=ChangeType.ADDED,

                        # This must be repository commit SHA.
                        current_sha=(
                            current_artifact.commit_sha
                        ),

                        source_sha=(
                            current_artifact.source_sha
                        ),

                        current_path=(
                            current_artifact
                            .source_path_or_object_id
                        ),
                    )
                )

                continue

            
            # DELETED
            

            if current_artifact is None:

                changes.append(
                    ArtifactChange(
                        stable_id=stable_id,
                        change_type=ChangeType.DELETED,

                        # This must be previous repository
                        # commit SHA.
                        previous_sha=(
                            previous_artifact.commit_sha
                        ),

                        source_sha=(
                            previous_artifact.source_sha
                        ),

                        previous_path=(
                            previous_artifact
                            .source_path_or_object_id
                        ),
                    )
                )

                continue

            
            # MODIFIED
            

            previous_source_sha = self._source_sha(
                previous_artifact
            )

            current_source_sha = self._source_sha(
                current_artifact
            )

            if (
                previous_source_sha
                != current_source_sha):

                changes.append(
                    ArtifactChange(
                        stable_id=stable_id,
                        change_type=ChangeType.MODIFIED,

                        # Repository revision provenance.
                        previous_sha=(
                            previous_artifact.commit_sha
                        ),
                        current_sha=(
                            current_artifact.commit_sha
                        ),

                        # Actual changed file/object SHA.
                        source_sha=(
                            current_artifact.source_sha
                        ),

                        previous_path=(
                            previous_artifact
                            .source_path_or_object_id
                        ),

                        current_path=(
                            current_artifact
                            .source_path_or_object_id
                        ),
                    )
                )

                continue

            
            # UNCHANGED
            

            changes.append(
                ArtifactChange(
                    stable_id=stable_id,
                    change_type=ChangeType.UNCHANGED,

                    previous_sha=(
                        previous_artifact.commit_sha
                    ),
                    current_sha=(
                        current_artifact.commit_sha
                    ),

                    source_sha=(
                        current_artifact.source_sha
                    ),

                    previous_path=(
                        previous_artifact
                        .source_path_or_object_id
                    ),

                    current_path=(
                        current_artifact
                        .source_path_or_object_id
                    ),
                )
            )

        return changes