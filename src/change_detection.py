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
    previous_sha: str | None = None
    current_sha: str | None = None
    source_sha: str | None = None
    previous_stable_id: str | None = None
    previous_path: str | None = None
    current_path: str | None = None


class ChangeDetector:
    """Compare two artifact snapshots and detect repository changes."""

    def detect(
        self,
        previous: list[ManifestArtifact],
        current: list[ManifestArtifact],
    ) -> list[ArtifactChange]:
        """Detect added, modified, renamed, deleted, and unchanged artifacts."""

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

        previous_by_source_sha: dict[
            str,
            list[tuple[str, ManifestArtifact]],
        ] = {}

        current_by_source_sha: dict[
            str,
            list[tuple[str, ManifestArtifact]],
        ] = {}

        for stable_id, artifact in deleted.items():
            if artifact.source_sha is None:
                continue

            previous_by_source_sha.setdefault(
                artifact.source_sha,
                [],
            ).append(
                (stable_id, artifact)
            )

        for stable_id, artifact in added.items():
            if artifact.source_sha is None:
                continue

            current_by_source_sha.setdefault(
                artifact.source_sha,
                [],
            ).append(
                (stable_id, artifact)
            )

        rename_matches: dict[
            str,
            tuple[str, ManifestArtifact],
        ] = {}

        for source_sha, previous_items in (
            previous_by_source_sha.items()
        ):
            current_items = current_by_source_sha.get(
                source_sha,
                [],
            )

            # Rename detection is only safe when the
            # source SHA maps one-to-one.
            if (
                len(previous_items) != 1
                or len(current_items) != 1
            ):
                continue

            previous_stable_id, previous_artifact = (
                previous_items[0]
            )

            current_stable_id, current_artifact = (
                current_items[0]
            )

            if (
                previous_artifact.artifact_type
                != current_artifact.artifact_type
            ):
                continue

            rename_matches[current_stable_id] = (
                previous_stable_id,
                previous_artifact,
            )

        # Stable IDs that participated in a rename must not
        # also be reported as added or deleted.
        renamed_previous_ids = {
            previous_stable_id
            for previous_stable_id, _ in (
                rename_matches.values()
            )
        }

        renamed_current_ids = set(rename_matches)

        changes: list[ArtifactChange] = []

        for stable_id in sorted(
            set(previous_by_id) | set(current_by_id)
        ):
            previous_artifact = previous_by_id.get(
                stable_id
            )

            current_artifact = current_by_id.get(
                stable_id
            )

            # A new stable ID with the same unique source SHA
            # represents a rename.
            if stable_id in rename_matches:
                (
                    previous_stable_id,
                    previous_artifact,
                ) = rename_matches[stable_id]

                changes.append(
                    ArtifactChange(
                        stable_id=stable_id,
                        change_type=ChangeType.RENAMED,
                        previous_sha=(
                            previous_artifact.commit_sha
                        ),
                        current_sha=(
                            current_artifact.commit_sha
                            if current_artifact
                            else None
                        ),
                        source_sha=(
                            current_artifact.source_sha
                            if current_artifact
                            else None
                        ),
                        previous_stable_id=(
                            previous_stable_id
                        ),
                        previous_path=(
                            previous_artifact.source_path_or_object_id
                        ),
                        current_path=(
                            current_artifact.source_path_or_object_id
                            if current_artifact
                            else None
                        ),
                    )
                )

                continue

            # The previous artifact was consumed by a rename.
            if stable_id in renamed_previous_ids:
                continue

            # The current artifact was consumed by a rename.
            if stable_id in renamed_current_ids:
                continue

            if previous_artifact is None:
                changes.append(
                    ArtifactChange(
                        stable_id=stable_id,
                        change_type=ChangeType.ADDED,
                        current_sha=(
                            current_artifact.commit_sha
                            if current_artifact
                            else None
                        ),
                        source_sha=(
                            current_artifact.source_sha
                            if current_artifact
                            else None
                        ),
                        current_path=(
                            current_artifact.source_path_or_object_id
                            if current_artifact
                            else None
                        ),
                    )
                )

                continue

            if current_artifact is None:
                changes.append(
                    ArtifactChange(
                        stable_id=stable_id,
                        change_type=ChangeType.DELETED,
                        previous_sha=(
                            previous_artifact.commit_sha
                        ),
                        source_sha=(
                            previous_artifact.source_sha
                        ),
                        previous_path=(
                            previous_artifact.source_path_or_object_id
                        ),
                    )
                )

                continue

            if (
                previous_artifact.commit_sha
                != current_artifact.commit_sha
            ):
                changes.append(
                    ArtifactChange(
                        stable_id=stable_id,
                        change_type=ChangeType.MODIFIED,
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
                            previous_artifact.source_path_or_object_id
                        ),
                        current_path=(
                            current_artifact.source_path_or_object_id
                        ),
                    )
                )

                continue

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
                        previous_artifact.source_path_or_object_id
                    ),
                    current_path=(
                        current_artifact.source_path_or_object_id
                    ),
                )
            )

        return changes