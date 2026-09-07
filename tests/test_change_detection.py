from src.change_detection import (
    ChangeDetector,
    ChangeType,
)
from src.ingestion_manifest import ManifestArtifact


def create_artifact(
    stable_id: str,
    path: str,
    commit_sha: str,
    source_sha: str | None = None,
) -> ManifestArtifact:
    return ManifestArtifact(
        stable_id=stable_id,
        source_path_or_object_id=path,
        artifact_type="code",
        status="processed",
        commit_sha=commit_sha,
        source_sha=source_sha,
    )


def test_detects_added_artifact() -> None:
    detector = ChangeDetector()

    current = [
        create_artifact(
            "artifact-1",
            "scrapy/http/request.py",
            "sha-1",
        )
    ]

    changes = detector.detect(
        previous=[],
        current=current,
    )

    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.ADDED
    assert changes[0].current_sha == "sha-1"


def test_detects_modified_artifact() -> None:
    detector = ChangeDetector()

    previous = [
        create_artifact(
            "artifact-1",
            "scrapy/http/request.py",
            "sha-old",
            "blob-old",
        )
    ]

    current = [
        create_artifact(
            "artifact-1",
            "scrapy/http/request.py",
            "sha-new",
            "blob-new",
        )
    ]

    changes = detector.detect(
        previous=previous,
        current=current,
    )

    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.MODIFIED
    assert changes[0].previous_sha == "sha-old"
    assert changes[0].current_sha == "sha-new"


def test_detects_deleted_artifact() -> None:
    detector = ChangeDetector()

    previous = [
        create_artifact(
            "artifact-1",
            "scrapy/http/request.py",
            "sha-1",
            "blob-1",
        )
    ]

    changes = detector.detect(
        previous=previous,
        current=[],
    )

    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.DELETED
    assert changes[0].previous_sha == "sha-1"
    assert changes[0].source_sha == "blob-1"


def test_detects_unchanged_artifact() -> None:
    detector = ChangeDetector()

    previous = [
        create_artifact(
            "artifact-1",
            "scrapy/http/request.py",
            "sha-1",
            "blob-1",
        )
    ]

    current = [
        create_artifact(
            "artifact-1",
            "scrapy/http/request.py",
            "sha-1",
            "blob-1",
        )
    ]

    changes = detector.detect(
        previous=previous,
        current=current,
    )

    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.UNCHANGED


def test_detects_renamed_artifact() -> None:
    detector = ChangeDetector()

    previous = [
        create_artifact(
            "old-id",
            "scrapy/http/old_request.py",
            "commit-old",
            "blob-123",
        )
    ]

    current = [
        create_artifact(
            "new-id",
            "scrapy/http/new_request.py",
            "commit-new",
            "blob-123",
        )
    ]

    changes = detector.detect(
        previous=previous,
        current=current,
    )

    assert len(changes) == 1

    change = changes[0]

    assert change.change_type == ChangeType.RENAMED
    assert change.stable_id == "new-id"
    assert change.previous_stable_id == "old-id"
    assert change.previous_path == (
        "scrapy/http/old_request.py"
    )
    assert change.current_path == (
        "scrapy/http/new_request.py"
    )
    assert change.source_sha == "blob-123"
    assert change.previous_sha == "commit-old"
    assert change.current_sha == "commit-new"


def test_duplicate_source_sha_does_not_create_ambiguous_rename() -> None:
    detector = ChangeDetector()

    previous = [
        create_artifact(
            "old-id-1",
            "src/old_one.py",
            "commit-old",
            "blob-123",
        ),
        create_artifact(
            "old-id-2",
            "src/old_two.py",
            "commit-old",
            "blob-123",
        ),
    ]

    current = [
        create_artifact(
            "new-id",
            "src/new_name.py",
            "commit-new",
            "blob-123",
        )
    ]

    changes = detector.detect(
        previous=previous,
        current=current,
    )

    assert not any(
        change.change_type == ChangeType.RENAMED
        for change in changes
    )


def test_modified_artifact_is_not_detected_as_rename() -> None:
    detector = ChangeDetector()

    previous = [
        create_artifact(
            "artifact-id",
            "src/example.py",
            "commit-old",
            "blob-old",
        )
    ]

    current = [
        create_artifact(
            "artifact-id",
            "src/example.py",
            "commit-new",
            "blob-new",
        )
    ]

    changes = detector.detect(
        previous=previous,
        current=current,
    )

    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.MODIFIED


def test_detects_multiple_change_types() -> None:
    detector = ChangeDetector()

    previous = [
        create_artifact(
            "artifact-unchanged",
            "unchanged.py",
            "sha-1",
            "blob-1",
        ),
        create_artifact(
            "artifact-modified",
            "modified.py",
            "sha-old",
            "blob-old",
        ),
        create_artifact(
            "artifact-deleted",
            "deleted.py",
            "sha-deleted",
            "blob-deleted",
        ),
    ]

    current = [
        create_artifact(
            "artifact-unchanged",
            "unchanged.py",
            "sha-1",
            "blob-1",
        ),
        create_artifact(
            "artifact-modified",
            "modified.py",
            "sha-new",
            "blob-new",
        ),
        create_artifact(
            "artifact-added",
            "added.py",
            "sha-added",
            "blob-added",
        ),
    ]

    changes = detector.detect(
        previous=previous,
        current=current,
    )

    change_map = {
        change.stable_id: change.change_type
        for change in changes
    }

    assert change_map == {
        "artifact-added": ChangeType.ADDED,
        "artifact-deleted": ChangeType.DELETED,
        "artifact-modified": ChangeType.MODIFIED,
        "artifact-unchanged": ChangeType.UNCHANGED,
    }