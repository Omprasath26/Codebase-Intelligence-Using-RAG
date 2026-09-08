from pathlib import Path

from src.ingestion_manifest import (
    IngestionManifest,
    ManifestArtifact,
)


def create_manifest() -> IngestionManifest:
    return IngestionManifest(
        run_id="run-001",
        repository="scrapy/scrapy",
        ref="master",
        commit_sha="abc123",
    )


def test_adds_artifact_to_manifest() -> None:
    manifest = create_manifest()

    manifest.add_artifact(
        ManifestArtifact(
            stable_id="chunk-001",
            source_path_or_object_id="scrapy/http/request.py",
            artifact_type="code",
            status="processed",
            commit_sha="abc123",
            source_url=(
                "https://github.com/scrapy/scrapy/"
                "blob/master/scrapy/http/request.py"
            ),
        )
    )

    assert len(manifest.artifacts) == 1
    assert manifest.artifacts[0].stable_id == "chunk-001"
    assert manifest.artifacts[0].status == "processed"


def test_records_exclusion_reason() -> None:
    manifest = create_manifest()

    manifest.add_artifact(
        ManifestArtifact(
            stable_id="excluded-001",
            source_path_or_object_id="build/output.py",
            artifact_type="other",
            status="excluded",
            exclusion_reason="excluded_directory",
        )
    )

    assert manifest.artifacts[0].status == "excluded"
    assert (
        manifest.artifacts[0].exclusion_reason
        == "excluded_directory"
    )


def test_records_failure_reason() -> None:
    manifest = create_manifest()

    manifest.add_artifact(
        ManifestArtifact(
            stable_id="failed-001",
            source_path_or_object_id="scrapy/http/request.py",
            artifact_type="code",
            status="failed",
            failure_reason="network timeout",
        )
    )

    assert manifest.artifacts[0].status == "failed"
    assert (
        manifest.artifacts[0].failure_reason
        == "network timeout"
    )


def test_checkpoint_is_resumable() -> None:
    manifest = create_manifest()

    manifest.update_checkpoint(
        last_processed_index=12
    )

    assert manifest.checkpoint is not None
    assert (
        manifest.checkpoint.last_processed_index
        == 12
    )
    assert manifest.checkpoint.status == "in_progress"


def test_record_artifact_advances_checkpoint() -> None:
    manifest = create_manifest()

    artifact = ManifestArtifact(
        stable_id="artifact-001",
        source_path_or_object_id="scrapy/http/request.py",
        artifact_type="code",
        status="processed",
        commit_sha="abc123",
        source_sha="blob-123",
    )

    manifest.record_artifact(
        artifact,
        processed_index=0,
    )

    assert len(manifest.artifacts) == 1
    assert manifest.artifacts[0].stable_id == "artifact-001"
    assert manifest.artifacts[0].commit_sha == "abc123"
    assert manifest.artifacts[0].source_sha == "blob-123"

    assert manifest.checkpoint is not None
    assert manifest.checkpoint.last_processed_index == 0
    assert manifest.checkpoint.status == "in_progress"


def test_resume_from_index_without_checkpoint() -> None:
    manifest = create_manifest()

    assert manifest.resume_from_index() == 0


def test_resume_from_index_returns_next_unprocessed_index() -> None:
    manifest = create_manifest()

    manifest.update_checkpoint(
        last_processed_index=5
    )

    assert manifest.resume_from_index() == 6


def test_checkpoint_can_be_completed() -> None:
    manifest = create_manifest()

    manifest.add_artifact(
        ManifestArtifact(
            stable_id="chunk-001",
            source_path_or_object_id="request.py",
            artifact_type="code",
            status="processed",
        )
    )

    manifest.mark_completed()

    assert manifest.status == "completed"
    assert manifest.checkpoint is not None
    assert manifest.checkpoint.status == "completed"


def test_failed_run_preserves_failure_state() -> None:
    manifest = create_manifest()

    manifest.update_checkpoint(
        last_processed_index=5
    )

    manifest.mark_failed(
        "GitHub API request failed"
    )

    assert manifest.status == "failed"
    assert manifest.checkpoint is not None
    assert manifest.checkpoint.status == "failed"
    assert (
        manifest.checkpoint.last_processed_index
        == 5
    )

    failure_records = [
        artifact
        for artifact in manifest.artifacts
        if artifact.status == "failed"
    ]

    assert len(failure_records) == 1
    assert (
        failure_records[0].failure_reason
        == "GitHub API request failed"
    )


def test_manifest_artifact_persists_source_sha() -> None:
    artifact = ManifestArtifact(
        stable_id="artifact-1",
        source_path_or_object_id="src/example.py",
        artifact_type="code",
        status="processed",
        commit_sha="commit-123",
        source_sha="blob-456",
    )

    assert artifact.source_sha == "blob-456"


def test_manifest_can_be_saved_and_loaded(tmp_path: Path) -> None:
    manifest = create_manifest()

    manifest.add_artifact(
        ManifestArtifact(
            stable_id="chunk-001",
            source_path_or_object_id="request.py",
            artifact_type="code",
            status="processed",
            commit_sha="abc123",
            source_sha="blob-456",
        )
    )

    manifest.update_checkpoint(
        last_processed_index=0
    )

    path = tmp_path / "manifest.json"

    manifest.save(path)

    loaded = IngestionManifest.load(path)

    assert loaded.run_id == "run-001"
    assert loaded.repository == "scrapy/scrapy"
    assert loaded.commit_sha == "abc123"
    assert len(loaded.artifacts) == 1
    assert loaded.artifacts[0].stable_id == "chunk-001"
    assert loaded.artifacts[0].source_sha == "blob-456"
    assert loaded.checkpoint is not None
    assert loaded.checkpoint.last_processed_index == 0