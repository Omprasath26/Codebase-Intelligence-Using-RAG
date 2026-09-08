"""Tests for incremental repository synchronization."""

from pathlib import Path
from unittest.mock import Mock

from src import sync
from src.change_detection import ChangeType
from src.ingestion_control import IngestionControl
from src.repository_connector import RepositoryReference
from src.sync import RepositorySync


def create_sync() -> RepositorySync:
    """Create a RepositorySync instance with mocked dependencies."""

    connector = Mock()
    ingestion_control = Mock(spec=IngestionControl)

    sync = RepositorySync(
        connector=connector,
        ingestion_control=ingestion_control,
    )

    return sync


def configure_connector(sync: RepositorySync) -> None:
    """Configure mocked repository acquisition."""

    sync.connector.validate_reference = Mock(
        return_value=RepositoryReference(
            owner="scrapy",
            name="scrapy",
            ref="master",
        )
    )

    sync.connector.fetch_repository_metadata = Mock(
        return_value={
            "owner": "scrapy",
            "name": "scrapy",
            "ref": "master",
            "commit_sha": "commit-old",
        }
    )

    sync.connector.fetch_tree = Mock(
        return_value=[]
    )

    sync.connector.fetch_files = Mock(
        return_value=[]
    )


def test_initial_sync_marks_files_as_added(tmp_path: Path) -> None:
    """Initial synchronization should report repository files as added."""

    sync = create_sync()
    configure_connector(sync)

    sync.connector.fetch_tree.return_value = [
        {
            "path": "scrapy/http/request.py",
            "type": "blob",
            "size": 100,
            "sha": "blob-1",
        }
    ]

    sync.connector.fetch_files.return_value = [
        {
            "path": "scrapy/http/request.py",
            "sha": "blob-1",
            "size": 100,
            "content": (
                "class Request:\n"
                "    pass\n"
            ),
            "source_url": (
                "https://example.com/request.py"
            ),
        }
    ]

    sync.ingestion_control.evaluate_file.return_value = Mock(
        accepted=True,
        reason="accepted",
    )

    sync.ingestion_control.normalize_file.return_value = Mock(
        stable_id="artifact-1",
        repository="scrapy/scrapy",
        artifact_type="code",
        source_path_or_object_id=(
            "scrapy/http/request.py"
        ),
        source_url=(
            "https://example.com/request.py"
        ),
        source_sha="blob-1",
        commit_sha="commit-old",
    )

    manifest_path = tmp_path / "manifest.json"

    changes = sync.synchronize(manifest_path)

    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.ADDED
    assert changes[0].current_sha == "commit-old"

    sync.connector.fetch_files.assert_called_once_with(
        sync.connector.validate_reference.return_value,
        ["scrapy/http/request.py"],
        "commit-old",
    )


def test_unchanged_file_is_not_fetched_again(tmp_path: Path) -> None:
    """Unchanged files should be carried forward without refetching."""

    sync = create_sync()
    configure_connector(sync)

    sync.connector.fetch_tree.return_value = [
        {
            "path": "scrapy/http/request.py",
            "type": "blob",
            "size": 100,
            "sha": "blob-1",
        }
    ]

    sync.connector.fetch_files.return_value = [
        {
            "path": "scrapy/http/request.py",
            "sha": "blob-1",
            "size": 100,
            "content": (
                "class Request:\n"
                "    pass\n"
            ),
            "source_url": (
                "https://example.com/request.py"
            ),
        }
    ]

    sync.ingestion_control.evaluate_file.return_value = Mock(
        accepted=True,
        reason="accepted",
    )

    sync.ingestion_control.normalize_file.return_value = Mock(
        stable_id="artifact-1",
        repository="scrapy/scrapy",
        artifact_type="code",
        source_path_or_object_id=(
            "scrapy/http/request.py"
        ),
        source_url=(
            "https://example.com/request.py"
        ),
        source_sha="blob-1",
        commit_sha="commit-old",
    )

    manifest_path = tmp_path / "manifest.json"

    first_changes = sync.synchronize(manifest_path)

    assert len(first_changes) == 1
    assert first_changes[0].change_type == ChangeType.ADDED

    sync.connector.fetch_repository_metadata.return_value[
        "commit_sha"
    ] = "commit-new"

    sync.connector.fetch_files.reset_mock()
    sync.ingestion_control.normalize_file.reset_mock()

    second_changes = sync.synchronize(manifest_path)

    assert len(second_changes) == 1

    assert (
        second_changes[0].change_type
        == ChangeType.UNCHANGED
    )

    assert second_changes[0].previous_sha == "commit-old"
    assert second_changes[0].current_sha == "commit-new"
    assert second_changes[0].source_sha == "blob-1"

    sync.connector.fetch_files.assert_called_once_with(
    sync.connector.validate_reference.return_value,
    [],
    "commit-new",
    )

    sync.ingestion_control.normalize_file.assert_not_called()


def test_modified_file_is_fetched_again(tmp_path: Path) -> None:
    """Modified files should be fetched and reported as modified."""

    sync = create_sync()
    configure_connector(sync)

    sync.connector.fetch_tree.return_value = [
        {
            "path": "scrapy/http/request.py",
            "type": "blob",
            "size": 100,
            "sha": "blob-old",
        }
    ]

    sync.connector.fetch_files.return_value = [
        {
            "path": "scrapy/http/request.py",
            "sha": "blob-old",
            "size": 100,
            "content": (
                "class Request:\n"
                "    pass\n"
            ),
            "source_url": (
                "https://example.com/request.py"
            ),
        }
    ]

    sync.ingestion_control.evaluate_file.return_value = Mock(
        accepted=True,
        reason="accepted",
    )

    sync.ingestion_control.normalize_file.return_value = Mock(
        stable_id="artifact-1",
        repository="scrapy/scrapy",
        artifact_type="code",
        source_path_or_object_id=(
            "scrapy/http/request.py"
        ),
        source_url=(
            "https://example.com/request.py"
        ),
        source_sha="blob-old",
        commit_sha="commit-old",
    )

    manifest_path = tmp_path / "manifest.json"

    sync.synchronize(manifest_path)

    sync.connector.fetch_repository_metadata.return_value[
        "commit_sha"
    ] = "commit-new"

    sync.connector.fetch_tree.return_value = [
        {
            "path": "scrapy/http/request.py",
            "type": "blob",
            "size": 120,
            "sha": "blob-new",
        }
    ]

    sync.connector.fetch_files.return_value = [
        {
            "path": "scrapy/http/request.py",
            "sha": "blob-new",
            "size": 120,
            "content": (
                "class Request:\n"
                "    def __init__(self):\n"
                "        pass\n"
            ),
            "source_url": (
                "https://example.com/request.py"
            ),
        }
    ]

    sync.ingestion_control.normalize_file.return_value = Mock(
        stable_id="artifact-1",
        repository="scrapy/scrapy",
        artifact_type="code",
        source_path_or_object_id=(
            "scrapy/http/request.py"
        ),
        source_url=(
            "https://example.com/request.py"
        ),
        source_sha="blob-new",
        commit_sha="commit-new",
    )

    sync.connector.fetch_files.reset_mock()

    changes = sync.synchronize(manifest_path)

    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.MODIFIED
    assert changes[0].previous_sha == "commit-old"
    assert changes[0].current_sha == "commit-new"
    assert changes[0].source_sha == "blob-new"

    sync.connector.fetch_files.assert_called_once_with(
        sync.connector.validate_reference.return_value,
        ["scrapy/http/request.py"],
        "commit-new",
    )


def test_deleted_file_is_detected(tmp_path: Path) -> None:
    """A file absent from the new tree should be reported as deleted."""

    sync = create_sync()
    configure_connector(sync)

    sync.connector.fetch_tree.return_value = [
        {
            "path": "scrapy/http/request.py",
            "type": "blob",
            "size": 100,
            "sha": "blob-1",
        }
    ]

    sync.connector.fetch_files.return_value = [
        {
            "path": "scrapy/http/request.py",
            "sha": "blob-1",
            "size": 100,
            "content": (
                "class Request:\n"
                "    pass\n"
            ),
            "source_url": (
                "https://example.com/request.py"
            ),
        }
    ]

    sync.ingestion_control.evaluate_file.return_value = Mock(
        accepted=True,
        reason="accepted",
    )

    sync.ingestion_control.normalize_file.return_value = Mock(
        stable_id="artifact-1",
        repository="scrapy/scrapy",
        artifact_type="code",
        source_path_or_object_id=(
            "scrapy/http/request.py"
        ),
        source_url=(
            "https://example.com/request.py"
        ),
        source_sha="blob-1",
        commit_sha="commit-old",
    )

    manifest_path = tmp_path / "manifest.json"

    sync.synchronize(manifest_path)

    sync.connector.fetch_repository_metadata.return_value[
        "commit_sha"
    ] = "commit-new"

    sync.connector.fetch_tree.return_value = []
    sync.connector.fetch_files.reset_mock()

    changes = sync.synchronize(manifest_path)

    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.DELETED
    assert changes[0].previous_sha == "commit-old"

    sync.connector.fetch_files.assert_called_once_with(
        sync.connector.validate_reference.return_value,
        [],
        "commit-new",
    )


def test_excluded_file_is_recorded(tmp_path: Path) -> None:
    """Excluded files should be recorded with an exclusion reason."""

    sync = create_sync()
    configure_connector(sync)

    sync.connector.fetch_tree.return_value = [
        {
            "path": "build/output.py",
            "type": "blob",
            "size": 100,
            "sha": "blob-build",
        }
    ]

    sync.connector.fetch_files.return_value = []

    sync.ingestion_control.evaluate_file.return_value = Mock(
        accepted=False,
        reason="excluded_directory",
    )

    manifest_path = tmp_path / "manifest.json"

    changes = sync.synchronize(manifest_path)

    assert changes == []

    sync.connector.fetch_files.assert_called_once_with(
        sync.connector.validate_reference.return_value,
        [],
        "commit-old",
    )


def test_failed_file_fetch_is_recorded(tmp_path: Path) -> None:
    """A missing fetch result should produce an explicit failure record."""

    sync = create_sync()
    configure_connector(sync)

    sync.connector.fetch_tree.return_value = [
        {
            "path": "scrapy/http/request.py",
            "type": "blob",
            "size": 100,
            "sha": "blob-1",
        }
    ]

    sync.connector.fetch_files.return_value = []

    sync.ingestion_control.evaluate_file.return_value = Mock(
        accepted=True,
        reason="accepted",
    )

    manifest_path = tmp_path / "manifest.json"

    changes = sync.synchronize(manifest_path)

    assert changes == []

    sync.connector.fetch_files.assert_called_once_with(
        sync.connector.validate_reference.return_value,
        ["scrapy/http/request.py"],
        "commit-old",
    )


def test_failed_artifact_from_connector_is_recorded(tmp_path: Path) -> None:
    """Connector failure records should be persisted by Sync."""

    sync = create_sync()
    configure_connector(sync)

    sync.connector.fetch_tree.return_value = [
        {
            "path": "scrapy/http/request.py",
            "type": "blob",
            "size": 100,
            "sha": "blob-1",
        }
    ]

    sync.connector.fetch_files.return_value = [
        {
            "path": "scrapy/http/request.py",
            "sha": "blob-1",
            "status": "failed",
            "failure_reason": (
                "unsupported_or_invalid_encoding"
            ),
            "source_url": (
                "https://example.com/request.py"
            ),
        }
    ]

    sync.ingestion_control.evaluate_file.return_value = Mock(
        accepted=True,
        reason="accepted",
    )

    manifest_path = tmp_path / "manifest.json"

    changes = sync.synchronize(manifest_path)

    assert changes == []

    sync.ingestion_control.normalize_file.assert_not_called()


def test_normalization_failure_is_recorded(tmp_path: Path) -> None:
    """Normalization failures should not abort the synchronization."""

    sync = create_sync()
    configure_connector(sync)

    sync.connector.fetch_tree.return_value = [
        {
            "path": "scrapy/http/request.py",
            "type": "blob",
            "size": 100,
            "sha": "blob-1",
        }
    ]

    sync.connector.fetch_files.return_value = [
        {
            "path": "scrapy/http/request.py",
            "sha": "blob-1",
            "size": 100,
            "content": (
                "class Request:\n"
                "    pass\n"
            ),
            "source_url": (
                "https://example.com/request.py"
            ),
        }
    ]

    sync.ingestion_control.evaluate_file.return_value = Mock(
        accepted=True,
        reason="accepted",
    )

    sync.ingestion_control.normalize_file.side_effect = (
        ValueError("invalid repository artifact")
    )

    manifest_path = tmp_path / "manifest.json"

    changes = sync.synchronize(manifest_path)

    assert changes == []

    sync.ingestion_control.normalize_file.assert_called_once()