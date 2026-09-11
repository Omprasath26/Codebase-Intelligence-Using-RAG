"""Tests for incremental repository synchronization."""

from pathlib import Path
from unittest.mock import Mock

from src.change_detection import ChangeType
from src.config.settings import (
    IngestionSettings,
    RepositorySettings,
    Settings,
)
from src.ingestion_control import IngestionControl
from src.repository_connector import (
    RepositoryConnector,
    RepositoryReference,
)
from src.sync import RepositorySync


def make_settings() -> Settings:
    """Create test settings."""

    return Settings(
        repository=RepositorySettings(
            url="https://github.com/scrapy/scrapy",
            ref="master",
        ),
        ingestion=IngestionSettings(
            max_file_size_bytes=1048576,
            allowed_extensions=[
                ".py",
                ".md",
            ],
            excluded_directories=[
                ".git",
                "build",
            ],
        ),
        github_token=None,
    )


def make_sync() -> RepositorySync:
    """Create a synchronization instance for tests."""

    connector = RepositoryConnector(
        make_settings()
    )

    return RepositorySync(
        connector=connector,
        ingestion_control=IngestionControl(
            make_settings()
        ),
    )


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
            "commit_sha": "commit-new",
        }
    )


def test_initial_sync_detects_added_artifact(
    tmp_path: Path,
) -> None:
    sync = make_sync()

    configure_connector(sync)

    sync.connector.fetch_tree = Mock(
        return_value=[
            {
                "path": "scrapy/http/request.py",
                "type": "blob",
                "size": 100,
                "sha": "blob-1",
            }
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            {
                "path": "scrapy/http/request.py",
                "sha": "blob-1",
                "size": 100,
                "content": "class Request:\n    pass\n",
                "source_url": (
                    "https://github.com/scrapy/scrapy/"
                    "blob/commit-new/scrapy/http/request.py"
                ),
            }
        ]
    )

    manifest_path = tmp_path / "manifest.json"

    changes = sync.synchronize(
        manifest_path
    )

    assert len(changes) == 1
    assert (
        changes[0].change_type
        == ChangeType.ADDED
    )
    assert changes[0].current_sha == "commit-new"

    assert manifest_path.exists()


def test_second_sync_detects_modified_artifact(
    tmp_path: Path,
) -> None:
    sync = make_sync()

    configure_connector(sync)

    manifest_path = tmp_path / "manifest.json"

    sync.connector.fetch_tree = Mock(
        return_value=[
            {
                "path": "scrapy/http/request.py",
                "type": "blob",
                "size": 100,
                "sha": "blob-old",
            }
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            {
                "path": "scrapy/http/request.py",
                "sha": "blob-old",
                "size": 100,
                "content": "class Request:\n    pass\n",
                "source_url": "https://example/request.py",
            }
        ]
    )

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
                "    value = 1\n"
            ),
            "source_url": "https://example/request.py",
        }
    ]

    changes = sync.synchronize(
        manifest_path
    )

    assert len(changes) == 1
    assert (
        changes[0].change_type
        == ChangeType.MODIFIED
    )
    assert changes[0].previous_sha == "commit-new"


def test_second_sync_detects_deleted_artifact(
    tmp_path: Path,
) -> None:
    sync = make_sync()

    configure_connector(sync)

    manifest_path = tmp_path / "manifest.json"

    sync.connector.fetch_tree = Mock(
        return_value=[
            {
                "path": "scrapy/http/request.py",
                "type": "blob",
                "size": 100,
                "sha": "blob-1",
            }
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            {
                "path": "scrapy/http/request.py",
                "sha": "blob-1",
                "size": 100,
                "content": "class Request:\n    pass\n",
                "source_url": "https://example/request.py",
            }
        ]
    )

    sync.synchronize(manifest_path)

    sync.connector.fetch_repository_metadata.return_value[
        "commit_sha"
    ] = "commit-new"

    sync.connector.fetch_tree.return_value = []

    sync.connector.fetch_files.return_value = []

    changes = sync.synchronize(
        manifest_path
    )

    assert len(changes) == 1
    assert (
        changes[0].change_type
        == ChangeType.DELETED
    )


def test_excluded_file_is_recorded(
    tmp_path: Path,
) -> None:
    sync = make_sync()

    configure_connector(sync)

    sync.connector.fetch_tree = Mock(
        return_value=[
            {
                "path": "build/generated.py",
                "type": "blob",
                "size": 100,
                "sha": "generated-sha",
            }
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[]
    )

    manifest_path = tmp_path / "manifest.json"

    changes = sync.synchronize(
        manifest_path
    )

    assert changes == []

    manifest = sync._load_previous_manifest(
        manifest_path
    )

    assert manifest is not None
    assert len(manifest.artifacts) == 1
    assert (
        manifest.artifacts[0].status
        == "excluded"
    )
    assert (
        manifest.artifacts[0].exclusion_reason
        == "excluded_directory"
    )


def test_unchanged_artifact_is_not_fetched_again(
    tmp_path: Path,
) -> None:
    sync = make_sync()

    configure_connector(sync)

    manifest_path = tmp_path / "manifest.json"

    tree = [
        {
            "path": "scrapy/http/request.py",
            "type": "blob",
            "size": 100,
            "sha": "blob-1",
        }
    ]

    files = [
        {
            "path": "scrapy/http/request.py",
            "sha": "blob-1",
            "size": 100,
            "content": "class Request:\n    pass\n",
            "source_url": "https://example/request.py",
        }
    ]

    sync.connector.fetch_tree = Mock(
        return_value=tree
    )

    sync.connector.fetch_files = Mock(
        return_value=files
    )

    sync.synchronize(manifest_path)

    sync.connector.fetch_files.reset_mock()

    changes = sync.synchronize(
        manifest_path
    )

    assert len(changes) == 1
    assert (
        changes[0].change_type
        == ChangeType.UNCHANGED
    )

    sync.connector.fetch_files.assert_not_called()


def test_only_modified_artifact_is_fetched(
    tmp_path: Path,
) -> None:
    sync = make_sync()

    configure_connector(sync)

    manifest_path = tmp_path / "manifest.json"

    initial_tree = [
        {
            "path": "scrapy/http/request.py",
            "type": "blob",
            "size": 100,
            "sha": "request-old",
        },
        {
            "path": "scrapy/http/response.py",
            "type": "blob",
            "size": 100,
            "sha": "response-unchanged",
        },
    ]

    initial_files = [
        {
            "path": "scrapy/http/request.py",
            "sha": "request-old",
            "size": 100,
            "content": "class Request:\n    pass\n",
            "source_url": "https://example/request.py",
        },
        {
            "path": "scrapy/http/response.py",
            "sha": "response-unchanged",
            "size": 100,
            "content": "class Response:\n    pass\n",
            "source_url": "https://example/response.py",
        },
    ]

    sync.connector.fetch_tree = Mock(
        return_value=initial_tree
    )

    sync.connector.fetch_files = Mock(
        return_value=initial_files
    )

    sync.synchronize(manifest_path)

    sync.connector.fetch_repository_metadata.return_value[
        "commit_sha"
    ] = "commit-new"

    sync.connector.fetch_tree.return_value = [
        {
            "path": "scrapy/http/request.py",
            "type": "blob",
            "size": 120,
            "sha": "request-new",
        },
        {
            "path": "scrapy/http/response.py",
            "type": "blob",
            "size": 100,
            "sha": "response-unchanged",
        },
    ]

    sync.connector.fetch_files.reset_mock()

    sync.connector.fetch_files.return_value = [
        {
            "path": "scrapy/http/request.py",
            "sha": "request-new",
            "size": 120,
            "content": (
                "class Request:\n"
                "    value = 1\n"
            ),
            "source_url": "https://example/request.py",
        }
    ]

    changes = sync.synchronize(
        manifest_path
    )

    change_types = {
        change.change_type
        for change in changes
    }

    assert change_types == {
        ChangeType.MODIFIED,
        ChangeType.UNCHANGED,
    }

    sync.connector.fetch_files.assert_called_once_with(
        sync.connector.validate_reference.return_value,
        ["scrapy/http/request.py"],
        "commit-new",
    )


def test_rename_fetches_only_new_path(
    tmp_path: Path,
) -> None:
    sync = make_sync()

    configure_connector(sync)

    manifest_path = tmp_path / "manifest.json"

    sync.connector.fetch_tree = Mock(
        return_value=[
            {
                "path": "scrapy/http/request.py",
                "type": "blob",
                "size": 100,
                "sha": "same-source-sha",
            }
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            {
                "path": "scrapy/http/request.py",
                "sha": "same-source-sha",
                "size": 100,
                "content": "class Request:\n    pass\n",
                "source_url": "https://example/request.py",
            }
        ]
    )

    sync.synchronize(manifest_path)

    sync.connector.fetch_repository_metadata.return_value[
        "commit_sha"
    ] = "commit-renamed"

    sync.connector.fetch_tree.return_value = [
        {
            "path": "scrapy/http/renamed_request.py",
            "type": "blob",
            "size": 100,
            "sha": "same-source-sha",
        }
    ]

    sync.connector.fetch_files.reset_mock()

    sync.connector.fetch_files.return_value = [
        {
            "path": "scrapy/http/renamed_request.py",
            "sha": "same-source-sha",
            "size": 100,
            "content": "class Request:\n    pass\n",
            "source_url": (
                "https://example/renamed_request.py"
            ),
        }
    ]

    changes = sync.synchronize(
        manifest_path
    )

    assert len(changes) == 1
    assert (
        changes[0].change_type
        == ChangeType.RENAMED
    )

    sync.connector.fetch_files.assert_called_once_with(
        sync.connector.validate_reference.return_value,
        ["scrapy/http/renamed_request.py"],
        "commit-renamed",
    )


def test_same_revision_still_returns_unchanged(
    tmp_path: Path,
) -> None:
    sync = make_sync()

    configure_connector(sync)

    manifest_path = tmp_path / "manifest.json"

    sync.connector.fetch_tree = Mock(
        return_value=[
            {
                "path": "scrapy/http/request.py",
                "type": "blob",
                "size": 100,
                "sha": "blob-1",
            }
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            {
                "path": "scrapy/http/request.py",
                "sha": "blob-1",
                "size": 100,
                "content": "class Request:\n    pass\n",
                "source_url": "https://example/request.py",
            }
        ]
    )

    first_changes = sync.synchronize(
        manifest_path
    )

    assert len(first_changes) == 1
    assert (
        first_changes[0].change_type
        == ChangeType.ADDED
    )

    sync.connector.fetch_files.reset_mock()

    second_changes = sync.synchronize(
        manifest_path
    )

    assert len(second_changes) == 1
    assert (
        second_changes[0].change_type
        == ChangeType.UNCHANGED
    )

    sync.connector.fetch_files.assert_not_called()