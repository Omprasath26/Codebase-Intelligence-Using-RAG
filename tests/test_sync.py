"""Tests for incremental repository synchronization."""

from pathlib import Path
from unittest.mock import Mock

from src.change_detection import ChangeType
from src.config.settings import (IngestionSettings,RepositorySettings,Settings)
from src.ingestion_control import IngestionControl
from src.repository_connector import (RepositoryConnector,RepositoryReference)
from src.sync import RepositorySync


def make_settings() -> Settings:
    """Create deterministic test settings."""
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
    settings = make_settings()

    connector = RepositoryConnector(
        settings
    )

    ingestion_control = IngestionControl(
        settings
    )

    return RepositorySync(
        connector=connector,
        ingestion_control=ingestion_control,
    )


def configure_connector(sync: RepositorySync) -> None:
    """Configure deterministic repository connector mocks."""

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


def make_tree_entry(path: str,source_sha: str,size: int = 100) -> dict:
    """Create a deterministic repository tree entry."""
    return {
        "path": path,
        "type": "blob",
        "size": size,
        "sha": source_sha,
    }


def make_file(path: str,source_sha: str,content: str,size: int = 100) -> dict:
    """Create a deterministic repository file response."""
    return {
        "path": path,
        "sha": source_sha,
        "size": size,
        "content": content,
        "source_url": (
            "https://example/"
            + path
        ),
    }


def test_initial_sync_detects_added_artifact(tmp_path: Path) -> None:
    """Initial synchronization reports a new artifact as added."""

    sync = make_sync()

    configure_connector(sync)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    sync.connector.fetch_tree = Mock(
        return_value=[
            make_tree_entry(
                "scrapy/http/request.py",
                "blob-1",
            )
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            make_file(
                "scrapy/http/request.py",
                "blob-1",
                (
                    "class Request:\n"
                    "    pass\n"
                ),
            )
        ]
    )

    changes = sync.synchronize(
        manifest_path
    )

    assert len(changes) == 1

    change = changes[0]

    assert (
        change.change_type
        == ChangeType.ADDED
    )

    assert (
        change.current_sha
        == "commit-new"
    )

    assert (
        change.source_sha
        == "blob-1"
    )

    assert manifest_path.exists()


def test_second_sync_detects_modified_artifact(tmp_path: Path) -> None:
    """A changed source SHA is detected as modified."""

    sync = make_sync()

    configure_connector(sync)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    sync.connector.fetch_tree = Mock(
        return_value=[
            make_tree_entry(
                "scrapy/http/request.py",
                "blob-old",
            )
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            make_file(
                "scrapy/http/request.py",
                "blob-old",
                (
                    "class Request:\n"
                    "    pass\n"
                ),
            )
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

    sync.connector.fetch_repository_metadata.return_value[
        "commit_sha"
    ] = "commit-next"

    sync.connector.fetch_tree.return_value = [
        make_tree_entry(
            "scrapy/http/request.py",
            "blob-new",
            size=120,
        )
    ]

    sync.connector.fetch_files.reset_mock()

    sync.connector.fetch_files.return_value = [
        make_file(
            "scrapy/http/request.py",
            "blob-new",
            (
                "class Request:\n"
                "    value = 1\n"
            ),
            size=120,
        )
    ]

    changes = sync.synchronize(
        manifest_path
    )

    assert len(changes) == 1

    change = changes[0]

    assert (
        change.change_type
        == ChangeType.MODIFIED
    )

    assert (
        change.previous_sha
        == "commit-new"
    )

    assert (
        change.current_sha
        == "commit-next"
    )

    assert (
        change.source_sha
        == "blob-new"
    )

    sync.connector.fetch_files.assert_called_once_with(
        RepositoryReference(
            owner="scrapy",
            name="scrapy",
            ref="master",
        ),
        [
            "scrapy/http/request.py",
        ],
        "commit-next",
    )


def test_second_sync_detects_deleted_artifact(tmp_path: Path) -> None:
    """An artifact missing from the current tree is deleted."""

    sync = make_sync()

    configure_connector(sync)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    sync.connector.fetch_tree = Mock(
        return_value=[
            make_tree_entry(
                "scrapy/http/request.py",
                "blob-1",
            )
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            make_file(
                "scrapy/http/request.py",
                "blob-1",
                (
                    "class Request:\n"
                    "    pass\n"
                ),
            )
        ]
    )

    sync.synchronize(
        manifest_path
    )

    sync.connector.fetch_repository_metadata.return_value[
        "commit_sha"
    ] = "commit-deleted"

    sync.connector.fetch_tree.return_value = []

    sync.connector.fetch_files.reset_mock()

    changes = sync.synchronize(
        manifest_path
    )

    assert len(changes) == 1

    change = changes[0]

    assert (
        change.change_type
        == ChangeType.DELETED
    )

    assert (
        change.previous_sha
        == "commit-new"
    )

    assert (
        change.current_sha
        is None
    )

    assert (
        change.source_sha
        == "blob-1"
    )

    sync.connector.fetch_files.assert_not_called()


def test_excluded_file_is_recorded(tmp_path: Path) -> None:
    """Excluded files are persisted with an exclusion reason."""

    sync = make_sync()

    configure_connector(sync)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    sync.connector.fetch_tree = Mock(
        return_value=[
            make_tree_entry(
                "build/generated.py",
                "generated-sha",
            )
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[]
    )

    changes = sync.synchronize(
        manifest_path
    )

    assert changes == []

    sync.connector.fetch_files.assert_not_called()

    manifest = sync._load_previous_manifest(
        manifest_path
    )

    assert manifest is not None

    assert len(
        manifest.artifacts
    ) == 1

    artifact = manifest.artifacts[0]

    assert artifact.status == "excluded"

    assert (
        artifact.exclusion_reason
        == "excluded_directory"
    )

    assert (
        artifact.source_sha
        == "generated-sha"
    )

    assert (
        artifact.commit_sha
        == "commit-new"
    )


def test_unchanged_artifact_is_not_fetched_again(tmp_path: Path) -> None:
    """Unchanged artifacts are carried forward without refetching."""

    sync = make_sync()

    configure_connector(sync)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    tree = [
        make_tree_entry(
            "scrapy/http/request.py",
            "blob-1",
        )
    ]

    files = [
        make_file(
            "scrapy/http/request.py",
            "blob-1",
            (
                "class Request:\n"
                "    pass\n"
            ),
        )
    ]

    sync.connector.fetch_tree = Mock(
        return_value=tree
    )

    sync.connector.fetch_files = Mock(
        return_value=files
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

    change = second_changes[0]

    assert (
        change.change_type
        == ChangeType.UNCHANGED
    )

    assert (
        change.source_sha
        == "blob-1"
    )

    sync.connector.fetch_files.assert_not_called()


def test_only_modified_artifact_is_fetched(tmp_path: Path) -> None:
    """Only the changed artifact is fetched on the next sync."""

    sync = make_sync()

    configure_connector(sync)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    initial_tree = [
        make_tree_entry(
            "scrapy/http/request.py",
            "request-old",
        ),
        make_tree_entry(
            "scrapy/http/response.py",
            "response-unchanged",
        ),
    ]

    initial_files = [
        make_file(
            "scrapy/http/request.py",
            "request-old",
            (
                "class Request:\n"
                "    pass\n"
            ),
        ),
        make_file(
            "scrapy/http/response.py",
            "response-unchanged",
            (
                "class Response:\n"
                "    pass\n"
            ),
        ),
    ]

    sync.connector.fetch_tree = Mock(
        return_value=initial_tree
    )

    sync.connector.fetch_files = Mock(
        return_value=initial_files
    )

    first_changes = sync.synchronize(
        manifest_path
    )

    assert len(first_changes) == 2

    assert {
        change.change_type
        for change in first_changes
    } == {
        ChangeType.ADDED,
    }

    sync.connector.fetch_repository_metadata.return_value[
        "commit_sha"
    ] = "commit-next"

    sync.connector.fetch_tree.return_value = [
        make_tree_entry(
            "scrapy/http/request.py",
            "request-new",
            size=120,
        ),
        make_tree_entry(
            "scrapy/http/response.py",
            "response-unchanged",
        ),
    ]

    sync.connector.fetch_files.reset_mock()

    sync.connector.fetch_files.return_value = [
        make_file(
            "scrapy/http/request.py",
            "request-new",
            (
                "class Request:\n"
                "    value = 1\n"
            ),
            size=120,
        )
    ]

    changes = sync.synchronize(
        manifest_path
    )

    assert len(changes) == 2

    change_by_path = {
        change.current_path: change
        for change in changes
        if change.current_path is not None
    }

    assert (
        change_by_path[
            "scrapy/http/request.py"
        ].change_type
        == ChangeType.MODIFIED
    )

    assert (
        change_by_path[
            "scrapy/http/response.py"
        ].change_type
        == ChangeType.UNCHANGED
    )

    sync.connector.fetch_files.assert_called_once_with(
        RepositoryReference(
            owner="scrapy",
            name="scrapy",
            ref="master",
        ),
        [
            "scrapy/http/request.py",
        ],
        "commit-next",
    )


def test_rename_fetches_only_new_path(tmp_path: Path) -> None:
    """A renamed artifact is fetched only at its new path."""

    sync = make_sync()

    configure_connector(sync)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    sync.connector.fetch_tree = Mock(
        return_value=[
            make_tree_entry(
                "scrapy/http/request.py",
                "same-source-sha",
            )
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            make_file(
                "scrapy/http/request.py",
                "same-source-sha",
                (
                    "class Request:\n"
                    "    pass\n"
                ),
            )
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

    sync.connector.fetch_repository_metadata.return_value[
        "commit_sha"
    ] = "commit-renamed"

    sync.connector.fetch_tree.return_value = [
        make_tree_entry(
            "scrapy/http/renamed_request.py",
            "same-source-sha",
        )
    ]

    sync.connector.fetch_files.reset_mock()

    sync.connector.fetch_files.return_value = [
        make_file(
            "scrapy/http/renamed_request.py",
            "same-source-sha",
            (
                "class Request:\n"
                "    pass\n"
            ),
        )
    ]

    changes = sync.synchronize(
        manifest_path
    )

    assert len(changes) == 1

    change = changes[0]

    assert (
        change.change_type
        == ChangeType.RENAMED
    )

    assert (
        change.previous_path
        == "scrapy/http/request.py"
    )

    assert (
        change.current_path
        == "scrapy/http/renamed_request.py"
    )

    assert (
        change.source_sha
        == "same-source-sha"
    )

    sync.connector.fetch_files.assert_called_once_with(
        RepositoryReference(
            owner="scrapy",
            name="scrapy",
            ref="master",
        ),
        [
            "scrapy/http/renamed_request.py",
        ],
        "commit-renamed",
    )


def test_same_revision_still_returns_unchanged(tmp_path: Path) -> None:
    """The same repository revision produces an unchanged artifact."""

    sync = make_sync()

    configure_connector(sync)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    tree = [
        make_tree_entry(
            "scrapy/http/request.py",
            "blob-1",
        )
    ]

    files = [
        make_file(
            "scrapy/http/request.py",
            "blob-1",
            (
                "class Request:\n"
                "    pass\n"
            ),
        )
    ]

    sync.connector.fetch_tree = Mock(
        return_value=tree
    )

    sync.connector.fetch_files = Mock(
        return_value=files
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


def test_manifest_preserves_unchanged_processed_artifact(tmp_path: Path) -> None:
    """The current manifest retains unchanged processed artifacts."""

    sync = make_sync()

    configure_connector(sync)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    sync.connector.fetch_tree = Mock(
        return_value=[
            make_tree_entry(
                "scrapy/http/request.py",
                "blob-1",
            )
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            make_file(
                "scrapy/http/request.py",
                "blob-1",
                (
                    "class Request:\n"
                    "    pass\n"
                ),
            )
        ]
    )

    sync.synchronize(
        manifest_path
    )

    sync.connector.fetch_files.reset_mock()

    sync.synchronize(
        manifest_path
    )

    manifest = sync._load_previous_manifest(
        manifest_path
    )

    assert manifest is not None

    processed_artifacts = [
        artifact
        for artifact in manifest.artifacts
        if artifact.status == "processed"
    ]

    assert len(
        processed_artifacts
    ) == 1

    artifact = processed_artifacts[0]

    assert (
        artifact.source_path_or_object_id
        == "scrapy/http/request.py"
    )

    assert (
        artifact.source_sha
        == "blob-1"
    )

    assert (
        artifact.commit_sha
        == "commit-new"
    )


def test_deleted_artifact_is_not_present_in_current_manifest(tmp_path: Path) -> None:
    """Deleted artifacts are not retained as current processed state."""

    sync = make_sync()

    configure_connector(sync)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    sync.connector.fetch_tree = Mock(
        return_value=[
            make_tree_entry(
                "scrapy/http/request.py",
                "blob-1",
            )
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            make_file(
                "scrapy/http/request.py",
                "blob-1",
                (
                    "class Request:\n"
                    "    pass\n"
                ),
            )
        ]
    )

    sync.synchronize(
        manifest_path
    )

    sync.connector.fetch_repository_metadata.return_value[
        "commit_sha"
    ] = "commit-deleted"

    sync.connector.fetch_tree.return_value = []

    sync.connector.fetch_files.reset_mock()

    changes = sync.synchronize(
        manifest_path
    )

    assert len(changes) == 1

    assert (
        changes[0].change_type
        == ChangeType.DELETED
    )

    manifest = sync._load_previous_manifest(
        manifest_path
    )

    assert manifest is not None

    processed_artifacts = [
        artifact
        for artifact in manifest.artifacts
        if artifact.status == "processed"
    ]

    assert processed_artifacts == []

    sync.connector.fetch_files.assert_not_called()


def test_failed_repository_file_is_recorded(tmp_path: Path) -> None:
    """A repository file fetch failure is recorded in the manifest."""

    sync = make_sync()

    configure_connector(sync)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    sync.connector.fetch_tree = Mock(
        return_value=[
            make_tree_entry(
                "scrapy/http/request.py",
                "blob-failed",
            )
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            {
                "path": "scrapy/http/request.py",
                "sha": "blob-failed",
                "size": 100,
                "status": "failed",
                "failure_reason": (
                    "repository_file_fetch_failed"
                ),
                "source_url": (
                    "https://example/request.py"
                ),
            }
        ]
    )

    changes = sync.synchronize(
        manifest_path
    )

    assert changes == []

    manifest = sync._load_previous_manifest(
        manifest_path
    )

    assert manifest is not None

    failed_artifacts = [
        artifact
        for artifact in manifest.artifacts
        if artifact.status == "failed"
    ]

    assert len(
        failed_artifacts
    ) == 1

    assert (
        failed_artifacts[0].source_path_or_object_id
        == "scrapy/http/request.py"
    )

    assert (
        failed_artifacts[0].failure_reason
        == "repository_file_fetch_failed"
    )


def test_invalid_normalized_artifact_is_recorded_as_failed(tmp_path: Path) -> None:
    """Normalization failures become explicit manifest failures."""

    sync = make_sync()

    configure_connector(sync)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    sync.connector.fetch_tree = Mock(
        return_value=[
            make_tree_entry(
                "scrapy/http/request.py",
                "blob-invalid",
            )
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            {
                "path": "scrapy/http/request.py",
                "sha": "blob-invalid",
                "size": 100,
                "content": None,
                "source_url": (
                    "https://example/request.py"
                ),
            }
        ]
    )

    changes = sync.synchronize(
        manifest_path
    )

    assert changes == []

    manifest = sync._load_previous_manifest(
        manifest_path
    )

    assert manifest is not None

    failed_artifacts = [
        artifact
        for artifact in manifest.artifacts
        if artifact.status == "failed"
    ]

    assert len(
        failed_artifacts
    ) == 1

    assert (
        failed_artifacts[0].source_path_or_object_id
        == "scrapy/http/request.py"
    )

    assert (
        failed_artifacts[0].commit_sha
        == "commit-new"
    )


def test_manifest_records_current_repository_revision(tmp_path: Path) -> None:
    """Manifest state remains tied to the repository revision."""

    sync = make_sync()

    configure_connector(sync)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    sync.connector.fetch_tree = Mock(
        return_value=[
            make_tree_entry(
                "scrapy/http/request.py",
                "blob-1",
            )
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            make_file(
                "scrapy/http/request.py",
                "blob-1",
                (
                    "class Request:\n"
                    "    pass\n"
                ),
            )
        ]
    )

    sync.synchronize(
        manifest_path
    )

    manifest = sync._load_previous_manifest(
        manifest_path
    )

    assert manifest is not None

    assert (
        manifest.repository
        == "scrapy/scrapy"
    )

    assert (
        manifest.ref
        == "master"
    )

    assert (
        manifest.commit_sha
        == "commit-new"
    )

    assert (
        manifest.status
        == "completed"
    )


def test_tree_is_always_checked_for_revision_changes(tmp_path: Path) -> None:
    """Synchronization checks the repository tree on each run."""

    sync = make_sync()

    configure_connector(sync)

    manifest_path = (
        tmp_path / "manifest.json"
    )

    sync.connector.fetch_tree = Mock(
        return_value=[
            make_tree_entry(
                "scrapy/http/request.py",
                "blob-1",
            )
        ]
    )

    sync.connector.fetch_files = Mock(
        return_value=[
            make_file(
                "scrapy/http/request.py",
                "blob-1",
                (
                    "class Request:\n"
                    "    pass\n"
                ),
            )
        ]
    )

    sync.synchronize(
        manifest_path
    )

    sync.connector.fetch_tree.reset_mock()
    sync.connector.fetch_files.reset_mock()

    sync.synchronize(
        manifest_path
    )

    sync.connector.fetch_tree.assert_called_once_with(
        RepositoryReference(
            owner="scrapy",
            name="scrapy",
            ref="master",
        ),
        "commit-new",
    )