from unittest.mock import Mock

from src.config.settings import (
    IngestionSettings,
    RepositorySettings,
    Settings,
)
from src.repository_connector import RepositoryConnector


def make_settings() -> Settings:
    """Create test settings without requiring a real GitHub token."""

    return Settings(
        repository=RepositorySettings(
            url="https://github.com/scrapy/scrapy",
            ref="master",
        ),
        ingestion=IngestionSettings(
            max_file_size_bytes=1048576,
            allowed_extensions=[".py", ".md"],
            excluded_directories=[".git", "build"],
        ),
        github_token=None,
    )


def test_repository_reference_validation() -> None:
    connector = RepositoryConnector(
        make_settings()
    )

    connector.client.get = Mock(
        return_value={
            "sha": "test-commit-sha"
        }
    )

    reference = connector.validate_reference()

    assert reference.owner == "scrapy"
    assert reference.name == "scrapy"
    assert reference.ref == "master"


def test_invalid_repository_url() -> None:
    settings = make_settings()

    settings.repository.url = "https://example.com/repository"

    connector = RepositoryConnector(settings)

    try:
        connector.validate_reference()
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Invalid GitHub repository URL."


def test_fetch_file_content_decodes_base64() -> None:
    connector = RepositoryConnector(
        make_settings()
    )

    connector.client.get = Mock(
        return_value={
            "path": "README.md",
            "sha": "file-sha",
            "size": 5,
            "content": "SGVsbG8=",
            "encoding": "base64",
            "html_url": (
                "https://github.com/scrapy/scrapy/"
                "blob/test/README.md"
            ),
        }
    )

    reference = connector.validate_reference()

    result = connector.fetch_file_content(
        reference,
        "README.md",
        "test-commit-sha",
    )

    assert result["content"] == "Hello"
    assert result["commit_sha"] == "test-commit-sha"


def test_paginated_requests() -> None:
    connector = RepositoryConnector(
        make_settings()
    )

    connector.client.get = Mock(
        side_effect=[
            [{"id": 1}] * 100,
            [{"id": 2}],
        ]
    )

    result = connector.client.get_paginated("/test")

    assert len(result) == 101
    assert connector.client.get.call_count == 2


def test_fetch_tree_rejects_truncated_response() -> None:
    connector = RepositoryConnector(
        make_settings()
    )

    connector.client.get = Mock(
        return_value={
            "truncated": True,
            "tree": [],
        }
    )

    reference = connector.validate_reference()

    try:
        connector.fetch_tree(
            reference,
            "test-commit-sha",
        )
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert str(exc) == (
            "GitHub returned a truncated repository tree."
        )