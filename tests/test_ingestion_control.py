from src.config.settings import (
    IngestionSettings,
    RepositorySettings,
    Settings,
)
from src.ingestion_control import (
    IngestionControl,
)


def make_settings() -> Settings:
    """Create settings for ingestion-control tests."""

    return Settings(
        repository=RepositorySettings(
            url="https://github.com/scrapy/scrapy",
            ref="master",
        ),
        ingestion=IngestionSettings(
            max_file_size_bytes=100,
            allowed_extensions=[
                ".py",
                ".md",
                ".rst",
                ".toml",
                ".yaml",
                ".yml",
                ".json",
                ".ini",
                ".cfg",
                ".txt",
            ],
            excluded_directories=[
                ".git",
                "__pycache__",
                ".pytest_cache",
                "build",
                "dist",
                "vendor",
            ],
        ),
        github_token=None,
    )


def test_classifies_python_file_as_code() -> None:
    controller = IngestionControl(make_settings())

    artifact_type = controller.classify_artifact(
        {
            "path": "scrapy/core.py",
        }
    )

    assert artifact_type == "code"


def test_classifies_test_file_as_test() -> None:
    controller = IngestionControl(make_settings())

    artifact_type = controller.classify_artifact(
        {
            "path": "tests/test_core.py",
        }
    )

    assert artifact_type == "test"


def test_excludes_configured_directory() -> None:
    controller = IngestionControl(make_settings())

    decision = controller.evaluate_file(
        {
            "path": "build/output.py",
            "size": 20,
            "content": "print('hello')",
        }
    )

    assert decision.accepted is False
    assert decision.reason == "excluded_directory"


def test_excludes_file_over_size_limit() -> None:
    controller = IngestionControl(make_settings())

    decision = controller.evaluate_file(
        {
            "path": "scrapy/core.py",
            "size": 101,
            "content": "print('hello')",
        }
    )

    assert decision.accepted is False
    assert decision.reason == "file_size_exceeds_limit"


def test_excludes_unsupported_file_type() -> None:
    controller = IngestionControl(make_settings())

    decision = controller.evaluate_file(
        {
            "path": "scrapy/image.png",
            "size": 20,
            "content": "not actually an image",
        }
    )

    assert decision.accepted is False
    assert decision.reason == "unsupported_file_type"


def test_excludes_potential_secret() -> None:
    controller = IngestionControl(make_settings())

    decision = controller.evaluate_file(
        {
            "path": "config/settings.py",
            "size": 50,
            "content": (
                "api_key = "
                "'abcdefghijklmnopqrstuvwxyz123456'"
            ),
        }
    )

    assert decision.accepted is False
    assert decision.reason == "potential_secret_detected"


def test_normalizes_file_with_provenance() -> None:
    controller = IngestionControl(make_settings())

    artifact = controller.normalize_file(
        raw_artifact={
            "path": "scrapy/core.py",
            "sha": "file-sha",
            "size": 20,
            "content": "print('hello')",
            "source_url": (
                "https://github.com/scrapy/scrapy/"
                "blob/abc123/scrapy/core.py"
            ),
        },
        repository="scrapy/scrapy",
        ref="master",
        commit_sha="abc123",
    )

    assert artifact.repository == "scrapy/scrapy"
    assert artifact.artifact_type == "code"
    assert artifact.source_path_or_object_id == "scrapy/core.py"
    assert artifact.source_url is not None
    assert artifact.commit_sha == "abc123"
    assert artifact.ref == "master"
    assert artifact.content == "print('hello')"
    assert artifact.stable_id


def test_stable_id_is_deterministic() -> None:
    controller = IngestionControl(make_settings())

    first = controller.normalize_file(
        raw_artifact={
            "path": "scrapy/core.py",
            "size": 20,
            "content": "first",
        },
        repository="scrapy/scrapy",
        ref="master",
        commit_sha="abc123",
    )

    second = controller.normalize_file(
        raw_artifact={
            "path": "scrapy/core.py",
            "size": 20,
            "content": "second",
        },
        repository="scrapy/scrapy",
        ref="master",
        commit_sha="different",
    )

    assert first.stable_id == second.stable_id