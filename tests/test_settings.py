"""Tests for the existing configuration system and Task39A extensions."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import (
    ConfigurationError,
    IngestionSettings,
    RepositorySettings,
    Settings,
    load_settings,
)


def create_config_file(tmp_path: Path) -> Path:
    """Create a minimal valid project configuration file."""

    config_path = tmp_path / "config.yaml"

    config = {
        "repository": {
            "url": "https://github.com/scrapy/scrapy",
            "ref": "master",
        },
        "ingestion": {
            "max_file_size_bytes": 1_000_000,
            "allowed_extensions": [".py", ".md"],
            "excluded_directories": [".git", "__pycache__"],
        },
    }

    config_path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    return config_path


def test_existing_repository_and_ingestion_settings_are_preserved() -> None:
    """Existing settings models remain available."""

    repository = RepositorySettings(
        url="https://github.com/scrapy/scrapy",
        ref="master",
    )

    ingestion = IngestionSettings(
        max_file_size_bytes=1_000_000,
        allowed_extensions=[".py"],
        excluded_directories=[".git"],
    )

    settings = Settings(
        repository=repository,
        ingestion=ingestion,
    )

    assert settings.repository.url == "https://github.com/scrapy/scrapy"
    assert settings.repository.ref == "master"
    assert settings.ingestion.max_file_size_bytes == 1_000_000


def test_load_settings_preserves_existing_yaml_configuration(tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> None:
    """The original YAML configuration continues to load."""

    config_path = create_config_file(tmp_path)

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("APP_DEBUG", raising=False)

    settings = load_settings(config_path)

    assert settings.repository.url == "https://github.com/scrapy/scrapy"
    assert settings.repository.ref == "master"
    assert settings.environment == "development"
    assert settings.api_port == 8000
    assert settings.auth_enabled is False


def test_environment_variables_are_loaded(tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime settings can be configured through environment variables."""

    config_path = create_config_file(tmp_path)

    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_DEBUG", "true")
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("MAX_QUERY_LENGTH", "2000")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("DATA_DIR", "runtime-data")
    monkeypatch.setenv("INDEX_DIR", "runtime-data/index")
    monkeypatch.setenv("RECOVERY_ENABLED", "true")
    monkeypatch.setenv("ALLOW_AUTOMATIC_REBUILD", "false")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = load_settings(config_path)

    assert settings.environment == "test"
    assert settings.debug is True
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 9000
    assert settings.auth_enabled is True
    assert settings.api_key == "test-secret"
    assert settings.max_query_length == 2000
    assert settings.request_timeout_seconds == 30
    assert settings.data_dir == Path("runtime-data")
    assert settings.index_dir == Path("runtime-data/index")
    assert settings.recovery_enabled is True
    assert settings.allow_automatic_rebuild is False
    assert settings.log_level == "DEBUG"


def test_authentication_requires_api_key() -> None:
    """Authentication cannot be enabled without an API key."""

    with pytest.raises(
        ValidationError,
        match="api_key is required",
    ):
        Settings(
            repository={
                "url": "https://github.com/scrapy/scrapy",
                "ref": "master",
            },
            ingestion={
                "max_file_size_bytes": 1_000_000,
                "allowed_extensions": [".py"],
                "excluded_directories": [".git"],
            },
            auth_enabled=True,
        )


def test_production_debug_mode_is_rejected() -> None:
    """Production mode cannot run with debug enabled."""

    with pytest.raises(
        ValidationError,
        match="debug must be false",
    ):
        Settings(
            repository={
                "url": "https://github.com/scrapy/scrapy",
                "ref": "master",
            },
            ingestion={
                "max_file_size_bytes": 1_000_000,
                "allowed_extensions": [".py"],
                "excluded_directories": [".git"],
            },
            environment="production",
            debug=True,
        )


def test_production_automatic_rebuild_is_rejected() -> None:
    """Automatic rebuild is disabled in production."""

    with pytest.raises(
        ValidationError,
        match="allow_automatic_rebuild must be false",
    ):
        Settings(
            repository={
                "url": "https://github.com/scrapy/scrapy",
                "ref": "master",
            },
            ingestion={
                "max_file_size_bytes": 1_000_000,
                "allowed_extensions": [".py"],
                "excluded_directories": [".git"],
            },
            environment="production",
            allow_automatic_rebuild=True,
        )


def test_invalid_boolean_environment_variable_is_rejected(tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid boolean environment values fail clearly."""

    config_path = create_config_file(tmp_path)

    monkeypatch.setenv("APP_DEBUG", "sometimes")

    with pytest.raises(ConfigurationError, match="APP_DEBUG"):
        load_settings(config_path)


def test_invalid_api_port_is_rejected() -> None:
    """API ports must be valid TCP port numbers."""

    with pytest.raises(
        ValidationError,
        match="api_port must be between",
    ):
        Settings(
            repository={
                "url": "https://github.com/scrapy/scrapy",
                "ref": "master",
            },
            ingestion={
                "max_file_size_bytes": 1_000_000,
                "allowed_extensions": [".py"],
                "excluded_directories": [".git"],
            },
            api_port=70000,
        )


def test_public_summary_does_not_expose_secrets() -> None:
    """Public configuration summaries must not contain secret values."""

    settings = Settings(
        repository={
            "url": "https://github.com/scrapy/scrapy",
            "ref": "master",
        },
        ingestion={
            "max_file_size_bytes": 1_000_000,
            "allowed_extensions": [".py"],
            "excluded_directories": [".git"],
        },
        github_token="github-secret",
        auth_enabled=True,
        api_key="api-secret",
    )

    summary = settings.public_summary()

    assert "github-secret" not in str(summary)
    assert "api-secret" not in str(summary)
    assert summary["github_token_configured"] is True
    assert summary["api_key_configured"] is True


def test_production_configuration_is_available() -> None:
    """A secure production configuration can be created."""

    settings = Settings(
        repository={
            "url": "https://github.com/scrapy/scrapy",
            "ref": "master",
        },
        ingestion={
            "max_file_size_bytes": 1_000_000,
            "allowed_extensions": [".py"],
            "excluded_directories": [".git"],
        },
        environment="production",
        debug=False,
        auth_enabled=True,
        api_key="production-secret",
        recovery_enabled=True,
        allow_automatic_rebuild=False,
    )

    assert settings.is_production is True
    assert settings.authentication_required is True