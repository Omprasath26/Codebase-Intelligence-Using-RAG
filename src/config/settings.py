"""Application configuration models and settings loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigurationError(ValueError):
    """Raised when an environment configuration value is invalid."""


class RepositorySettings(BaseModel):
    """Repository configuration."""

    model_config = ConfigDict(extra="ignore")

    url: str
    ref: str = "main"


class IngestionSettings(BaseModel):
    """Repository ingestion configuration."""

    model_config = ConfigDict(extra="ignore")

    max_file_size_bytes: int = 1_000_000

    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".py", ".md"]
    )

    excluded_directories: list[str] = Field(
        default_factory=lambda: [".git", "__pycache__"]
    )


class Settings(BaseModel):
    """Complete application configuration."""

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
    )

    repository: RepositorySettings
    ingestion: IngestionSettings

    environment: str = "development"
    debug: bool = False

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    auth_enabled: bool = False
    api_key: str | None = None

    github_token: str | None = None

    max_query_length: int = 1000
    request_timeout_seconds: int = 60

    data_dir: Path = Path("data")
    index_dir: Path = Path("data/index")

    recovery_enabled: bool = False
    allow_automatic_rebuild: bool = False

    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        """Return True when the application runs in production."""

        return self.environment.lower() == "production"

    @property
    def authentication_required(self) -> bool:
        """Return whether API authentication is required."""

        return self.auth_enabled

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        """Validate cross-field configuration rules."""

        if self.auth_enabled and not self.api_key:
            raise ValueError(
                "api_key is required when auth_enabled is true"
            )

        if self.is_production and self.debug:
            raise ValueError(
                "debug must be false in production"
            )

        if self.is_production and self.allow_automatic_rebuild:
            raise ValueError(
                "allow_automatic_rebuild must be false in production"
            )

        if not 1 <= self.api_port <= 65535:
            raise ValueError(
                "api_port must be between 1 and 65535"
            )

        if self.max_query_length <= 0:
            raise ValueError(
                "max_query_length must be greater than zero"
            )

        if self.request_timeout_seconds <= 0:
            raise ValueError(
                "request_timeout_seconds must be greater than zero"
            )

        return self

    def public_summary(self) -> dict[str, Any]:
        """Return a safe configuration summary without secret values."""

        return {
            "environment": self.environment,
            "debug": self.debug,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "auth_enabled": self.auth_enabled,
            "authentication_required": self.authentication_required,
            "github_token_configured": bool(self.github_token),
            "api_key_configured": bool(self.api_key),
            "repository": {
                "url": self.repository.url,
                "ref": self.repository.ref,
            },
            "ingestion": {
                "max_file_size_bytes": (
                    self.ingestion.max_file_size_bytes
                ),
                "allowed_extensions": list(
                    self.ingestion.allowed_extensions
                ),
                "excluded_directories": list(
                    self.ingestion.excluded_directories
                ),
            },
            "max_query_length": self.max_query_length,
            "request_timeout_seconds": (
                self.request_timeout_seconds
            ),
            "data_dir": str(self.data_dir),
            "index_dir": str(self.index_dir),
            "recovery_enabled": self.recovery_enabled,
            "allow_automatic_rebuild": (
                self.allow_automatic_rebuild
            ),
            "log_level": self.log_level,
        }


def _parse_bool(value: str,variable_name: str) -> bool:
    """Parse a boolean environment variable safely."""

    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes", "y", "on"}:
        return True

    if normalized in {"false", "0", "no", "n", "off"}:
        return False

    raise ConfigurationError(
        f"{variable_name} must be a boolean value, got {value!r}"
    )


def _env_int(name: str,default: int) -> int:
    """Read an optional integer environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be an integer, got {value!r}"
        ) from exc


def _apply_environment_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply supported environment variable overrides."""

    repository = dict(config.get("repository") or {})
    ingestion = dict(config.get("ingestion") or {})

    config["repository"] = repository
    config["ingestion"] = ingestion

    if "APP_ENV" in os.environ:
        config["environment"] = os.environ["APP_ENV"]

    if "APP_DEBUG" in os.environ:
        config["debug"] = _parse_bool(
            os.environ["APP_DEBUG"],
            "APP_DEBUG",
        )

    if "API_HOST" in os.environ:
        config["api_host"] = os.environ["API_HOST"]

    if "API_PORT" in os.environ:
        config["api_port"] = _env_int(
            "API_PORT",
            8000,
        )

    if "AUTH_ENABLED" in os.environ:
        config["auth_enabled"] = _parse_bool(
            os.environ["AUTH_ENABLED"],
            "AUTH_ENABLED",
        )

    if "API_KEY" in os.environ:
        config["api_key"] = os.environ["API_KEY"]

    if "GITHUB_TOKEN" in os.environ:
        config["github_token"] = os.environ["GITHUB_TOKEN"]

    if "MAX_QUERY_LENGTH" in os.environ:
        config["max_query_length"] = _env_int(
            "MAX_QUERY_LENGTH",
            1000,
        )

    if "REQUEST_TIMEOUT_SECONDS" in os.environ:
        config["request_timeout_seconds"] = _env_int(
            "REQUEST_TIMEOUT_SECONDS",
            60,
        )

    if "DATA_DIR" in os.environ:
        config["data_dir"] = Path(os.environ["DATA_DIR"])

    if "INDEX_DIR" in os.environ:
        config["index_dir"] = Path(os.environ["INDEX_DIR"])

    if "RECOVERY_ENABLED" in os.environ:
        config["recovery_enabled"] = _parse_bool(
            os.environ["RECOVERY_ENABLED"],
            "RECOVERY_ENABLED",
        )

    if "ALLOW_AUTOMATIC_REBUILD" in os.environ:
        config["allow_automatic_rebuild"] = _parse_bool(
            os.environ["ALLOW_AUTOMATIC_REBUILD"],
            "ALLOW_AUTOMATIC_REBUILD",
        )

    if "LOG_LEVEL" in os.environ:
        config["log_level"] = os.environ["LOG_LEVEL"]

    return config


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from YAML and environment variables."""

    if config_path is None:
        config_path = Path("config/config.yaml")
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file does not exist: {config_path}"
        )

    try:
        with config_path.open(
            "r",
            encoding="utf-8",
        ) as config_file:
            loaded_config = yaml.safe_load(config_file) or {}

    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML configuration: {config_path}"
        ) from exc

    if not isinstance(loaded_config, dict):
        raise ConfigurationError(
            "The YAML configuration root must be a mapping"
        )

    config = _apply_environment_overrides(
        dict(loaded_config)
    )

    return Settings.model_validate(config)