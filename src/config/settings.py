"""Application configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class RepositorySettings(BaseModel):
    """Configuration required to identify the source repository."""

    url: str
    ref: str


class IngestionSettings(BaseModel):
    """Configuration controlling repository ingestion boundaries."""

    max_file_size_bytes: int
    allowed_extensions: list[str]
    excluded_directories: list[str]


class Settings(BaseModel):
    """Complete application settings used by the system."""

    repository: RepositorySettings
    ingestion: IngestionSettings
    github_token: str | None = None


def load_settings(config_path: str | Path = "config/config.yaml",) -> Settings:
    """Load YAML configuration and environment-based secrets."""

    load_dotenv()

    path = Path(config_path)

    with path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file) or {}

    config["github_token"] = os.getenv("GITHUB_TOKEN")

    return Settings.model_validate(config)

