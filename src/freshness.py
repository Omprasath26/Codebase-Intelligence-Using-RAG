"""Repository version and index freshness models for Problem Statement 4."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FreshnessStatus(str, Enum):
    """State of an index relative to the repository revision it should serve."""

    CURRENT = "current"
    BEHIND = "behind"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"


class PipelineVersion(BaseModel):
    """Version identifiers that determine whether an index is compatible."""

    model_config = ConfigDict(frozen=True)

    index_version: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    chunking_version: str = Field(min_length=1)


class FreshnessPolicy(BaseModel):
    """Policy for deciding whether repository/index revisions may be served."""

    model_config = ConfigDict(frozen=True)

    require_latest_revision: bool = True
    historical_window_start: datetime | None = None
    historical_window_end: datetime | None = None

    @model_validator(mode="after")
    def validate_historical_window(self) -> "FreshnessPolicy":
        if (self.historical_window_start is None) != (
            self.historical_window_end is None
        ):
            raise ValueError(
                "historical_window_start and historical_window_end "
                "must be provided together."
            )

        if (
            self.historical_window_start is not None
            and self.historical_window_end is not None
            and self.historical_window_start > self.historical_window_end):
            raise ValueError(
                "historical_window_start must not be after "
                "historical_window_end."
            )

        return self

    @property
    def has_historical_window(self) -> bool:
        """Return whether an explicit historical time window is configured."""

        return (
            self.historical_window_start is not None
            and self.historical_window_end is not None
        )


class FreshnessState(BaseModel):
    """Snapshot of repository and index revision/version state."""

    model_config = ConfigDict(frozen=True)

    repository: str = Field(min_length=1)
    ref: str = Field(min_length=1)
    current_revision: str | None = None
    indexed_revision: str | None = None
    pipeline_version: PipelineVersion
    status: FreshnessStatus

    @property
    def is_current(self) -> bool:
        """Return whether the indexed revision matches the repository revision."""

        return self.status == FreshnessStatus.CURRENT

    @property
    def is_behind(self) -> bool:
        """Return whether the index is known to be behind the repository."""

        return self.status == FreshnessStatus.BEHIND

    def can_serve(self, policy: FreshnessPolicy) -> bool:
        """Return whether this state may be served under the given policy."""

        if self.status in {
            FreshnessStatus.UNAVAILABLE,
            FreshnessStatus.INCOMPATIBLE}:
            return False

        if self.status == FreshnessStatus.BEHIND:
            return not policy.require_latest_revision

        return True


class FreshnessEvaluator:
    """Evaluate revision and pipeline compatibility without performing sync work."""

    def evaluate(
        self,
        repository: str,
        ref: str,
        current_revision: str | None,
        indexed_revision: str | None,
        pipeline_version: PipelineVersion,
        expected_pipeline_version: PipelineVersion | None = None,
        policy: FreshnessPolicy | None = None) -> FreshnessState:
        """Create a deterministic freshness state from known revision metadata."""

        if not repository:
            raise ValueError("repository must not be empty.")

        if not ref:
            raise ValueError("ref must not be empty.")

        if expected_pipeline_version is not None and (pipeline_version != expected_pipeline_version):
            status = FreshnessStatus.INCOMPATIBLE
        elif current_revision is None or indexed_revision is None:
            status = FreshnessStatus.UNAVAILABLE
        elif current_revision == indexed_revision:
            status = FreshnessStatus.CURRENT
        else:
            status = FreshnessStatus.BEHIND

        return FreshnessState(
            repository=repository,
            ref=ref,
            current_revision=current_revision,
            indexed_revision=indexed_revision,
            pipeline_version=pipeline_version,
            status=status,
        )