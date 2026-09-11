from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from src.freshness import (FreshnessEvaluator,FreshnessPolicy,FreshnessStatus,FreshnessState,PipelineVersion)


def make_version() -> PipelineVersion:
    return PipelineVersion(
        index_version="v1",
        embedding_model="model-a",
        chunking_version="chunk-v1")


def test_pipeline_version_tracks_index_embedding_and_chunking() -> None:
    version = make_version()

    assert version.index_version == "v1"
    assert version.embedding_model == "model-a"
    assert version.chunking_version == "chunk-v1"


def test_pipeline_version_rejects_empty_identifiers() -> None:
    with pytest.raises(ValidationError):
        PipelineVersion(
            index_version="",
            embedding_model="model-a",
            chunking_version="chunk-v1",
        )


def test_historical_window_requires_both_bounds() -> None:
    with pytest.raises(ValidationError):
        FreshnessPolicy(
            historical_window_start=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            )
        )


def test_historical_window_rejects_reverse_bounds() -> None:
    with pytest.raises(ValidationError):
        FreshnessPolicy(
            historical_window_start=datetime(
                2026,
                2,
                1,
                tzinfo=timezone.utc,
            ),
            historical_window_end=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        )


def test_current_revision_produces_current_state() -> None:
    state = FreshnessEvaluator().evaluate(
        repository="scrapy/scrapy",
        ref="master",
        current_revision="abc",
        indexed_revision="abc",
        pipeline_version=make_version(),
    )

    assert state.status == FreshnessStatus.CURRENT
    assert state.is_current
    assert not state.is_behind


def test_different_revision_produces_behind_state() -> None:
    state = FreshnessEvaluator().evaluate(
        repository="scrapy/scrapy",
        ref="master",
        current_revision="new",
        indexed_revision="old",
        pipeline_version=make_version(),
    )

    assert state.status == FreshnessStatus.BEHIND
    assert state.is_behind


def test_missing_revision_produces_unavailable_state() -> None:
    state = FreshnessEvaluator().evaluate(
        repository="scrapy/scrapy",
        ref="master",
        current_revision="new",
        indexed_revision=None,
        pipeline_version=make_version(),
    )

    assert state.status == FreshnessStatus.UNAVAILABLE


def test_incompatible_pipeline_version_has_priority() -> None:
    current = make_version()

    expected = PipelineVersion(
        index_version="v2",
        embedding_model="model-a",
        chunking_version="chunk-v1",
    )

    state = FreshnessEvaluator().evaluate(
        repository="scrapy/scrapy",
        ref="master",
        current_revision="same",
        indexed_revision="same",
        pipeline_version=current,
        expected_pipeline_version=expected,
    )

    assert state.status == FreshnessStatus.INCOMPATIBLE


def test_policy_can_allow_serving_non_latest_revision() -> None:
    policy = FreshnessPolicy(
        require_latest_revision=False
    )

    state = FreshnessEvaluator().evaluate(
        repository="scrapy/scrapy",
        ref="master",
        current_revision="new",
        indexed_revision="old",
        pipeline_version=make_version(),
        policy=policy)

    assert state.status == FreshnessStatus.BEHIND
    assert state.can_serve(policy)


def test_policy_default_requires_latest_revision() -> None:
    policy = FreshnessPolicy()

    state = FreshnessEvaluator().evaluate(
        repository="scrapy/scrapy",
        ref="master",
        current_revision="new",
        indexed_revision="old",
        pipeline_version=make_version(),
        policy=policy)

    assert state.status == FreshnessStatus.BEHIND
    assert not state.can_serve(policy)


def test_freshness_state_is_immutable() -> None:
    state = FreshnessState(
        repository="scrapy/scrapy",
        ref="master",
        current_revision="abc",
        indexed_revision="abc",
        pipeline_version=make_version(),
        status=FreshnessStatus.CURRENT)

    with pytest.raises(ValidationError):
        state.status = FreshnessStatus.BEHIND