"""Tests for the sync trigger and scheduler."""

from __future__ import annotations
import threading
import time
import pytest
from src.scheduler import (ManualSyncTrigger,SyncScheduler)


def test_manual_trigger_executes_sync_callable() -> None:
    calls: list[str] = []

    def sync() -> str:
        calls.append("sync")
        return "completed"

    trigger = ManualSyncTrigger(sync)

    result = trigger.trigger()

    assert result == "completed"
    assert calls == ["sync"]


def test_manual_trigger_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="sync_callable must be callable"):
        ManualSyncTrigger("invalid")  # type: ignore[arg-type]


def test_scheduler_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="sync_callable must be callable"):
        SyncScheduler(
            sync_callable="invalid",  # type: ignore[arg-type]
            interval_seconds=1,
        )


def test_scheduler_rejects_non_positive_interval() -> None:
    def sync() -> None:
        pass

    with pytest.raises(
        ValueError,
        match="interval_seconds must be greater than zero",):
        SyncScheduler(
            sync_callable=sync,
            interval_seconds=0,
        )


def test_scheduler_manual_trigger_executes_sync() -> None:
    calls: list[int] = []

    def sync() -> str:
        calls.append(1)
        return "ok"

    scheduler = SyncScheduler(
        sync_callable=sync,
        interval_seconds=60,
    )

    result = scheduler.trigger()

    assert result == "ok"
    assert calls == [1]
    assert scheduler.last_result == "ok"
    assert scheduler.last_error is None


def test_scheduler_records_sync_failure() -> None:
    error = RuntimeError("sync failed")

    def sync() -> None:
        raise error

    scheduler = SyncScheduler(
        sync_callable=sync,
        interval_seconds=60,
    )

    with pytest.raises(RuntimeError, match="sync failed"):
        scheduler.trigger()

    assert scheduler.last_result is None
    assert scheduler.last_error is error


def test_scheduler_success_clears_previous_error() -> None:
    calls = [0]

    def sync() -> str:
        calls[0] += 1

        if calls[0] == 1:
            raise RuntimeError("temporary failure")

        return "success"

    scheduler = SyncScheduler(
        sync_callable=sync,
        interval_seconds=60,
    )

    with pytest.raises(RuntimeError):
        scheduler.trigger()

    assert scheduler.last_error is not None

    result = scheduler.trigger()

    assert result == "success"
    assert scheduler.last_result == "success"
    assert scheduler.last_error is None


def test_scheduler_start_is_idempotent() -> None:
    calls: list[int] = []
    started = threading.Event()

    def sync() -> None:
        calls.append(1)
        started.set()

    scheduler = SyncScheduler(
        sync_callable=sync,
        interval_seconds=0.05,
    )

    scheduler.start()
    scheduler.start()

    assert scheduler.running is True

    scheduler.stop()

    assert scheduler.running is False


def test_scheduler_runs_periodically() -> None:
    calls: list[int] = []
    called = threading.Event()

    def sync() -> None:
        calls.append(1)
        called.set()

    scheduler = SyncScheduler(
        sync_callable=sync,
        interval_seconds=0.05,
    )

    scheduler.start()

    try:
        assert called.wait(timeout=1.0)
    finally:
        scheduler.stop()

    assert len(calls) >= 1
    assert scheduler.running is False


def test_scheduler_continues_after_scheduled_failure() -> None:
    calls = [0]
    second_run = threading.Event()

    def sync() -> None:
        calls[0] += 1

        if calls[0] == 1:
            raise RuntimeError("temporary failure")

        second_run.set()

    scheduler = SyncScheduler(
        sync_callable=sync,
        interval_seconds=0.05,
    )

    scheduler.start()

    try:
        assert second_run.wait(timeout=1.0)
    finally:
        scheduler.stop()

    assert calls[0] >= 2
    assert scheduler.running is False


def test_scheduler_stop_is_idempotent() -> None:
    def sync() -> None:
        pass

    scheduler = SyncScheduler(
        sync_callable=sync,
        interval_seconds=60,
    )

    scheduler.stop()
    scheduler.stop()

    assert scheduler.running is False


def test_scheduler_does_not_execute_immediately_on_start() -> None:
    calls: list[int] = []

    def sync() -> None:
        calls.append(1)

    scheduler = SyncScheduler(
        sync_callable=sync,
        interval_seconds=0.2,
    )

    scheduler.start()

    try:
        time.sleep(0.05)
        assert calls == []
    finally:
        scheduler.stop()