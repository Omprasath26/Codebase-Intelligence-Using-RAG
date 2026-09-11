"""Sync trigger and scheduler for Problem Statement 4."""

from __future__ import annotations
import threading
import time
from collections.abc import Callable
from typing import Any


class SyncScheduler:
    """Trigger repository synchronization manually or on an interval.

    The scheduler owns execution timing only. The injected synchronization
    callable remains responsible for repository revision checks, change
    detection, affected-artifact processing, indexing, validation, and
    atomic state management.
    """

    def __init__(self,sync_callable: Callable[[], Any],interval_seconds: float) -> None:
        if not callable(sync_callable):
            raise TypeError(
                "sync_callable must be callable."
            )

        if interval_seconds <= 0:
            raise ValueError(
                "interval_seconds must be greater than zero."
            )

        self.sync_callable = sync_callable
        self.interval_seconds = float(interval_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self._last_result: Any = None
        self._last_error: Exception | None = None

    @property
    def running(self) -> bool:
        """Return whether the background scheduler is running."""
        with self._lock:
            return self._running

    @property
    def last_result(self) -> Any:
        """Return the result from the most recent successful synchronization."""
        with self._lock:
            return self._last_result

    @property
    def last_error(self) -> Exception | None:
        """Return the most recent synchronization error, if any."""
        with self._lock:
            return self._last_error

    def trigger(self) -> Any:
        """Execute one synchronization immediately.

        This is the manual-trigger boundary and is also used internally by
        the scheduled loop.
        """
        try:
            result = self.sync_callable()
        except Exception as exc:
            with self._lock:
                self._last_error = exc
            raise

        with self._lock:
            self._last_result = result
            self._last_error = None

        return result

    def start(self) -> None:
        """Start the background synchronization scheduler."""
        with self._lock:
            if self._running:
                return

            self._stop_event.clear()
            self._running = True

            self._thread = threading.Thread(
                target=self._run,
                name="repository-sync-scheduler",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the background synchronization scheduler."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._stop_event.set()
            thread = self._thread

        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(
                timeout=max(
                    self.interval_seconds,
                    1.0,
                )
            )

        with self._lock:
            self._thread = None

    def _run(self) -> None:
        """Run scheduled synchronization until stopped."""
        try:
            while not self._stop_event.wait(self.interval_seconds):
                try:
                    self.trigger()
                except Exception:
                    # A failed scheduled run must not terminate the
                    # scheduler. The error remains observable through
                    # last_error and the next scheduled run can retry.
                    continue
        finally:
            with self._lock:
                self._running = False
                self._thread = None


class ManualSyncTrigger:
    """Small explicit manual-trigger adapter for repository synchronization."""

    def __init__(self,sync_callable: Callable[[], Any]) -> None:
        if not callable(sync_callable):
            raise TypeError(
                "sync_callable must be callable."
            )

        self.sync_callable = sync_callable

    def trigger(self) -> Any:
        """Execute one synchronization immediately."""
        return self.sync_callable()