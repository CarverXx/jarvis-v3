"""
events — append-only JSONL event stream for the JARVIS v3 TUI dashboard.

Design goals:
  * Zero impact on the voice pipeline when the TUI isn't running (best-effort
    fire-and-forget; any IO error is swallowed with a debug log, never raises
    into the caller).
  * Thread-safe: a background queue + single writer thread serialises writes
    so multiple asyncio tasks / the mic reader thread / the tts executor
    thread can all log without racing on the file handle.
  * Single-line atomic append (\n-terminated) so a parallel tailer can
    line-by-line read partially-flushed files without losing events.
  * Bounded queue (drop-oldest) so a stuck tailer cannot backpressure the
    daemon.

Consumer: `tui/dashboard.py` opens the same file and tails it.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

_logger = logging.getLogger("events")

DEFAULT_PATH = os.environ.get(
    "JARVIS_EVENTS_PATH",
    os.path.expanduser("~/jarvis-v3-logs/events.jsonl"),
)
_MAX_QUEUE = 4096  # events; drops oldest when full


class EventLogger:
    """Singleton JSONL appender. Use `events.log(...)` module-level instead
    of instantiating directly."""

    _instance: "EventLogger | None" = None
    _lock = threading.Lock()

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue[str] = queue.Queue(maxsize=_MAX_QUEUE)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._writer_loop, name="events-writer", daemon=True,
        )
        self._thread.start()

    @classmethod
    def get(cls) -> "EventLogger":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _writer_loop(self):
        f = None
        try:
            f = open(self.path, "a", encoding="utf-8", buffering=1)
        except Exception:
            _logger.exception("cannot open events file %s", self.path)
            return
        try:
            while not self._stop.is_set():
                try:
                    line = self._q.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    f.write(line + "\n")
                except Exception:
                    _logger.debug("events write failed", exc_info=True)
        finally:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass

    def log(self, event_type: str, **data: Any):
        """Enqueue one event. Never raises.

        Common convention: keep value types primitive (str/int/float/bool);
        binary or ndarray should be summarised (mean/peak/shape) before
        logging to keep JSONL small.
        """
        try:
            payload = {
                "t": time.time(),
                "type": event_type,
                **data,
            }
            line = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            _logger.debug("events serialise failed", exc_info=True)
            return
        try:
            self._q.put_nowait(line)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(line)
            except Exception:
                pass

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


def log(event_type: str, **data: Any) -> None:
    """Module-level convenience: route to the singleton."""
    try:
        EventLogger.get().log(event_type, **data)
    except Exception:
        _logger.debug("events.log failed", exc_info=True)
