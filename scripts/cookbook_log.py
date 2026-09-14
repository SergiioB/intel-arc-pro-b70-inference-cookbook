#!/usr/bin/env python3
"""Structured JSON-lines logging for cookbook scripts, stdlib only.

One JSON object per line: timestamp (ISO-8601 UTC), level, event, run_id,
plus arbitrary key=value kwargs passed through ``extra``. Use
``get_logger()`` once per invocation so every event carries the same run_id.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections.abc import MutableMapping
from datetime import UTC, datetime
from typing import Any

LOGGER_NAME = "cookbook"
# Standard LogRecord attributes that never become event fields.
RESERVED_RECORD_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


def new_run_id() -> str:
    """Short uuid4 hex identifier; generate one per script invocation."""
    return uuid.uuid4().hex[:12]


class JsonLineFormatter(logging.Formatter):
    """Serialize each record as one JSON line with the core fields plus kwargs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "event": record.getMessage(),
            "run_id": str(getattr(record, "run_id", "")),
        }
        for key, value in record.__dict__.items():
            if key not in RESERVED_RECORD_FIELDS and key not in ("run_id", "message"):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class RunLogger(logging.LoggerAdapter[logging.Logger]):
    """LoggerAdapter that merges per-call ``extra`` kwargs with the run fields."""

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        merged: dict[str, Any] = dict(self.extra or {})
        merged.update(kwargs.get("extra") or {})
        kwargs["extra"] = merged
        return msg, kwargs


def get_logger(
    run_id: str | None = None, name: str = LOGGER_NAME
) -> logging.LoggerAdapter[logging.Logger]:
    """Return the structured logger bound to one run_id.

    ``run_id`` is generated when absent so every event of an invocation is
    correlatable; pass an explicit value to continue a run across processes.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonLineFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return RunLogger(logger, {"run_id": run_id or new_run_id()})
