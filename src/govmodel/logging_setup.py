"""Structured logging for govmodel.

stdlib `logging` formatted as JSON-lines so dashboards / log aggregators
(Loki, ELK, Cloud Logging) can ingest without a parser. Use:

    from govmodel.logging_setup import configure_logging
    configure_logging(level="INFO")
    logger = logging.getLogger("govmodel.app")
    logger.info("classified", extra={"label": "bezwaar", "score": 0.91})

Anything in `extra=` lands as top-level keys in the JSON line.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k in _RESERVED or k.startswith("_"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except TypeError:
                payload[k] = repr(v)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | None = None, *, json_output: bool | None = None) -> None:
    """Idempotent — safe to call multiple times.

    `level` defaults to `GOVMODEL_LOG_LEVEL` env var or `INFO`.
    `json_output` defaults to `GOVMODEL_LOG_JSON=1` (default 1).
    """
    if level is None:
        level = os.environ.get("GOVMODEL_LOG_LEVEL", "INFO")
    if json_output is None:
        json_output = os.environ.get("GOVMODEL_LOG_JSON", "1") == "1"

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
    root.addHandler(handler)
    root.setLevel(level)
