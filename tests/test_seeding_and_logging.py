"""Smoke tests for seeding + structured logging."""
from __future__ import annotations

import json
import logging

from govmodel.logging_setup import JsonFormatter, configure_logging
from govmodel.seeding import seed_everything


def test_seed_everything_is_idempotent_on_int_returns():
    assert seed_everything(7) == 7
    assert seed_everything(123) == 123


def test_seed_everything_produces_reproducible_random():
    import random
    seed_everything(42)
    a = [random.random() for _ in range(5)]
    seed_everything(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_json_formatter_emits_valid_json_with_extras(caplog):
    logger = logging.getLogger("govmodel.test")
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    record = logger.makeRecord(
        name="govmodel.test", level=logging.INFO, fn="x", lno=1,
        msg="hello", args=(), exc_info=None, extra={"label": "bezwaar", "score": 0.91},
    )
    line = handler.format(record)
    parsed = json.loads(line)
    assert parsed["msg"] == "hello"
    assert parsed["label"] == "bezwaar"
    assert parsed["score"] == 0.91
    assert parsed["level"] == "INFO"


def test_configure_logging_runs_idempotently():
    configure_logging(level="WARNING", json_output=True)
    configure_logging(level="INFO", json_output=False)
    # No exception = pass; the test above proves the formatter itself works.
