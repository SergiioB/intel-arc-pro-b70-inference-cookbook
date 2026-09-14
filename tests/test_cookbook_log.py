import importlib.util
import json
import logging
import pathlib
import re
import sys
import unittest
from datetime import datetime

ROOT = pathlib.Path(__file__).parents[1]
MODULE = ROOT / "scripts" / "cookbook_log.py"

spec = importlib.util.spec_from_file_location("cookbook_log", MODULE)
assert spec is not None and spec.loader is not None
cookbook_log = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cookbook_log
spec.loader.exec_module(cookbook_log)

TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")


class CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class CookbookLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.formatter = cookbook_log.JsonLineFormatter()
        self.handler = CaptureHandler()

    def capture(self, run_id: str | None = None, **extra: object) -> str:
        logger = cookbook_log.get_logger(run_id=run_id)
        self.addCleanup(logger.logger.removeHandler, self.handler)
        logger.logger.addHandler(self.handler)
        logger.info("gate-finish", extra=extra)
        formatted: str = self.formatter.format(self.handler.records[0])
        return formatted

    def test_event_line_is_valid_json_with_core_fields(self) -> None:
        payload = json.loads(self.capture(gate="todos"))
        self.assertEqual(payload["event"], "gate-finish")
        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["gate"], "todos")

    def test_timestamp_is_iso8601_utc(self) -> None:
        payload = json.loads(self.capture())
        self.assertRegex(payload["timestamp"], TIMESTAMP_RE)
        parsed = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
        offset = parsed.utcoffset()
        self.assertIsNotNone(offset)
        assert offset is not None
        self.assertEqual(offset.total_seconds(), 0)

    def test_run_id_is_propagated_when_given(self) -> None:
        adapter = cookbook_log.get_logger(run_id="feedface1234")
        self.assertEqual(adapter.extra["run_id"], "feedface1234")
        payload = json.loads(self.capture(run_id="feedface1234"))
        self.assertEqual(payload["run_id"], "feedface1234")

    def test_run_id_is_generated_when_absent(self) -> None:
        self.assertRegex(cookbook_log.new_run_id(), RUN_ID_RE)
        adapter = cookbook_log.get_logger()
        self.assertRegex(str(adapter.extra["run_id"]), RUN_ID_RE)
        payload = json.loads(self.capture())
        self.assertRegex(payload["run_id"], RUN_ID_RE)

    def test_duration_field_is_carried_on_the_event(self) -> None:
        payload = json.loads(self.capture(gate="unit-tests", duration_s=1.25, ok=True))
        self.assertEqual(payload["duration_s"], 1.25)
        self.assertTrue(payload["ok"])


if __name__ == "__main__":
    unittest.main()
