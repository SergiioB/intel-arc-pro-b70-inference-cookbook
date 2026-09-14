import importlib.util
import json
import pathlib
import sys
import unittest
from typing import Any

ROOT = pathlib.Path(__file__).parents[1]
MAP_PATH = ROOT / "data" / "system-map.v1.json"
VALIDATOR = ROOT / "scripts" / "validate-system-map.py"

spec = importlib.util.spec_from_file_location("system_map_validator", VALIDATOR)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)


class SystemMapTests(unittest.TestCase):
    system_map: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.system_map = json.loads(MAP_PATH.read_text())

    def test_live_system_map_is_valid(self) -> None:
        self.assertEqual(validator.validate_system_map(self.system_map), [])

    def test_missing_authority_fails_closed(self) -> None:
        broken = json.loads(json.dumps(self.system_map))
        broken["authorities"]["missing"] = "docs/DOES-NOT-EXIST.md"
        errors = validator.validate_system_map(broken)
        self.assertIn("missing referenced path: docs/DOES-NOT-EXIST.md", errors)

    def test_validation_has_one_stable_entry_point(self) -> None:
        self.assertEqual(self.system_map["validation"], ["python3 scripts/check-repo.py"])
        broken = json.loads(json.dumps(self.system_map))
        broken["validation"].append("python3 scripts/validate-system-map.py")
        self.assertIn(
            "validation must contain only the stable check-repo entry point",
            validator.validate_system_map(broken),
        )

    def test_control_graph_has_all_intents_and_layers(self) -> None:
        self.assertEqual(set(self.system_map["intents"]), validator.REQUIRED_INTENTS)
        self.assertEqual(
            [layer["id"] for layer in self.system_map["layers"]], ["L0", "L1", "L2", "L3", "L4"]
        )

    def test_every_concrete_route_path_exists(self) -> None:
        for route in self.system_map["intents"].values():
            for relative in [route["start"], *route["then"]]:
                if "<" not in relative:
                    self.assertTrue((ROOT / relative).exists(), relative)


if __name__ == "__main__":
    unittest.main()
