"""Unit tests for the stdlib-only /proc/meminfo parsing in host-preflight."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_MEMINFO = (
    "MemTotal:       16384000 kB\n"
    "MemFree:         1024000 kB\n"
    "MemAvailable:    6144000 kB\n"
    "Buffers:          204800 kB\n"
    "Cached:          4096000 kB\n"
    "SwapTotal:      34816000 kB\n"
    "SwapFree:       33816576 kB\n"
)


def load_preflight() -> ModuleType:
    path = ROOT / "scripts" / "setup" / "host-preflight.py"
    spec = importlib.util.spec_from_file_location("host_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preflight = load_preflight()


class MeminfoParsingTests(unittest.TestCase):
    def test_parse_meminfo_converts_kb_to_bytes(self) -> None:
        fields = preflight.parse_meminfo(SAMPLE_MEMINFO)
        self.assertEqual(fields["MemTotal"], 16384000 * 1024)
        self.assertEqual(fields["MemAvailable"], 6144000 * 1024)
        self.assertEqual(fields["SwapTotal"], 34816000 * 1024)

    def test_parse_meminfo_ignores_malformed_lines(self) -> None:
        fields = preflight.parse_meminfo("GarbageLine\nMemTotal: 1024 kB\nEmpty:\n")
        self.assertEqual(fields, {"MemTotal": 1024 * 1024})

    def test_virtual_memory_matches_psutil_semantics(self) -> None:
        vm = preflight.virtual_memory(preflight.parse_meminfo(SAMPLE_MEMINFO))
        self.assertEqual(vm.total, 16384000 * 1024)
        self.assertEqual(vm.available, 6144000 * 1024)
        self.assertEqual(vm.used, (16384000 - 6144000) * 1024)
        self.assertAlmostEqual(vm.percent, 10240000 / 16384000 * 100)

    def test_virtual_memory_defaults_missing_fields_to_zero(self) -> None:
        vm = preflight.virtual_memory({"MemTotal": 1024 * 1024})
        self.assertEqual(vm.total, 1024 * 1024)
        self.assertEqual(vm.available, 0)
        self.assertEqual(vm.used, 1024 * 1024)
        self.assertEqual(vm.percent, 100.0)

    def test_virtual_memory_handles_empty_fields_without_dividing_by_zero(self) -> None:
        vm = preflight.virtual_memory({})
        self.assertEqual(vm.total, 0)
        self.assertEqual(vm.available, 0)
        self.assertEqual(vm.used, 0)
        self.assertEqual(vm.percent, 0.0)

    def test_swap_total_reads_swap_entry(self) -> None:
        fields = preflight.parse_meminfo(SAMPLE_MEMINFO)
        self.assertEqual(preflight.swap_total(fields), 34816000 * 1024)
        self.assertEqual(preflight.swap_total({}), 0)


if __name__ == "__main__":
    unittest.main()
