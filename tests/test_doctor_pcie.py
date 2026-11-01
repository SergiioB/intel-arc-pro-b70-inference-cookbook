"""Unit tests for the PCIe ancestry walk in doctor.py.

The B70 GPU function and the card's own bridge always report 2.5 GT/s x1, live
and maximum. The negotiated host link is only readable at the switch upstream
port and the root port, so the doctor must walk the sysfs ancestry instead of
reading the endpoint. Every case here runs against a fake sysfs tree built in a
temporary directory; no hardware is touched.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_doctor() -> ModuleType:
    path = ROOT / "scripts" / "doctor.py"
    spec = importlib.util.spec_from_file_location("doctor_pcie", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


doctor = load_doctor()

# Path components top-down: root port, the card's switch upstream port, the
# card's internal downstream port, the GPU function.
CHAIN = ("0000:00:03.1", "0000:0a:00.0", "0000:0b:01.0", "0000:0c:00.0")
LINK_STATE = {
    "0000:00:03.1": ("16.0 GT/s PCIe", "8", "16.0 GT/s PCIe", "8"),
    "0000:0a:00.0": ("16.0 GT/s PCIe", "8", "32.0 GT/s PCIe", "16"),
    "0000:0b:01.0": ("2.5 GT/s PCIe", "1", "2.5 GT/s PCIe", "1"),
    "0000:0c:00.0": ("2.5 GT/s PCIe", "1", "2.5 GT/s PCIe", "1"),
}


def write_attrs(device_dir: Path, values: dict[str, str]) -> None:
    device_dir.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        (device_dir / name).write_text(f"{value}\n", encoding="utf-8")


def build_tree(root: Path, states: dict[str, tuple[str, str, str, str]] = LINK_STATE) -> Path:
    """Build a sysfs-shaped tree; returns the symlinked endpoint under the bus root."""
    parent = root / "sys" / "devices" / "pci0000:00"
    for bdf in CHAIN:
        parent = parent / bdf
        speed, width, max_speed, max_width = states.get(bdf, ("", "", "", ""))
        write_attrs(
            parent,
            {
                "current_link_speed": speed,
                "current_link_width": width,
                "max_link_speed": max_speed,
                "max_link_width": max_width,
            },
        )
    bus = root / "sys" / "bus" / "pci" / "devices"
    bus.mkdir(parents=True)
    (bus / CHAIN[-1]).symlink_to(parent)
    return bus / CHAIN[-1]


class PcieChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_chain_lists_every_hop_top_down(self) -> None:
        endpoint = build_tree(self.root)
        hops = doctor.pcie_chain(str(endpoint))
        self.assertEqual([hop["bdf"] for hop in hops], list(CHAIN))
        self.assertEqual(
            [hop["role"] for hop in hops], ["root_port", "bridge", "bridge", "endpoint"]
        )

    def test_internal_hops_are_marked_unreliable(self) -> None:
        endpoint = build_tree(self.root)
        hops = doctor.pcie_chain(str(endpoint))
        self.assertEqual([hop["link_state_reliable"] for hop in hops], [True, True, False, False])

    def test_effective_link_ignores_the_endpoint_misreport(self) -> None:
        endpoint = build_tree(self.root)
        link = doctor.effective_link(doctor.pcie_chain(str(endpoint)))
        self.assertEqual((link["speed"], link["width"]), ("16.0 GT/s PCIe", "8"))
        self.assertEqual(link["source_bdf"], "0000:00:03.1")
        self.assertFalse(link["degraded"])

    def test_effective_link_takes_the_narrowest_reliable_hop(self) -> None:
        states = dict(LINK_STATE)
        states["0000:00:03.1"] = ("16.0 GT/s PCIe", "4", "16.0 GT/s PCIe", "8")
        endpoint = build_tree(self.root, states)
        link = doctor.effective_link(doctor.pcie_chain(str(endpoint)))
        self.assertEqual((link["speed"], link["width"]), ("16.0 GT/s PCIe", "4"))

    def test_root_port_below_maximum_width_is_reported_degraded(self) -> None:
        states = dict(LINK_STATE)
        states["0000:00:03.1"] = ("8.0 GT/s PCIe", "8", "16.0 GT/s PCIe", "16")
        endpoint = build_tree(self.root, states)
        link = doctor.effective_link(doctor.pcie_chain(str(endpoint)))
        self.assertTrue(link["degraded"])
        self.assertEqual(link["reference_bdf"], "0000:00:03.1")

    def test_all_hops_misreporting_falls_back_to_the_endpoint(self) -> None:
        only_endpoint = {"0000:0c:00.0": ("2.5 GT/s PCIe", "1", "2.5 GT/s PCIe", "1")}
        device = self.root / "sys" / "devices" / "pci0000:00" / "0000:0c:00.0"
        write_attrs(
            device,
            {
                "current_link_speed": only_endpoint["0000:0c:00.0"][0],
                "current_link_width": only_endpoint["0000:0c:00.0"][1],
                "max_link_speed": only_endpoint["0000:0c:00.0"][2],
                "max_link_width": only_endpoint["0000:0c:00.0"][3],
            },
        )
        link = doctor.effective_link(doctor.pcie_chain(str(device)))
        self.assertEqual((link["speed"], link["width"]), ("2.5 GT/s PCIe", "1"))
        self.assertEqual(link["source_bdf"], "0000:0c:00.0")

    def test_missing_link_attributes_do_not_raise(self) -> None:
        device = self.root / "sys" / "devices" / "pci0000:00" / "0000:0c:00.0"
        device.mkdir(parents=True)
        link = doctor.effective_link(doctor.pcie_chain(str(device)))
        self.assertEqual(link["speed"], "")
        self.assertFalse(link["degraded"])

    def test_unreadable_intermediate_hop_does_not_hide_the_host_link(self) -> None:
        endpoint = build_tree(self.root)
        # A hop whose attributes cannot be read must not become the "narrowest" hop.
        bridge = endpoint.resolve().parents[1]  # 0000:0a:00.0, the switch upstream port
        self.assertEqual(bridge.name, "0000:0a:00.0")
        for name in ("current_link_speed", "current_link_width"):
            (bridge / name).unlink()
        link = doctor.effective_link(doctor.pcie_chain(str(endpoint)))
        self.assertEqual((link["speed"], link["width"]), ("16.0 GT/s PCIe", "8"))
        self.assertEqual(link["source_bdf"], "0000:00:03.1")


if __name__ == "__main__":
    unittest.main()
