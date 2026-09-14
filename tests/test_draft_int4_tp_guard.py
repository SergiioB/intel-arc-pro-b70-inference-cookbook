import pathlib
import unittest

ROOT = pathlib.Path(__file__).parents[1]
PATCHES = [
    ROOT / "patches" / "patch_draft_mtp_int4.py",
    ROOT / "patches" / "patch_draft_lmhead_int4.py",
]
VENDORED = [
    ROOT / "windows" / bundle / "patches" / patch.name
    for bundle in ("Qwen38-Docker-Standalone", "Qwen38-WSLC-Standalone")
    for patch in PATCHES
]


class DraftInt4TpGuardTests(unittest.TestCase):
    def test_each_patch_reads_runtime_tp_and_fails_closed(self) -> None:
        for path in PATCHES:
            text = path.read_text()
            with self.subTest(path=path.name):
                self.assertIn('"parallel_config", None', text)
                self.assertIn('"tensor_parallel_size", None', text)
                self.assertIn("TP unknown; skipping", text)
                self.assertIn("fail-closed, issue #9", text)
                self.assertIn("TP>1 (tp=%d) detected; skipping", text)

    def test_each_patch_checks_guard_before_quantization(self) -> None:
        for path in PATCHES:
            text = path.read_text()
            with self.subTest(path=path.name):
                guard = text.index('tensor_parallel_size", None)')
                quantize = text.index("quantize_", guard)
                self.assertLess(guard, quantize)

    def test_lmhead_hook_does_not_reenter_after_tp_block(self) -> None:
        text = PATCHES[1].read_text()
        self.assertIn(
            'and not getattr(self, \\"_b70_lmhead_int4_tp_blocked\\", False)',
            text,
        )

    def test_vendored_patches_match_canonical(self) -> None:
        for vendored in VENDORED:
            canonical = ROOT / "patches" / vendored.name
            with self.subTest(path=vendored.relative_to(ROOT)):
                self.assertEqual(vendored.read_bytes(), canonical.read_bytes())


if __name__ == "__main__":
    unittest.main()
