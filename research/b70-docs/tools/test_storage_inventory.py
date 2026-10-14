"""resolve(): old note paths map to their current location or to the delete record."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import storage_inventory as si


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "new" / "models" / "kept").mkdir(parents=True)
        (root / "deleted.txt").write_text("/mnt/<disk>/gone\n")
        self.old_root, si.ROOT = si.ROOT, root
        self.layout = {
            "mount_aliases": {"/mnt/<disk>": "/mnt/<disk>"},
            "mount_map": {"/mnt/<disk>": str(root / "new")},
            "path_map": [[str(root / "new" / "kept"), str(root / "new" / "models" / "kept")]],
            "deletions": [{"date": "2026-10-13", "list": "deleted.txt"}],
        }

    def tearDown(self):
        si.ROOT = self.old_root
        self.tmp.cleanup()

    def run_resolve(self, path):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = si.resolve(self.layout, path)
        return code, out.getvalue()

    def test_alias_then_move(self):
        code, out = self.run_resolve("/mnt/<disk>/kept")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), str(Path(self.tmp.name) / "new" / "models" / "kept"))

    def test_child_of_deleted_folder_reports_record(self):
        code, out = self.run_resolve("/mnt/<disk>/gone/sub/file.bin")
        self.assertEqual(code, 1)
        self.assertIn("deleted 2026-10-13", out)

    def test_sibling_with_shared_prefix_is_not_deleted(self):
        code, out = self.run_resolve("/mnt/<disk>/gone-but-different")
        self.assertNotIn("deleted", out)


if __name__ == "__main__":
    unittest.main()
