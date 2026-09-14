import importlib.util
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).parents[1]
CHECKER = ROOT / "scripts" / "check-markdown-links.py"

spec = importlib.util.spec_from_file_location("markdown_link_checker", CHECKER)
assert spec is not None and spec.loader is not None
checker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checker
spec.loader.exec_module(checker)


class MarkdownLinkTests(unittest.TestCase):
    def test_tracked_markdown_links_are_valid(self) -> None:
        self.assertEqual(checker.validate_links(checker.repository_markdown()), [])

    def test_missing_relative_target_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = pathlib.Path(directory) / "source.md"
            source.write_text("[missing](not-here.md)\n")
            errors = checker.validate_links([source])
        self.assertEqual(len(errors), 1)
        self.assertIn("missing link target: not-here.md", errors[0])

    def test_external_and_anchor_links_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = pathlib.Path(directory) / "source.md"
            source.write_text("[web](https://example.com) [anchor](#section)\n")
            self.assertEqual(checker.validate_links([source]), [])

    def test_missing_image_target_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = pathlib.Path(directory) / "source.md"
            source.write_text("![missing](missing.png)\n")
            errors = checker.validate_links([source])
        self.assertEqual(len(errors), 1)
        self.assertIn("missing link target: missing.png", errors[0])


if __name__ == "__main__":
    unittest.main()
