"""Safety regression test: invalid draft prototypes must fail before imports/builds."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).parent

class QuarantineTest(unittest.TestCase):
    def test_all_entrypoints_fail_closed(self):
        for p in ROOT.rglob('*'):
            if 'preserved-originals' in p.parts or p.name == Path(__file__).name:
                continue
            if p.suffix == '.py':
                body = ast.parse(p.read_text()).body
                first = body[1] if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) else body[0]
                self.assertIsInstance(first, ast.Raise, str(p))
            elif p.suffix == '.cpp':
                self.assertTrue(p.read_text().startswith('#error "QUARANTINED'), str(p))
            elif p.suffix == '.sh':
                self.assertIn('exit 78', '\n'.join(p.read_text().splitlines()[:5]), str(p))
            elif p.suffix == '.patch':
                self.assertTrue(p.read_text().startswith('QUARANTINED NON-APPLICABLE'), str(p))

if __name__ == '__main__':
    unittest.main()
