#!/usr/bin/env python3
"""CPU-only D5 regression tests: execute the actual ops.py function AST with stubs.
No torch/Triton imports, extension load, GPU or build. Tests dispatch, not numerics.
Run after applying P5: python3 -m unittest discover -s tests -p test_reconstruct_support.py
Capture status: CPU routing tests only; D3 remains HOLD.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class Tensor:
    dtype = 'fp16'
    device = 'stub'

    def __init__(self, *shape):
        self.shape = shape

    def reshape(self, *shape):
        return self

    def view(self, *shape):
        return self

    def is_contiguous(self):
        return True

    def numel(self):
        result = 1
        for x in self.shape:
            result *= x
        return result

    def __getitem__(self, key):
        return self


class Routing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = Path(__file__).resolve().parents[1] / 'exl3xpu' / 'ops.py'
        tree = ast.parse(src.read_text())
        wanted = {'_supports_reconstruct', 'exl3_linear_impl', 'exl3_linear'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        if {n.name for n in nodes} != wanted:
            raise RuntimeError('P5 routing functions missing')
        cls.code = compile(ast.Module(body=nodes, type_ignores=[]), str(src), 'exec')

    def setup_env(self, enabled):
        calls = []

        class Extension:
            def exl3_supported(self, k, cb):
                return (cb == 2 and k in (4, 6)) or (enabled and
                       ((cb == 2 and k in (2, 3, 5)) or (k in (4, 6) and cb in (0, 1))))

            def exl3_reconstruct(self, *args):
                if not self.exl3_supported(args[-2], args[-1]):
                    raise RuntimeError('unsupported K/cb')
                calls.append('esimd_reconstruct')

            def exl3_gemm_small(self, *args):
                calls.append('esimd_small')

            def linear(self, *args):
                calls.append('cpp_linear')
                return 'cpp'

        extension = Extension()
        tk = SimpleNamespace(had_in=lambda *a: Tensor(1, 129, 128),
                             had_out=lambda *a: None,
                             reconstruct=lambda *a, **kw: calls.append('triton_reconstruct'),
                             gemm_small=lambda *a: calls.append('triton_small'))
        torch = SimpleNamespace(Tensor=Tensor, float16='fp16',
                                empty=lambda shape, **kw: Tensor(*shape),
                                matmul=lambda *a, **kw: None)
        env = {'torch': torch, 'tk': tk, 'SMALL_M_MAX': 128, 'RECON_SLICE_N': 16384,
               'INT8_PREFILL': False, '_backend': 'auto', '_get_esimd': lambda: extension,
               '_weight_buffer': lambda *a: Tensor(128, 128),
               '_exl3_linear_py': lambda *a: calls.append('python_linear') or 'python'}
        exec(self.code, env)
        return env, extension, calls

    @staticmethod
    def args(m, k=3, cb=2):
        return (Tensor(m, 128), Tensor(8, 8, 16 * k), Tensor(1, 128),
                Tensor(128), Tensor(1), [0, 128], k, cb)

    def test_symbol_presence_does_not_enable_k3(self):
        env, extension, calls = self.setup_env(False)
        self.assertTrue(hasattr(extension, 'exl3_reconstruct'))
        self.assertFalse(env['_supports_reconstruct'](extension, 3, 2))
        env['exl3_linear_impl'](*self.args(129))
        self.assertEqual(calls, ['triton_reconstruct'])

    def test_enabled_k3_python_small_and_large(self):
        for m, expected in ((1, 'esimd_small'), (129, 'esimd_reconstruct')):
            env, _, calls = self.setup_env(True)
            env['exl3_linear_impl'](*self.args(m))
            self.assertEqual(calls, [expected])

    def test_public_both_regimes(self):
        for enabled in (False, True):
            for m in (1, 128, 129, 512):
                env, _, calls = self.setup_env(enabled)
                self.assertEqual(env['exl3_linear'](*self.args(m)), 'cpp' if enabled else 'python')
                self.assertEqual(calls, ['cpp_linear' if enabled else 'python_linear'])

    def test_supported_pair_truth_table(self):
        for enabled in (False, True):
            env, extension, _ = self.setup_env(enabled)
            for k in range(1, 8):
                for cb in range(4):
                    expected = (cb == 2 and k in (4, 6)) or (enabled and
                               ((cb == 2 and k in (2, 3, 5)) or (k == 4 and cb in (0, 1))))
                    self.assertEqual(env['_supports_reconstruct'](extension, k, cb), expected)
            self.assertFalse(env['_supports_reconstruct'](False, 3, 2))
            self.assertFalse(env['_supports_reconstruct'](SimpleNamespace(exl3_reconstruct=True), 3, 2))

    def test_k6_other_codebooks_only_small_cpp(self):
        for cb in (0, 1):
            for m in (1, 129):
                env, _, calls = self.setup_env(True)
                self.assertEqual(env['exl3_linear'](*self.args(m, 6, cb)), 'cpp' if m == 1 else 'python')
                if m == 129:
                    env['exl3_linear_impl'](*self.args(m, 6, cb))
                    self.assertEqual(calls, ['python_linear', 'triton_reconstruct'])


if __name__ == '__main__':
    unittest.main()
