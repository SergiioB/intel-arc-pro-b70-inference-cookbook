#!/usr/bin/env python3
"""Generate review-only unified patches from the pinned source without changing it.
Stdlib + read-only git identity checks. No compiler, torch, GPU or Docker calls.
Outputs stay beside this authoring script. Existing draft patches are not overwritten.
"""
from pathlib import Path
import difflib
import subprocess

ROOT = Path(__file__).resolve().parent
SRC = Path('/tmp/b70-k3-exl3-source-20260929')
BASE = '15ded2f3add148c4db3c900cba7de878238f53bc'
assert subprocess.check_output(['git', '-C', str(SRC), 'rev-parse', 'HEAD'], text=True).strip() == BASE
assert not subprocess.check_output(['git', '-C', str(SRC), 'status', '--porcelain'], text=True).strip()
state = {}


def get(path):
    if path not in state:
        state[path] = (SRC / path).read_text()
    return state[path]


def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def patch(name, subject, notes, changes):
    out = (f'Subject: {subject}\nBase-commit: {BASE}\n'
           'Status: local-only draft; PROVISIONAL — NOT FOR PUBLIC HEADLINE\n'
           + notes + '\n\n')
    for path, new in changes.items():
        old = get(path) if (SRC / path).exists() or path in state else ''
        out += f'diff --git a/{path} b/{path}\n'
        if not old:
            out += 'new file mode 100644\n'
        out += ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                          fromfile=f'a/{path}' if old else '/dev/null',
                                          tofile=f'b/{path}', n=8))
        state[path] = new
    with (ROOT / name).open('x') as f:
        f.write(out)


build = get('scripts/build_ext.sh')
build = replace(build, 'T=$(python3', '''# Optional codebooks are opt-in; preserve the K4/K6-only default.
case "${EXL3_ALL_CODEBOOKS:-0}" in
  0) ;;
  1) EXL3_FLAGS="${EXL3_FLAGS:-} -DEXL3_ALL_CODEBOOKS" ;;
  *) printf '%s\\n' 'EXL3_ALL_CODEBOOKS must be 0 or 1' >&2; exit 2 ;;
esac
T=$(python3''')
build = replace(build, ' $DNNL_LIBS 2>&1 | grep -E "error" || true', ' $DNNL_LIBS')
docker = replace(get('docker/Dockerfile'), 'RUN bash scripts/build_ext.sh',
                 'ARG EXL3_ALL_CODEBOOKS=0\nARG EXL3_FLAGS=""\nRUN EXL3_ALL_CODEBOOKS="$EXL3_ALL_CODEBOOKS" EXL3_FLAGS="$EXL3_FLAGS" bash scripts/build_ext.sh')
patch('0001-P1-build-enablement.patch', '[P1] Parameterize optional codebooks; propagate compiler failure',
      'Anchors: scripts/build_ext.sh:14-19; docker/Dockerfile:11.\n'
      'D2: remove the filtering/swallowing pipeline altogether; set -e now propagates\n'
      'compiler failure even if an older output .so exists. Retain full diagnostics.\n'
      'EXL3_ALL_CODEBOOKS=1 appends the define through EXL3_FLAGS; default remains 0.\n'
      'Capture-safety: build-only change; no runtime allocation/sync added. D3 HOLD\n'
      'remains; this is not authorization to build. Apply P1b before ALL_CODEBOOKS.',
      {'scripts/build_ext.sh': build, 'docker/Dockerfile': docker})

ops = get('csrc/exl3_ops.sycl')
ops = replace(ops, '// K=4 M=1 -> 8 (longer contiguous reads), K=4 M=2..4 -> 4, K=6 -> 4 (two 32-lane tile pairs), M=8 -> 2.\nconstexpr int nt_of(int K, int MR) { return MR == 8 ? 2 : (K <= 4 ? (MR == 1 ? 8 : 4) : 4); }',
'''// K=4 M=1 -> 8, K=4 M=2..4 -> 4, K=6 -> 4; M=8 -> 2 except K=3/5.
// Geo<3/5>::G=8 => TP=4: every instantiated MR needs NT % TP == 0.
constexpr int nt_of(int K, int MR) {
    return MR == 8 ? ((K == 3 || K == 5) ? 4 : 2) : (K <= 4 ? (MR == 1 ? 8 : 4) : 4);
}''')
patch('0002-P1b-strip-constraint.patch', '[P1b] Repair K3 and K5 MR8 tile pairing',
      'Anchor: csrc/exl3_ops.sycl:181-195; exl3_esimd.h:464-467.\n'
      'D1 derivation: G=8 -> TP=32/G=4 -> NT%TP must be zero. NT=2 at MR8\n'
      'fails for BOTH K3 and K5, unconditionally instantiated by dispatch_mr.\n'
      'Choose minimum legal NT=4; MR itself stays 8. This is a template repair,\n'
      'not empirical tuning and not a measured speed improvement.\n'
      'Capture-safety: constexpr geometry only; no new allocation/sync. Whole-op D3 HOLD.',
      {'csrc/exl3_ops.sycl': ops})

ops = get('csrc/exl3_ops.sycl')
ops = replace(ops, 'constexpr int nt_of(int K, int MR) {\n    return MR == 8',
'''// K3 starts from the repaired pattern, not a claimed measured optimum.
// Restrict the sweep to 64/128 columns: valid pairing and no 128-column shard crossing.
#ifndef EXL3_K3_NT_MR1
#define EXL3_K3_NT_MR1 8
#endif
#ifndef EXL3_K3_NT_MR2
#define EXL3_K3_NT_MR2 4
#endif
#ifndef EXL3_K3_NT_MR4
#define EXL3_K3_NT_MR4 4
#endif
#ifndef EXL3_K3_NT_MR8
#define EXL3_K3_NT_MR8 4
#endif
static_assert(EXL3_K3_NT_MR1 == 4 || EXL3_K3_NT_MR1 == 8, "K3 MR1 NT must be 4 or 8");
static_assert(EXL3_K3_NT_MR2 == 4 || EXL3_K3_NT_MR2 == 8, "K3 MR2 NT must be 4 or 8");
static_assert(EXL3_K3_NT_MR4 == 4 || EXL3_K3_NT_MR4 == 8, "K3 MR4 NT must be 4 or 8");
static_assert(EXL3_K3_NT_MR8 == 4 || EXL3_K3_NT_MR8 == 8, "K3 MR8 NT must be 4 or 8");
constexpr int nt_of(int K, int MR) {
    if (K == 3) return MR == 1 ? EXL3_K3_NT_MR1 : MR == 2 ? EXL3_K3_NT_MR2
                           : MR == 4 ? EXL3_K3_NT_MR4 : EXL3_K3_NT_MR8;
    return MR == 8''')
patch('0003-P2-k3-strip-selection.patch', '[P2] Name K3 strip choices independently of K4 tuning',
      'Anchor: csrc/exl3_ops.sycl:181-184, after P1b. Defaults [8,4,4,4].\n'
      'Sweep geometry, legality, paper GRF budget and gates: P2-SWEEP-SPEC.txt.\n'
      'Trellis decode equality is zero-tolerance; dense output uses P3 frozen tolerance.\n'
      'Capture-safety: compile-time constants only; no new allocation/sync. D3 HOLD.',
      {'csrc/exl3_ops.sycl': ops, 'tests/k3_sweep_spec.txt': (ROOT / 'P2-SWEEP-SPEC.txt').read_text()})

ops = get('csrc/exl3_ops.sycl')
ops = replace(ops, '    TORCH_CHECK(ok);\n    (void)S;',
'''#ifdef EXL3_ALL_CODEBOOKS
    if (K == 3 && cb == 2) ok = use_vec ? dispatch_mr<3, 2>(MR, xh_blocked, trellis, shard_of_nb, part, M, Kdim, N, P, rps, q)
                                        : dispatch_dpas<3, 2>(MB, xh_blocked, trellis, shard_of_nb, part, M, Kdim, N, P, rps, q);
#endif
    TORCH_CHECK(ok);
    (void)S;''')
patch('0004-P3-real-geometry-correctness.patch', '[P3] Add real-checkpoint K3 numerical and replay gate',
      'Anchors: diagnostic raw dispatch csrc/exl3_ops.sycl:554-591; reconstruct\n'
      'reference :545; production small/large-M :656-748. Adds K3 raw diagnostics\n'
      'only so the otherwise unreachable MR8 specialization can be tested.\n'
      'New owned test recensuses real width48 x400, never substitutes K4 payload.\n'
      'Capture-safety: no production change; test sync outside capture is intentional.\n'
      'Replay correctness is not allocator certification: D3 HOLD. GPU mode unrun.',
      {'csrc/exl3_ops.sycl': ops,
       'tests/test_k3_real_geometry.py': (ROOT / 'test_k3_real_geometry.py').read_text(),
       'tests/k3_runner_spec.txt': (ROOT / 'P3-RUNNER-SPEC.txt').read_text()})

patch('0005-P4-matched-ab-spec.patch', '[P4] Specify matched token-rate, prefill and quality gates',
      'New-file specification anchored to pinned dispatch :661 (int8 K4/K6 only).\n'
      'No performance result; inherited rates are explicitly diagnostic/reference.\n'
      'Capture-safety: specification only, no runtime changes. D3 HOLD.',
      {'tests/k3_ab_spec.txt': (ROOT / 'P4-AB-SPEC.txt').read_text()})

py = get('exl3xpu/ops.py')
py = replace(py, '\n\n_wbuf: dict = {}', '''

def _supports_reconstruct(esimd, K: int, cb: int) -> bool:
    # Symbol presence is not compiled K support. Match the narrower reconstruction
    # dispatch: unlike GEMM, K6/cb0 and K6/cb1 are not implemented here.
    return bool(esimd and hasattr(esimd, "exl3_reconstruct")
                and hasattr(esimd, "exl3_supported") and esimd.exl3_supported(K, cb)
                and ((cb == 2 and K in (2, 3, 4, 5, 6)) or (K == 4 and cb in (0, 1))))


_wbuf: dict = {}''')
py = replace(py, '            if esimd and hasattr(esimd, "exl3_reconstruct"):',
                 '            if _supports_reconstruct(esimd, K, cb):')
py = replace(py, '    if esimd and hasattr(esimd, "linear") and esimd.exl3_supported(K, cb):',
'''    if (esimd and hasattr(esimd, "linear") and esimd.exl3_supported(K, cb)
            and (x.numel() // x.shape[-1] <= SMALL_M_MAX or _supports_reconstruct(esimd, K, cb))):''')
patch('0006-P5-k-aware-reconstruction.patch', '[P5] Gate reconstruction on compiled K/codebook support',
      'D5 anchors: exl3xpu/ops.py:100-103 and public routing :130-137.\n'
      'D5 derivation: exported exl3_reconstruct exists in K4/K6-only builds but\n'
      'K3 dispatch is compiled out; hasattr passes and engine-init large-M throws.\n'
      'Require compiled support plus actual reconstruction pair set; unsupported\n'
      'K3 falls back to Triton, enabled K3 reaches C++ small AND large M.\n'
      'Also keep K6/cb0/1 out of the large-M C++ route (GEMM advertises them but\n'
      'reconstruct does not). Integer metadata check only, no tensor scalar reads.\n'
      'Capture-safety: no new allocation or device sync. Existing fallback/private\n'
      'scratch behavior is not certified; whole-op D3 HOLD. Test added is CPU-only.',
      {'exl3xpu/ops.py': py,
       'tests/test_reconstruct_support.py': (ROOT / 'test_reconstruct_support.py').read_text()})
print('Authored six numbered patches; pinned clone unchanged.')
