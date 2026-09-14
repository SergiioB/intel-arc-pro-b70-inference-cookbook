#!/usr/bin/env python3
"""Port del PR abierto vllm-project/vllm#53919 (runner V1) + diagnostica V2.

CORRUPCION SILENCIOSA que arregla (modelos hibridos Mamba/GDN + spec-decode +
prefijado, que fuerza mamba_cache_mode="align", + async scheduling):

  El paso N copia num_accepted_tokens D2H SIN bloqueo al buffer fijado
  input_batch.num_accepted_tokens_cpu_tensor en el ORDEN DE FILAS del paso N.
  El _update_states del paso N+1 permuta ese mismo buffer en host
  (add_request / swap_states / condense) SIN esperar el evento;
  _prepare_inputs sincroniza despues y entonces aplica el gather
  prev_positions. Con presion de host la copia cae antes, condense() ya mapeo
  los valores en orden actual y el gather los permuta UNA SEGUNDA VEZ -> una
  request recibe el contador de OTRA request, que se convierte en su
  desplazamiento de conv y su selector de snapshot GDN, y se reescribe al
  cache SSM (no se autocorrige). Nada lanza excepcion.

Ambas mitades son necesarias: upstream midio que el wait solo es PEOR que el
baseline (garantiza el aterrizaje temprano que dispara la doble permutacion).
  1. esperar num_accepted_tokens_event al principio de los movimientos de
     filas de _update_states;
  2. eliminar el gather prev_positions de _prepare_inputs (redundante una vez
     que la copia ya cayo antes).

QUE RUNNER ESTE VIVO EN 0.28.1rc1 (IMPORTANTE)
----------------------------------------------
Esta imagen corre Model Runner V2 (el server loguea "Using V2 Model Runner"), y
en V2 el fichero que parcheamos (v1/worker/gpu_model_runner.py) NO se ejecuta.
El script SIGUE parcheandolo -- es la red de seguridad si caemos a V1
(VLLM_USE_V2_MODEL_RUNNER=0, o la imagen legacy 0.26.1) -- pero IMPRIME una
diagnostica del runner con evidencia de grep re-buscada en el arbol real (los
numeros de linea no estan hardcodeados, asi que un nightly futuro que mueva el
patron al runner V2 se detecta en vez de callarse):

  * el selector vive en config/vllm.py: propiedad use_v2_model_runner,
    gobernada por envs.VLLM_USE_V2_MODEL_RUNNER (None = auto: V2 salvo que
    falte Triton o haya features no soportadas);
  * los tres simbolos de la carrera (num_accepted_tokens_cpu, prev_positions,
    num_accepted_tokens_event) aparecen SOLO en los ficheros V1
    (v1/worker/gpu_model_runner.py, v1/worker/gpu_input_batch.py y el camino
    host preprocess_mamba / collect_mamba_copy_meta de v1/worker/mamba_utils.py);
  * en el arbol V2 (v1/worker/gpu/model_runner.py, v1/worker/gpu/input_batch.py,
    v1/worker/gpu/model_states/mamba_hybrid.py) ese conteo es CERO: los
    contadores de aceptados son GPU-residents (num_accepted_tokens_gpu) y se
    reordenan con gathers/scatters por idx_mapping EN DISPOSITIVO
    (_scatter_num_accepted_kernel / _fill_num_accepted_kernel en
    postprocess_state; gather en prepare_attn). Sin buffer fijado en host no
    hay D2H que llegue tarde ni doble permutacion posible: la carrera NO
    EXISTE POR DISENO en V2 y no hace falta analogo de este parche.

CODIGOS DE SALIDA (fail-closed donde importa)
---------------------------------------------
  * runner V1 vivo (VLLM_USE_V2_MODEL_RUNNER=0) y anclas no aplicadas -> rc 1;
  * runner V2 vivo y V1 parcheado (o ya aplicado)                    -> rc 0 +
    aviso explcito de "fichero muerto por V2" con la evidencia;
  * runner V2 vivo y V1 SIN anclas (deriva upstream en un fichero que no se
    ejecuta) -> rc 0 con WARNING, para no tumbar un arranque sano;
  * B70_FIX_ACCEPT_SYNC_REQUIRE=1 -> solo rc 0 con el parche realmente
    aplicado, sea cual sea el runner (para CI estricta / verify_patches.sh).

Se desactiva para A/B con B70_FIX_ACCEPT_SYNC=0 (deja vLLM intacto).
B70_VLLM_ROOTS="dir1:dir2" fuerza la lista de arboles (lo usa verify_patches.sh).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REL = Path("v1/worker/gpu_model_runner.py")

# Marcador generico (lo buscan verify_patches.sh y los lanzadores) + marcadores
# propios por hunk, para no dar por aplicado un hunk cuando solo esta el otro.
MARKER = "B70_ACCEPT_SYNC"
MARK_WAIT = "B70_ACCEPT_SYNC_WAIT"
MARK_GATHER = "B70_ACCEPT_SYNC_GATHER"

# --- hunk 1: el wait, justo antes del primer movimiento de filas -----------
HUNK_UPDATE_STATES_OLD = (
    "                if orig != req_state.prev_num_draft_len:\n"
    "                    req_state.prev_num_draft_len = orig\n"
    "\n"
    "        # Add the new or resumed requests to the persistent batch."
)

HUNK_UPDATE_STATES_NEW = (
    "                if orig != req_state.prev_num_draft_len:\n"
    "                    req_state.prev_num_draft_len = orig\n"
    "\n"
    f"        # {MARK_WAIT} (port of vllm#53919): the previous step's D2H\n"
    "        # fills num_accepted_tokens_cpu_tensor in THAT step's row order,\n"
    "        # and the row moves below permute the same pinned buffer. Wait\n"
    "        # here so the copy always lands first.\n"
    "        if self.num_accepted_tokens_event is not None:\n"
    "            self.num_accepted_tokens_event.synchronize()\n"
    "\n"
    "        # Add the new or resumed requests to the persistent batch."
)

# --- hunk 2: fuera el gather (evita la segunda permutacion) ----------------
HUNK_GATHER_OLD = (
    "            # Async mode: condense() reordered indices, use prev_positions mapping\n"
    "            if self.use_async_scheduling and prev_req_id_to_index:\n"
    "                prev_idx = self.prev_positions.np[:num_reqs]\n"
    "                new_mask = prev_idx < 0\n"
    "                self.num_accepted_tokens.np[:num_reqs] = (\n"
    "                    self.input_batch.num_accepted_tokens_cpu[\n"
    "                        np.where(new_mask, 0, prev_idx)\n"
    "                    ]\n"
    "                )\n"
    "                self.num_accepted_tokens.np[:num_reqs][new_mask] = 1\n"
    "                self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (\n"
    "                    self.num_accepted_tokens.np[:num_reqs]\n"
    "                )\n"
    "            else:\n"
    "                # Non-async mode: use values directly\n"
    "                self.num_accepted_tokens.np[:num_reqs] = (\n"
    "                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]\n"
    "                )"
)

HUNK_GATHER_NEW = (
    f"            # {MARK_GATHER} (port of vllm#53919): _update_states now\n"
    "            # waits for the copy before moving rows, so the counts are\n"
    "            # already in current order; gathering again would permute\n"
    "            # them twice.\n"
    "            self.num_accepted_tokens.np[:num_reqs] = (\n"
    "                self.input_batch.num_accepted_tokens_cpu[:num_reqs]\n"
    "            )"
)

# (nombre, marcador propio, ancla, reemplazo, apariciones esperadas)
HUNKS: tuple[tuple[str, str, str, str, int], ...] = (
    ("hunk1 wait en _update_states", MARK_WAIT,
     HUNK_UPDATE_STATES_OLD, HUNK_UPDATE_STATES_NEW, 1),
    ("hunk2 gather en _prepare_inputs", MARK_GATHER,
     HUNK_GATHER_OLD, HUNK_GATHER_NEW, 1),
)

# Simbolos que definen la carrera. Si alguno aparece en el arbol V2, el parche
# deja de ser un no-op ahi y habria que escribir su analogo.
RACE_SYMBOLS = ("num_accepted_tokens_cpu", "prev_positions",
                "num_accepted_tokens_event")

# Ficheros del runner V2 (el que corre en esta imagen).
V2_FILES = (
    Path("v1/worker/gpu/model_runner.py"),
    Path("v1/worker/gpu/input_batch.py"),
    Path("v1/worker/gpu/model_states/mamba_hybrid.py"),
    Path("v1/worker/gpu/model_states/default.py"),
)

# Donde vive el patron hoy (referencia del aviso).
V1_FILES = (
    Path("v1/worker/gpu_model_runner.py"),
    Path("v1/worker/gpu_input_batch.py"),
    Path("v1/worker/mamba_utils.py"),
)


def vllm_roots() -> list[Path]:
    override = os.environ.get("B70_VLLM_ROOTS", "").strip()
    if override:
        return [Path(p) for p in override.split(os.pathsep) if p.strip()]
    roots: list[Path] = []
    try:
        out = subprocess.run(
            [sys.executable, "-c",
             "import vllm, os; print(os.path.dirname(vllm.__file__))"],
            capture_output=True, text=True, timeout=600,
        ).stdout.strip()
        if out:
            roots.append(Path(out))
    except Exception:
        pass
    for cand in (
        Path("/workspace/vllm/vllm"),
        Path("/opt/venv/lib/python3.12/site-packages/vllm"),
        Path("/opt/vllm/lib/python3.12/site-packages/vllm"),
    ):
        if cand not in roots:
            roots.append(cand)
    return roots


def hit_lines(root: Path, rel: Path, needle: str) -> list[int]:
    """Lineas (1-based) de root/rel que contienen `needle` ([] si no existe)."""
    try:
        text = (root / rel).read_text()
    except OSError:
        return []
    return [i + 1 for i, line in enumerate(text.split("\n")) if needle in line]


def host_path_callers(roots: list[Path]) -> list[str]:
    """Rutas RELATIVAS de ficheros bajo v1/worker/ que invocan el camino HOST de
    contadores (preprocess_mamba / postprocess_mamba_align_gpu /
    collect_mamba_copy_meta, que leen y escriben num_accepted_tokens_cpu).
    Recorremos TODOS los arboles y devolvemos rutas relativas: si el arbol vivo
    y el espejo difieren, la lista lo refleja. Si algun fichero del arbol V2
    aparece aqui, la carrera SI alcanzaria a V2 y habria que parchearlo tambien.
    """
    found: set[str] = set()
    for root in roots:
        base = root / "v1" / "worker"
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.py")):
            if f.name == "mamba_utils.py":
                continue  # es la definicion, no un caller
            try:
                text = f.read_text()
            except OSError:
                continue
            for sym in ("preprocess_mamba(", "postprocess_mamba_align_gpu(",
                        "collect_mamba_copy_meta("):
                if sym in text:
                    found.add(f"{f.relative_to(root)} llama a {sym}")
    return sorted(found)


def runner_report(roots: list[Path]) -> tuple[bool | None, list[str]]:
    """(runner V2 vivo?, lineas de informe). None = no decidible estaticamente."""
    report: list[str] = []
    root = next((r for r in roots if (r / "config" / "vllm.py").is_file()),
                roots[0] if roots else Path("/nonexistent"))
    env_forced = os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "").strip()

    # 1) el selector
    cfg = root / "config" / "vllm.py"
    sel = hit_lines(root, Path("config/vllm.py"), "def use_v2_model_runner")
    if sel:
        report.append(f"  selector: {cfg}:{sel[0]} define la propiedad "
                      f"use_v2_model_runner")
        for ln in hit_lines(root, Path("config/vllm.py"),
                            "VLLM_USE_V2_MODEL_RUNNER")[:2]:
            report.append(f"           {cfg}:{ln} la goberna via "
                          f"envs.VLLM_USE_V2_MODEL_RUNNER")
    else:
        report.append("  selector: NO existe use_v2_model_runner en "
                      "config/vllm.py -> imagen sin runner V2")

    # 2) simbolos de la carrera en el arbol V2
    v2_hits: list[str] = []
    for sym in RACE_SYMBOLS:
        for rel in V2_FILES:
            for ln in hit_lines(root, rel, sym):
                v2_hits.append(f"{root / rel}:{ln} contiene {sym}")
    v2_present = any((root / rel).is_file() for rel in V2_FILES)
    if v2_hits:
        report.append("  V2: ATENCION, simbolos de la carrera en el arbol V2 "
                      "-> hace falta un parche analogo en V2:")
        report.extend(f"           {w}" for w in v2_hits[:8])
    else:
        report.append("  V2: 0 apariciones de (" + ", ".join(RACE_SYMBOLS) +
                      ") en el arbol V2 revisado -> la doble permutacion no "
                      "existe ahi")

    # 3) donde vive el patron (evidencia positiva del lado V1)
    for sym in RACE_SYMBOLS:
        where = [(rel, hit_lines(root, rel, sym)) for rel in V1_FILES]
        where = [(rel, lns) for rel, lns in where if lns]
        if where:
            report.append(f"  patron {sym}: " + "; ".join(
                f"{root / rel} ({len(lns)} lineas, primera l.{lns[0]})"
                for rel, lns in where))

    gpu = [f"{root / rel}:{ln}" for rel in V2_FILES
           for ln in hit_lines(root, rel, "num_accepted_tokens_gpu")[:2]]
    if gpu:
        report.append("  V2: contadores GPU-residents en " + "; ".join(gpu[:6]))

    # 3b) quien invoca el camino host de contadores (si V2 lo llamara, la
    # carrera tambien lo alcanzaria)
    callers = host_path_callers(roots)
    if callers:
        v2_callers = [c for c in callers if c.startswith(f"v1{os.sep}worker{os.sep}gpu{os.sep}")
                      or c.startswith("v1/worker/gpu/")]
        report.append("  camino host ('preprocess_mamba'/'postprocess_mamba_align_gpu'"
                      f"/'collect_mamba_copy_meta') llamado desde {len(callers)} "
                      f"fichero(s):")
        for c in callers[:6]:
            report.append(f"           {c}")
        if v2_callers:
            report.append("  ATENCION: el arbol V2 llama al camino host -> la "
                          "carrera alcanza a V2 y hace falta un parche analogo")
    else:
        report.append("  camino host de contadores: 0 callers fuera de "
                      "mamba_utils.py -> inalcanzable desde V2")

    # 4) decision
    if env_forced:
        vivo = env_forced.lower() not in ("0", "false", "no", "off")
        report.append(f"  decision: VLLM_USE_V2_MODEL_RUNNER={env_forced} "
                      f"forzado -> runner {'V2' if vivo else 'V1'} VIVO")
        return vivo, report
    if not sel or not v2_present:
        report.append("  decision: no decidible estaticamente -> se parchea V1 "
                      "y se asume que podria estar vivo")
        return None, report
    if v2_hits:
        # El patron migro a V2: hay que parchearlo ahi (este script lo avisa,
        # el analogo se escribe aparte). V1 sigue vivo como fallback posible.
        report.append("  decision: runner V2 vivo PERO con el patron de la "
                      "carrera -> NO tratar V1 como muerto; revisar los "
                      "simbolos V2 arriba")
        return True, report
    # El mero selector NO decide: los builds pre-0.28 (0.26.1/0.27.2) gatean
    # V2 con una allowlist de arquitecturas
    # (_is_default_v2_model_runner_model); 0.28.1+ cae en `return True`
    # global. Evaluar el modo real del arbol antes de afirmar nada.
    if _gate_mode(root):
        v2 = _arch_in_v2_allowlist(root)
        if v2 is None:
            report.append("  decision: allowlist V2 no evaluable aqui (sin "
                          "config de modelo o fallo el import) -> se parchea "
                          "V1 asumiendo que puede estar vivo")
            return None, report
        if v2:
            report.append("  decision: la arquitectura servida ESTA en la "
                          "allowlist V2 de este build -> runner V2 VIVO")
            return True, report
        report.append("  decision: allowlist pre-0.28: la arquitectura servida "
                      "NO esta en DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES (y el "
                      "gate excluye ademas las hibridas no listas) -> runner "
                      "V1 VIVO: el fix queda ACTIVO en este build")
        return False, report
    report.append("  decision: selector sin allowlist (default V2 global) y "
                  "sin forzar por env -> runner V2 VIVO (confirma con el log "
                  "'Using V2 Model Runner' del arranque)")
    return True, report


def _gate_mode(root: Path) -> bool:
    """True si EL CUERPO del selector usa la allowlist (pre-0.28). Buscar el
    nombre por todo el fichero daria falsos positivos si 0.28+ lo conserva
    muerto; se limita al cuerpo de use_v2_model_runner."""
    cfg = root / "config" / "vllm.py"
    try:
        t = cfg.read_text()
    except OSError:
        return False
    i = t.find("def use_v2_model_runner")
    if i < 0:
        return False
    j = t.find("\n    def ", i + 10)
    body = t[i:j if j > 0 else i + 5000]
    return "_is_default_v2_model_runner_model()" in body


def _model_architectures() -> list[str]:
    model_dir = Path(os.environ.get("B70_MODEL_DIR", "/model"))
    cfg = model_dir / "config.json"
    if not cfg.is_file():
        return []
    try:
        import json
        return list(json.loads(cfg.read_text()).get("architectures") or [])
    except Exception:
        return []


def _arch_in_v2_allowlist(root: Path):
    """None = no evaluable. Evalua la allowlist REAL del build contra el
    config del modelo servido (import en subprocess, CPU-puro)."""
    archs = _model_architectures()
    if not archs:
        return None
    code = (
        "import sys\n"
        "sys.path = [q for q in sys.path if q not in ('',)]\n"
        "try:\n"
        "    from vllm.config.vllm import (\n"
        "        default_v2_model_runner_architectures as _f)\n"
        "    a = set(_f())\n"
        "except Exception:\n"
        "    print('B70HIT None'); raise SystemExit(0)\n"
        f"print('B70HIT', any(x in a for x in {archs!r}))\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True,
                             timeout=600).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if line.startswith("B70HIT True"):
            return True
        if line.startswith("B70HIT False"):
            return False
    return None


def patch_tree(f: Path) -> tuple[int, bool]:
    """(hunks aplicados ahora, arbol parcheable). SystemExit(1) si el runner
    V1 es el vivo y alguna ancla no cuadra exactamente."""
    if not f.is_file():
        return 0, False
    text = f.read_text()
    applied = 0
    for name, marker, anchor, replacement, expected in HUNKS:
        if marker in text:
            print(f"[accept-sync]   {name}: ya aplicado ({marker})")
            continue
        n = text.count(anchor)
        if n != expected:
            print(f"[accept-sync]   {name}: ANCLA NO COINCIDENTE ({n} de "
                  f"{expected}) en {f}", file=sys.stderr)
            return 0, False
        text = text.replace(anchor, replacement, expected)
        applied += expected
        print(f"[accept-sync]   {name}: aplicado ({expected} sitio(s))")
    if applied:
        f.write_text(text)
    return applied, True


def main() -> int:
    if os.environ.get("B70_FIX_ACCEPT_SYNC", "1") == "0":
        print("[accept-sync] desactivado por B70_FIX_ACCEPT_SYNC=0 (vLLM intacto)")
        return 0

    roots = vllm_roots()
    v2_alive, report = runner_report(roots)
    require = os.environ.get("B70_FIX_ACCEPT_SYNC_REQUIRE", "0") == "1"
    live_v1 = v2_alive is False

    print("[accept-sync] === diagnostica de model runner ===")
    for line in report:
        print(line)
    print(f"[accept-sync] parcheando {REL} (fallback V1 / imagen legacy) en "
          f"{len(roots)} arbol(es) candidatos")

    total = 0
    ok_trees = 0
    for root in roots:
        n, ok = patch_tree(root / REL)
        if ok:
            ok_trees += 1
            total += n

    if ok_trees == 0:
        # Nadie tenia las anclas (o el fichero no existe).
        if live_v1 or require:
            print(
                "[accept-sync] FAIL-CLOSED: anclas de vllm#53919 no aplicadas en "
                "ningun arbol y el runner V1 es el vivo (o se exigió con "
                "B70_FIX_ACCEPT_SYNC_REQUIRE=1); el servidor quedaria EXPUESTO "
                "a la corrupcion por doble permutacion del contador de "
                "aceptados.", file=sys.stderr)
            return 1
        print(
            "[accept-sync] WARNING: gpu_model_runner.py no parcheable en esta "
            "imagen (anclas movidas o fichero ausente). Corre runner V2, asi "
            "que ESTE ARRANQUE NO AFECTADO: los contadores de aceptados son "
            "GPU-residents y la carrera no existe (evidencia arriba). Si cayeras "
            "a V1 (VLLM_USE_V2_MODEL_RUNNER=0 o 0.26.1) si quedarias expuesto.",
            file=sys.stderr)
        return 0

    if v2_alive:
        print(f"[accept-sync] OK: {total} hunks en {ok_trees} arbol(es), PERO EL "
              f"FICHERO PARCHEADO ESTA MUERTO EN ESTA CONFIG: corre Model "
              f"Runner V2 y ahi los contadores de aceptados son GPU-residents "
              f"(ver evidencia arriba). Se deja aplicado por si caes a "
              f"V1/0.26.1.")
    else:
        tag = "V1 VIVO: el fix esta ACTIVO" if live_v1 else "runner no decidible"
        print(f"[accept-sync] OK: {total} hunks aplicados ahora en {ok_trees} "
              f"arbol(es) ({tag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
