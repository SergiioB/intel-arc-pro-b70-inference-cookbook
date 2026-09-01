#!/usr/bin/env python3
"""Guarda contra copias de estado mamba HACIA ATRAS (vllm#53505).

POR QUE LA COPIA HACIA ATRAS ES DESTRUCTIVA
-------------------------------------------
En mamba_cache_mode="align" el estado recurrente corre por columnas de bloque
de la tabla de bloques de cada request. Dos kernels copian estado entre
columnas reutilizando el mismo cuerpo ``_copy_mamba_state_block``
(mamba_utils.py:192), que hace una copia de BYTES cruda:

    conv:      state[bt[src_col], token_bias:] -> state[bt[dst_col], :conv_width-token_bias]
    temporal:  state[bt[src_col + token_bias]] -> state[bt[dst_col]]

Nada en ese cuerpo comprueba el orden: si ``dst_col < src_col`` la copia
escribe el estado de una posicion POSTERIOR de la secuencia dentro de una
columna ANTERIOR. La columna anterior es precisamente

  1. el slot desde el que la decodificacion en curso lee su estado inicial
     cuando aun no se ha cruzado el boundary, y
  2. la pagina que el prefix cache ya puede haber publicado
     (``MambaManager`` cachea el bloque de frontera; cualquier request que
     comparta ese prefijo la lee).

Pisarla con un estado de otra posicion no levanta ninguna excepcion: la
secuencia sigue emitiendo tokens normales hasta que el contexto degradado
empieza a pesar, y entonces degenera token a token. Esa es la firma que
reporta upstream vllm#53505 ("prefijo normal, luego degradacion progresiva"),
donde el autor barre el espacio de decisiones
(num_computed, num_scheduled, num_draft, num_accepted) y encuentra 1197
combinaciones que producen una copia hacia atras y NINGUNA que necesite una
hacia delante: cuando ``needs_copy`` salta cruzando un boundary de abajo a
arriba, la copia hacia atras es siempre destructiva, nunca intencionada.

Como se alcanza el caso (verificado en el arbol 0.28.1rc1.dev199):
  * postprocess (mamba_utils.py:370, VIVO en V2 via
    ``run_fused_postprocess_align`` l.1236 <- mamba_hybrid.py:383): con
    aceptacion especulativa,
      ``accept_token_bias = aligned_new_computed - num_tokens_running_state``
      ``dest_block_idx    = aligned_new_computed // block_size - 1``
    puede quedar por debajo de ``src_block_idx`` (el indice de estado que
    llevaba la request), de modo que el destino es una columna anterior.
  * precopy (mamba_utils.py:552, VIVO en V2 via ``model_state.preprocess_state``
    <- gpu/model_runner.py:1602): ``src_col``/``dst_col`` los produce
    ``preprocess_mamba_align_fused_kernel`` (l.506); si el computado real de la
    request retrocede (retract/forced-preemption + resume, chunked prefill que
    reescribe la posicion), ``new_state_idx < state_idx`` y el pre-copy corre
    hacia atras. Su guarda actual (l.609) solo rechaza ``src_col < 0`` y
    ``src_col == dst_col``: NO rechaza ``src_col > dst_col``.
  * collect_mamba_copy_meta (mamba_utils.py:1333, ruta host V1): el docstring de
    ``cleanup_mamba_state_idx`` (l.1390-1395) admite literalmente que las
    entradas de ``mamba_state_idx`` de una request force-preempted "can point to
    block indices beyond the new (smaller) block allocation", es decir
    ``src_block_idx > dest_block_idx``.

COMO SE HACE EL PARCHE
----------------------
Inserta ``if src > dest: return`` en los TRES sitios de copia, cada uno con su
propia ancla contextual unica, su propio expected-count y su propio marcador de
idempotencia. La guarda es estrictamente mayor: deja intacto el caso legitimo
``src == dest`` (deslizamiento dentro del mismo bloque) y no toca el reset de
``num_accepted_tokens`` del kernel de postprocess, que va justo ANTES (l.474) y
por eso se conserva tal cual.

Ademas comprueba invariantes globales del fichero (numero de llamadas a
``_copy_mamba_state_block``, numero de guardas de self-copy) para que un sitio
de copia NUEVO que aparezca upstream no pase desapercibido: falla cerrado en vez
de parchear a medias.

Se desactiva para A/B con B70_FIX_BACKWARD_COPY=0 (deja vLLM intacto).
B70_VLLM_ROOTS="dir1:dir2" fuerza la lista de arboles (lo usa verify_patches.sh).

EXHAUSTIVIDAD (verificado en el arbol 0.28.1rc1.dev199): TODO copy de estado
mamba pasa por uno de estos tres sitios -- ``_copy_mamba_state_block`` tiene
exactamente 2 llamadas (postprocess + precopy) y ``batch_memcpy`` un solo
caller en todo el tree (``do_mamba_copy_block``, alimentado por
``collect_mamba_copy_meta``). No hay otras asignaciones state[dst]=state[src]
en model_executor/layers/mamba ni en v1/worker. Los contadores de FILE_
INVARIANTS y la auditoria AST de verify_patches.sh fallan cerrados si upstream
añade un cuarto sitio.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REL = Path("v1/worker/mamba_utils.py")

# Marcador generico que buscan verify_patches.sh y los lanzadores.
MARKER = "B70_BACKWARD_COPY_GUARD"

# --- sitios de copia -------------------------------------------------------
# Cada entrada: (nombre, marcador propio, ancla unica, reemplazo, expected).
# Los contadores se exigen EXACTOS: ni de mas (sitio nuevo sin revisar) ni de
# menos (upstream movio el codigo y nuestra guarda caería en otro sitio).
SITES: list[tuple[str, str, str, str, int]] = [
    (
        # mamba_utils.py:478 - postprocess_mamba_fused_kernel (V1 y V2)
        # Ancla extendida: el reset de num_accepted_tokens de l.474 va justo
        # antes y no debe moverse.
        "postprocess_mamba_fused_kernel",
        "B70_BACKWARD_COPY_GUARD_POSTPROCESS",
        """    # Skip no-op self-copy.
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return""",
        """    # B70_BACKWARD_COPY_GUARD_POSTPROCESS (vllm#53505): una copia con
    # dest_block_idx < src_block_idx pisaria con el estado de una posicion
    # posterior la columna anterior (estado inicial en curso y/o pagina ya
    # publicada en el prefix cache). Solo src == dest (deslizamiento
    # intra-bloque) es legitimo ademas del avance normal src < dest.
    if src_block_idx > dest_block_idx:
        return

    # Skip no-op self-copy.
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return""",
        1,
    ),
    (
        # mamba_utils.py:609 - precopy_mamba_align_fused_kernel (V2 LIVE)
        "precopy_mamba_align_fused_kernel",
        "B70_BACKWARD_COPY_GUARD_PRECOPY",
        """    if src_col < 0 or src_col == dst_col:
        return""",
        """    # B70_BACKWARD_COPY_GUARD_PRECOPY (vllm#53505): la guarda original solo
    # descarta state nuevo (src_col < 0) y no-op (src_col == dst_col); una
    # columna de origen POR DETRAS de la de destino (num_computed retrocedido
    # tras retract/preemption/resume) copiaria hacia atras y corromperia el
    # estado de un boundary anterior, posiblemente ya cacheado.
    if src_col > dst_col:
        return

    if src_col < 0 or src_col == dst_col:
        return""",
        1,
    ),
    (
        # mamba_utils.py:1344 - collect_mamba_copy_meta (ruta host, V1/
        # spec-decode escalares). Ancla extendida por la firma de la funcion
        # para no confundirse con la del kernel (texto identico).
        "collect_mamba_copy_meta",
        "B70_BACKWARD_COPY_GUARD_COPY_META",
        """    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return""",
        """    forward_context: dict[str, Any],
) -> None:
    # B70_BACKWARD_COPY_GUARD_COPY_META (vllm#53505): mismo criterio que en los
    # kernels; aqui ademas cleanup_mamba_state_idx documenta que un entry
    # obsoleto puede apuntar mas alla de la asignacion nueva (src > dest).
    if src_block_idx > dest_block_idx:
        return

    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return""",
        1,
    ),
]

# Invariantes globales del fichero: si upstream anade quita sitios de copia,
# estos conteos cambian y fallamos cerrado en lugar de dejar un sitio sin guarda.
# re.M porque los patrones van anclados al inicio de LINEA.
FILE_INVARIANTS: list[tuple[str, re.Pattern[str], int]] = [
    ("llamadas a _copy_mamba_state_block",
     re.compile(r"^    _copy_mamba_state_block\(", re.M), 2),
    ("guardas de self-copy (src == dest)",
     re.compile(r"^    if src_block_idx == dest_block_idx and accept_token_bias == 0:$", re.M), 2),
    ("guardas del precopy (src_col/dst_col)",
     re.compile(r"^    if src_col < 0 or src_col == dst_col:$", re.M), 1),
    # Ruta host (collect_mamba_copy_meta): cada entrada de memcpy se resuelve
    # contra esta linea; si aparece una segunda, hay otro sitio que cubrir.
    ("destinos de memcpy host",
     re.compile(r"^        dest_block_id = block_ids\[dest_block_idx\]$", re.M), 1),
]


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


def patch_tree(f: Path) -> int | None:
    """Parchea un arbol. Devuelve el numero de sitios de copia CUBIERTOS por la
    guarda (aplicados ahora o en una ejecucion previa), o None si este arbol no
    es aplicable (se omite). Lanza SystemExit(1) si falla cerrado."""
    text = f.read_text()
    covered = 0
    dirty = False

    for name, marker, anchor, guarded, expected in SITES:
        if marker in text:
            marks = text.count(marker)
            if marks != expected:
                print(
                    f"[backward-copy] FAIL-CLOSED {f}: marcador {marker} "
                    f"aparece {marks} veces (se esperaban {expected}); el "
                    f"parche se aplico dos veces o el fichero esta raro",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            print(f"[backward-copy]   {name}: ya aplicado ({marker})")
            covered += expected
            continue
        n = text.count(anchor)
        if n != expected:
            print(
                f"[backward-copy] FAIL-CLOSED {f}: sitio '{name}' -> {n} "
                f"apariciones de la ancla (se esperaban exactamente {expected}). "
                f"El codigo de copia de mamba_utils.py cambio respecto a "
                f"0.28.1rc1.dev199: revisar donde va la guarda antes de "
                f"arrancar con spec-decode + align.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        text = text.replace(anchor, guarded, expected)
        covered += expected
        dirty = True
        print(f"[backward-copy]   {name}: guarda insertada ({expected} sitio(s))")

    # Invariantes globales (sobre el texto YA parcheado: los patrones de
    # self-copy siguen contando 2 porque la guarda nueva usa otra variable).
    for label, pat, expected in FILE_INVARIANTS:
        found = len(pat.findall(text))
        if found != expected:
            print(
                f"[backward-copy] FAIL-CLOSED {f}: invariante '{label}' = "
                f"{found} (se esperaban {expected}); puede haber un sitio de "
                f"copia mamba nuevo sin cobertura de guarda",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"[backward-copy]   invariante OK: {label} = {found}")

    if dirty:
        f.write_text(text)
    return covered


def main() -> int:
    if os.environ.get("B70_FIX_BACKWARD_COPY", "1") == "0":
        print("[backward-copy] desactivado por B70_FIX_BACKWARD_COPY=0 (vLLM intacto)")
        return 0

    total_sites = sum(expected for _, _, _, _, expected in SITES)
    done = 0
    covered_sites = 0
    for root in vllm_roots():
        f = root / REL
        if not f.is_file():
            continue
        # Sin atajo de "ya aplicado" a nivel de fichero: patch_tree siempre
        # revalida los invariantes del arbol parcheado.
        n = patch_tree(f)
        if n is None:
            continue
        done += 1
        covered_sites += n

    if not done:
        print("[backward-copy] FAIL-CLOSED: no se encontro mamba_utils.py "
              "parcheable en ningun arbol; el servidor queda EXPUESTO a la "
              "copia hacia atras (vllm#53505)", file=sys.stderr)
        return 1
    print(f"[backward-copy] OK: {done} arbol(es) x {total_sites} sitios de copia "
          f"cubiertos ({covered_sites} en total: postprocess_mamba_fused_kernel "
          f"+ precopy_mamba_align_fused_kernel [V2 LIVE] + collect_mamba_copy_meta)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
