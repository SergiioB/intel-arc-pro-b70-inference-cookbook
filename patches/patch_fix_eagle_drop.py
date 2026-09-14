#!/usr/bin/env python3
"""Port de vllm#48375 (drop_eagle_block en MambaManager) a las DOS ramas.

EL PROBLEMA
-----------
El llamador (``kv_cache_coordinator.py``) pide explicitamente que se descarte
el ultimo bloque emparejado cuando hay un draft model (EAGLE/MTP). El
comentario real del coordinador (0.28.1rc1.dev199, l.832-838) dice:

    «Eagle matches one extra drop unit (one hash unit for fine-grained
    managers, else one cache block) and then drops it, landing back at the
    candidate length. No margin for mamba: its finder never drops (draft
    models have no mamba layers), so the hit would grow past the candidate.»

Y efectivamente ``if drop_eagle_block and not isinstance(spec, MambaSpec):``
(l.840) da margen eagle al resto de managers pero NINGUNO a mamba, porque
``MambaManager.find_longest_cache_hit`` (single_type_kv_cache_manager.py:1393)
RECIBE ``drop_eagle_block`` y lo IGNORA en las dos ramas: empieza a buscar en
el indice mas alto permitido (``max_num_blocks - 1`` /
``max_num_partial_units - 1``) y lo acepta como hit. La ultima pagina cacheada
sigue siendo alcanzable aunque su snapshot de estado recurrente pueda haber
sido escrito sobre posiciones de draft que la verificacion rechazo despues. Las
requests posteriores que compartan ese prefijo heredan un estado que no
corresponde a esa posicion: contaminacion cruzada PERSISTENTE a traves del
cache (upstream: vllm#43559 / #53912 / #48375 / #43650).

Nota: el PR de upstream solo toca la rama gruesa, y aqui se cubren las dos.
``drop_eagle_block`` SI puede llegar True a un grupo mamba:
``eagle_group_ids`` son los grupos con ``is_eagle_group``, y cuando el config
pedea eagle sin ninguno marcado se activan TODOS los grupos
(kv_cache_coordinator.py:112-113), incluido el de mamba.

CUAL RAMA ES LA NUESTRA (ojo: al reves de lo que afirmaba esta docstring antes)
-------------------------------------------------------------------------------
Dentro de ``find_longest_cache_hit`` la eleccion es

    ``if alignment_tokens < block_size and block_size % alignment_tokens == 0:``   (l.1423)

con ``alignment_tokens = _cache_hit_alignment_tokens``, que vale
``hash_block_size`` si ``enable_partial_hash_hits`` y ``scheduler_block_size``
en caso contrario (coordinador l.666-673); ``enable_partial_hash_hits`` solo es
True si hay un grupo mamba "align" con ``block_size > hash_block_size``
(l.630-641).

* Rama GRUESA = LA NUESTRA POR DEFECTO. Con la geometria auto-ajustada de esta
  maquina (atencion block_size 1664 == pagina mamba 1664, ``hash_block_size``
  1664, ``enable_partial_hash_hits=False``) se cumple ``alignment_tokens ==
  block_size``: la condicion fina es False y se entra por el bloque grueso. El
  parche baja aqui el techo EXACTAMENTE 1 bloque (1664 tokens), la semantica de
  vllm#48375.
* Rama FINA = LA QUE ACTIVA ``--prefix-match-unit``. Con
  ``--prefix-match-unit 64`` resulta ``hash_block_size=64 < 1664`` ->
  ``enable_partial_hash_hits=True`` -> ``alignment_tokens=64`` -> rama fina.
  ``scale_factor = block_size // hash_block_size = 26``.

El docstring anterior decia justo lo contrario (que esta config entraba por la
rama fina); queda corregido. Como el servidor puede arrancarse de las dos
formas, las DOS ramas quedan parcheadas.

PARAMETRO DE LA RAMA FINA: B70_EAGLE_DROP_FINE_UNITS
----------------------------------------------------
Cuantas unidades de hash baja el techo de busqueda la rama fina. Default ``1``.

* ``1`` (default): una unidad de hash = 64 tokens. Es la semantica literal del
  comentario del coordinador ("one hash unit for fine-grained managers") y la
  que conserva todo el beneficio del ``--prefix-match-unit``. Proteccion floja:
  si el snapshot sospechoso es el de la pagina entera, seguimos aceptando hits
  dentro de esa pagina. El auditor advierte de que con replay multi-turno esto
  puede ser un no-op (la snapshot posiblemente contaminada se reutiliza igual).
* ``page`` / ``full`` / ``scale``: una PAGINA DE MAMBA ENTERA, o sea
  ``scale_factor`` unidades (= 1664 // 64 = 26 aqui, 1664 tokens). Proteccion
  fuerte, a cambio de hasta una pagina menos de reutilizacion por lookup. Se
  resuelve CONTRA EL ``scale_factor`` DE RUNTIME (no se graba el 26 en el
  fuente), asi que sigue siendo correcto si cambia la geometria.
* cualquier entero ``>= 1``: ese numero de unidades, truncado a
  ``scale_factor`` (bajar mas de una pagina no protege nada extra y solo cuesta
  reutilizacion).

Un valor ``< 1`` o no parseable hace FAIL-CLOSED al parche (rc 1, nada tocado):
la eleccion entre reutilizacion y proteccion la decide el operador, no este
script. La rama GRUESA ignora este parametro: baja siempre 1 bloque.

Se desactiva para A/B con B70_FIX_EAGLE_DROP=0 (deja vLLM intacto).
B70_VLLM_ROOTS="dir1:dir2" fuerza la lista de arboles (lo usa verify_patches.sh).
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

REL = Path("v1/core/single_type_kv_cache_manager.py")

# Marcador generico que buscan verify_patches.sh y los lanzadores.
MARKER = "B70_EAGLE_DROP"

# Detectores de idempotencia POR HUNK. Tienen que ser cadenas exclusivas: el
# helper cita el nombre del env (B70_EAGLE_DROP_FINE_UNITS) en su docstring, asi
# que usar "B70_EAGLE_DROP_FINE" como marcador haria que el hunk de la rama fina
# se diera por aplicado sin estarlo.
HELPER_MARKER = "def _b70_eagle_drop_fine_units"
FINE_MARKER = "B70_EAGLE_DROP_FINE_HUNK"
COARSE_MARKER = "B70_EAGLE_DROP_COARSE_HUNK"
HELPER = "_b70_eagle_drop_fine_units"

PAGE_KEYWORDS = ("page", "full", "scale", "scale_factor", "block")
RUNTIME_KEYWORDS = '"page", "full", "scale", "scale_factor", "block"'


def fine_units_setting() -> tuple[str, int | None]:
    """('page', None) si el operador pide una pagina entera; ('units', N) si pide
    un numero concreto. Fail-closed (rc 1) con valores invalidos."""
    raw = os.environ.get("B70_EAGLE_DROP_FINE_UNITS", "1").strip().lower()
    if raw in PAGE_KEYWORDS:
        return ("page", None)
    try:
        n = int(raw)
    except ValueError:
        print(
            f"[eagle-drop] FAIL-CLOSED: B70_EAGLE_DROP_FINE_UNITS={raw!r} no es "
            f"un entero ni una de {PAGE_KEYWORDS}. No se parchea nada.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if n < 1:
        print(
            f"[eagle-drop] FAIL-CLOSED: B70_EAGLE_DROP_FINE_UNITS={n} < 1. Bajar "
            f"0 unidades desactivaria el drop y dejaria el cache EXPUESTO a "
            f"#48375; para desactivar el parche entero usa B70_FIX_EAGLE_DROP=0.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return ("units", n)


# --- hunk A: helper de modulo (resuelve scale_factor en runtime) ------------
# Plantilla con placeholders @...@ (se sustituyen con .replace): asi el codigo
# generado puede llevar sus propias llaves de f-string sin escaparse.
HELPER_ANCHOR = "class MambaManager(SingleTypeKVCacheManager):"
HELPER_TEMPLATE = '''def @HELPER@(scale_factor: int) -> int:
    """@MARKER@ (rama fina): unidades de hash que descarta el drop EAGLE en
    ``MambaManager.find_longest_cache_hit``.

    Controlado por el env B70_EAGLE_DROP_FINE_UNITS: 1 (default) = la semantica
    del comentario de kv_cache_coordinator; page/full/scale = pagina de mamba
    entera (scale_factor unidades). Nunca mas de una pagina: bajar mas no
    protege nada extra y solo cuesta reutilizacion.
    """
    raw = os.environ.get("B70_EAGLE_DROP_FINE_UNITS", "1").strip().lower()
    if raw in (@KEYWORDS@):
        units = scale_factor
    else:
        try:
            units = int(raw)
        except ValueError:
            raise RuntimeError(
                "B70_EAGLE_DROP_FINE_UNITS must be an integer >= 1 or one of "
                f"page/full/scale; got {raw!r}") from None
    if units < 1:
        raise RuntimeError(
            f"B70_EAGLE_DROP_FINE_UNITS must be >= 1, got {units}")
    return min(units, scale_factor)


'''
HELPER_CODE = (
    HELPER_TEMPLATE
    .replace("@HELPER@", HELPER)
    .replace("@MARKER@", MARKER)
    .replace("@KEYWORDS@", RUNTIME_KEYWORDS)
)

# --- hunk B: rama FINA (hash-block-aligned) --------------------------------
FINE_OLD = """            scale_factor = block_size // hash_block_size
            max_num_partial_units = min(
                max_length // hash_block_size, len(block_hashes)
            )"""

FINE_NEW = """            scale_factor = block_size // hash_block_size
            max_num_partial_units = min(
                max_length // hash_block_size, len(block_hashes)
            )
            if drop_eagle_block and max_num_partial_units > 0:
                # B70_EAGLE_DROP_FINE_HUNK (port de vllm#48375 a la rama fina):
                # el techo incluia el ultimo hash unit y no se descartaba nada,
                # asi que su snapshot (posiblemente escrito sobre posiciones de
                # draft rechazadas) se reutilizaba. Se bajan
                # B70_EAGLE_DROP_FINE_UNITS unidades (1 por defecto; "page" para
                # una pagina de mamba entera), sin bajar nunca de 0.
                max_num_partial_units -= min(
                    _b70_eagle_drop_fine_units(scale_factor),
                    max_num_partial_units,
                )"""

# --- hunk C: rama GRUESA (cache-block-aligned) -- la nuestra por defecto ----
COARSE_OLD = """        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found."""

COARSE_NEW = """        max_num_blocks = max_length // block_size
        if drop_eagle_block and max_num_blocks > 0:
            # B70_EAGLE_DROP_COARSE_HUNK (port de vllm#48375): descarta la
            # ultima pagina emparejada, como hacen los demas managers gracias al
            # eagle_margin del coordinador. Siempre EXACTAMENTE 1 bloque: aqui la
            # unidad de busqueda ya es la pagina de mamba, y el parametro de la
            # rama fina no aplica a esta rama.
            max_num_blocks -= 1
        # Search from right to left and early stop when a match is found."""

HUNKS: list[tuple[str, str, str, str, int]] = [
    ("helper B70_EAGLE_DROP_FINE_UNITS", HELPER_MARKER, HELPER_ANCHOR,
     HELPER_CODE + HELPER_ANCHOR, 1),
    ("rama fina", FINE_MARKER, FINE_OLD, FINE_NEW, 1),
    ("rama gruesa", COARSE_MARKER, COARSE_OLD, COARSE_NEW, 1),
]

# Techos de busqueda del fichero: si upstream anade una tercera rama, estos
# conteos se mueven y fallamos cerrado en vez de dejarla sin drop.
SEARCH_CAP_PATTERNS: list[tuple[str, str, int]] = [
    ("            max_num_partial_units = min(", "rama fina", 1),
    ("        max_num_blocks = max_length // block_size", "rama gruesa", 1),
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


def ensure_import_os(text: str, f: Path) -> str:
    """El helper lee el entorno: sin `import os` a nivel de modulo, el arbol
    parcheado reventaria en el primer cache hit. Insertado con ast para no
    caer nunca dentro del docstring del modulo."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        print(f"[eagle-drop] FAIL-CLOSED {f}: el texto parcheado no compila "
              f"({exc}); no se escribe nada", file=sys.stderr)
        raise SystemExit(1)
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
            a.name == "os" and a.asname is None for a in node.names
        ):
            return text
    lines = text.split("\n")
    insert_at = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            insert_at = node.lineno - 1
            break
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            insert_at = node.end_lineno  # docstring de modulo
            break
    lines.insert(insert_at, f"import os  # {MARKER}")
    print(f"[eagle-drop]   'import os' inyectado en {f}:{insert_at + 1}")
    return "\n".join(lines)


def patch_tree(f: Path) -> int:
    """Parchea un arbol y SIEMPRE valida los invariantes, incluso cuando el
    parche ya estaba aplicado: es la unica forma de detectar que upstream anadio
    una rama de busqueda nueva."""
    original = text = f.read_text()
    applied = 0

    for name, marker, anchor, replacement, expected in HUNKS:
        if marker in text:
            print(f"[eagle-drop]   {name}: ya aplicado ({marker})")
            continue
        n = text.count(anchor)
        if n != expected:
            print(
                f"[eagle-drop] FAIL-CLOSED {f}: hunk '{name}' -> {n} apariciones "
                f"de la ancla (se esperaban exactamente {expected}); el codigo de "
                f"find_longest_cache_hit cambio respecto a 0.28.1rc1.dev199.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        text = text.replace(anchor, replacement, expected)
        applied += expected
        print(f"[eagle-drop]   {name}: aplicado ({expected} sitio(s))")

    for pat, label, expected in SEARCH_CAP_PATTERNS:
        found = text.count(pat)
        if found != expected:
            print(
                f"[eagle-drop] FAIL-CLOSED {f}: techo de busqueda '{pat.strip()}' "
                f"({label}) = {found}, se esperaban {expected}; puede haber una "
                f"rama de busqueda nueva sin parchear.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"[eagle-drop]   invariante OK: {label} = {found}")

    text = ensure_import_os(text, f)
    if text != original:
        f.write_text(text)
    return applied


def main() -> int:
    if os.environ.get("B70_FIX_EAGLE_DROP", "1") == "0":
        print("[eagle-drop] desactivado por B70_FIX_EAGLE_DROP=0 (vLLM intacto)")
        return 0

    mode, units = fine_units_setting()
    raw = os.environ.get("B70_EAGLE_DROP_FINE_UNITS", "1")
    fine_desc = ("una pagina de mamba entera (scale_factor unidades, resueltas "
                 "en runtime)" if mode == "page" else
                 f"{units} unidad(es) de hash")
    print(f"[eagle-drop] B70_EAGLE_DROP_FINE_UNITS={raw!r} -> rama fina baja: "
          f"{fine_desc} | rama gruesa baja: 1 bloque (fijo)")

    done = patched = 0
    for root in vllm_roots():
        f = root / REL
        if not f.is_file():
            continue
        patched += patch_tree(f)
        done += 1

    if not done:
        print(
            "[eagle-drop] FAIL-CLOSED: no se encontro el MambaManager "
            "parcheable; el servidor queda EXPUESTO a la contaminacion por "
            "cache (vllm#48375). No usar --enable-prefix-caching sin este fix.",
            file=sys.stderr,
        )
        return 1
    print(f"[eagle-drop] OK: {done} arbol(es), {patched} hunks nuevos "
          f"(helper + rama fina + rama gruesa)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
