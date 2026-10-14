#!/usr/bin/env python3
"""Weight sources for the Qwen3.8-Flash-Next (Qwen4Exp) KL harness.

CPU only, no GPU.  Two sources behind one access API:

  EXL3Source(root, refsrc)      local exllamav3 artifact dir; quantized groups are
                                reconstructed with the pinned ``ref`` module.
  BF16Source(local_dir=...)     local safetensors shards (seek + pread, no full-file map)
  BF16Source(repo=, rev=, ...)  official HF checkpoint; one ranged HTTP read per byte
                                range, with Content-Range verification

  names() -> set[str]                  logical names (cached)
  meta(name) -> TensorMeta             logical shape / dtype / nbytes
  get(name) -> torch.Tensor            whole tensor
  get_rows(name, ids) -> Tensor        gather rows along axis 0 (row-major)
  get_columns(name, c0, c1) -> Tensor  output-axis block == get(name)[c0:c1]
  get_many(names, max_gap_bytes)       coalesced multi-tensor read
  stats: dict                          cumulative counters

Layout convention, 2-D tensors: **HF / safetensors orientation** ``[out_features,
in_features]`` for both sources.  ``get()`` on an EXL3 *quantized* tensor returns the
reconstruction of ``W_orig`` (which exllamav3 stores as ``(in, out)``) transposed to
``[out, in]`` and cast to bfloat16, mirroring the 27B harness's
``weight.to(torch.bfloat16)``.  The EXL3 trellis axis 1 is the *out* axis, ``suh`` is the
*in* axis and ``svh`` the *out* axis; ``get_columns`` therefore decodes
``trellis[:, c0//16:c1//16]`` with ``svh[c0:c1]``.

Known limit: the FlashNext EXL3 artifact's ``ngram_embedding.safetensors`` uses a custom
"exl3_ngram_trellis" row layout whose bit packing is *not* resolvable from the pinned
``ref`` decoder; its tensors are addressable (``names()``/``meta()``) but row reads raise a
descriptive ``RuntimeError`` instead of returning guessed values.  See
``logs/sources_api.md``.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import sys
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "TensorMeta",
    "read_safetensors_headers",
    "build_index",
    "BaseSource",
    "EXL3Source",
    "BF16Source",
    "SAFETENSORS_DTYPES",
    "COMPONENT_SUFFIXES",
    "HEADER_MAX_BYTES",
]

SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U64": torch.uint64,
    "U32": torch.uint32,
    "U16": torch.uint16,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

# exllamav3 companion keys written next to a quantized tensor ``<stem>``
COMPONENT_SUFFIXES = (".trellis", ".suh", ".svh", ".mul1", ".mcg")

HEADER_MAX_BYTES = 1 << 30          # safetensors header length sanity bound
INDEX_NAME = "model.safetensors.index.json"
NGRAM_FILE = "ngram_embedding.safetensors"
HTTP_WORKERS_MAX = max(1, int(os.environ.get("FN_HTTP_WORKERS", "4")))   # hard cap on concurrent fetch threads
DEFAULT_MAX_SPAN = 256 << 20        # coalesced read-span budget (bytes)
HTTP_MAX_RANGE = 256 << 20          # largest single ranged GET

warnings.filterwarnings(
    "ignore", message="The given buffer is not writable.*", category=UserWarning)


# --------------------------------------------------------------------------- helpers


def _prod(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _tensor_from_buffer(buf, dtype: torch.dtype, shape: Sequence[int]) -> torch.Tensor:
    """Zero-copy torch tensor viewing ``buf`` (bytes / bytearray / memoryview)."""
    try:
        t = torch.frombuffer(buf, dtype=dtype)
    except (TypeError, RuntimeError, ValueError):
        t = torch.frombuffer(bytearray(buf), dtype=dtype)
    return t.reshape(tuple(int(s) for s in shape))


def _canon(name: str) -> str:
    """``.weight``-less spelling, used for stem lookups ('lm_head' <-> 'lm_head.weight')."""
    return name[:-len(".weight")] if name.endswith(".weight") else name


def _runs(ids: Sequence[int]) -> List[Tuple[int, int]]:
    """Sorted unique ids -> maximal contiguous ``[start, stop)`` runs."""
    out: List[Tuple[int, int]] = []
    for v in ids:
        if out and v == out[-1][1]:
            out[-1] = (out[-1][0], v + 1)
        else:
            out.append((v, v + 1))
    return out


def _coalesce(ranges: Sequence[Tuple[int, int, str]], max_gap: int,
              max_span: int = DEFAULT_MAX_SPAN) -> List[Tuple[int, int]]:
    """``[(off, nbytes, key)]`` -> merged ``[(off, nbytes)]``; gap <= max_gap, span <= max_span."""
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda r: (r[0], r[1]))
    spans: List[Tuple[int, int]] = []
    cur_off, cur_end = ordered[0][0], ordered[0][0] + ordered[0][1]
    for off, nbytes, _ in ordered[1:]:
        end = off + nbytes
        if off - cur_end <= max_gap and end - cur_off <= max_span:
            cur_end = max(cur_end, end)
        else:
            spans.append((cur_off, cur_end - cur_off))
            cur_off, cur_end = off, end
    spans.append((cur_off, cur_end - cur_off))
    return spans


def _span_starts(spans: Sequence[Tuple[int, int]]) -> List[int]:
    return [s[0] for s in spans]


def _slice_within(spans: Sequence[Tuple[int, int]], off: int, nbytes: int,
                  starts: Optional[Sequence[int]] = None) -> int:
    """Index of the span covering ``[off, off+nbytes)`` (-1 when uncovered).

    ``spans`` must be ascending and non-overlapping (as produced by ``_coalesce``);
    pass ``starts`` to avoid rebuilding the key list in a loop."""
    if starts is None:
        starts = _span_starts(spans)
    i = bisect.bisect_right(starts, off) - 1
    if i >= 0:
        soff, sn = spans[i]
        if soff <= off and off + nbytes <= soff + sn:
            return i
    return -1


# ---------------------------------------------------------------- safetensors headers


@dataclass
class TensorMeta:
    """One stored safetensors tensor (or one EXL3 quantized group).

    name     logical tensor name (EXL3 quantized: the index stem, e.g. ``lm_head`` or
             ``model.language_model.layers.0.mlp.gate_proj.weight``)
    dtype    stored dtype tag for plain tensors ("BF16", "F16", ...); "BF16" for an EXL3
             quantized tensor (the dtype its reconstruction is returned in)
    shape    logical shape.  EXL3 quantized: the reconstructed ``[out_features,
             in_features]``, *not* the trellis shape.
    file     shard file name (relative to the artifact root)
    offsets  ``[start, end)`` from the safetensors header, relative to the start of the data
             section (absolute = ``8 + header_len + offsets[i]``).  EXL3 quantized: the
             *trellis* tensor's offsets.
    nbytes   logical byte size (``prod(shape) * element_size`` of the returned dtype)
    """

    name: str
    dtype: str
    shape: List[int]
    file: str
    offsets: List[int]
    nbytes: int
    # ---- additive bookkeeping (beyond the frozen 6-field record) ----
    quant: bool = False            # True for an EXL3 trellis group
    ks: int = 0                    # trellis K (bits/weight) when quant
    stored_nbytes: int = 0         # bytes stored (trellis group payload when quant)
    stored_shape: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if not self.stored_nbytes:
            self.stored_nbytes = self.nbytes

    @property
    def numel(self) -> int:
        return _prod(self.shape)


@dataclass
class _FileHeader:
    path: Path
    names: Dict[str, dict]
    header_len: int
    metadata: dict
    file_size: int
    remote: bool = False

    @property
    def data_start(self) -> int:
        return 8 + self.header_len

    @property
    def data_end(self) -> int:
        end = 0
        for v in self.names.values():
            end = max(end, int(v["data_offsets"][1]))
        return end

    @property
    def complete(self) -> bool:
        return self.remote or self.file_size >= self.data_start + self.data_end


def _parse_header_bytes(raw: bytes, where: str) -> Tuple[dict, int]:
    if len(raw) < 8:
        raise ValueError(f"{where}: shorter than the 8-byte safetensors header length")
    n = int.from_bytes(raw[:8], "little")
    if n <= 0 or n > HEADER_MAX_BYTES:
        raise ValueError(f"{where}: implausible safetensors header length {n}")
    if 8 + n > len(raw):
        raise ValueError(f"{where}: truncated header ({len(raw)} bytes for {8 + n})")
    hdr = json.loads(raw[8:8 + n].decode("utf-8"))
    if not isinstance(hdr, dict):
        raise ValueError(f"{where}: safetensors header is not a JSON object")
    return hdr, n


def _header_from_parsed(hdr: dict, n: int, path: Path, file_size: int,
                        remote: bool = False) -> _FileHeader:
    body = dict(hdr)
    meta = body.pop("__metadata__", {}) or {}
    names = {k: v for k, v in body.items()
             if isinstance(v, dict) and "data_offsets" in v}
    if not names:
        raise ValueError(f"{path}: safetensors header holds no tensors")
    return _FileHeader(path=path, names=names, header_len=n, metadata=meta,
                       file_size=file_size, remote=remote)


def _read_file_header(path: Path) -> _FileHeader:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        n = int.from_bytes(raw, "little") if len(raw) == 8 else 0
        if n <= 0 or n > HEADER_MAX_BYTES:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        raw += f.read(n)
    hdr, n = _parse_header_bytes(raw, str(path))
    return _header_from_parsed(hdr, n, path, size)


def _meta_from_entry(name: str, entry: dict, fname: str, fh: _FileHeader,
                     strict: bool) -> TensorMeta:
    dt = entry["dtype"]
    if dt not in SAFETENSORS_DTYPES:
        raise ValueError(f"{fname}:{name}: unsupported safetensors dtype {dt!r}")
    shape = [int(s) for s in entry["shape"]]
    off = [int(o) for o in entry["data_offsets"]]
    itemsize = SAFETENSORS_DTYPES[dt].itemsize
    nbytes = _prod(shape) * itemsize
    if len(off) != 2 or off[1] - off[0] != nbytes:
        raise ValueError(
            f"{fname}:{name}: data_offsets {off} != prod(shape)*item_size ({nbytes})")
    if strict and fh.data_start + off[1] > fh.file_size:
        raise ValueError(
            f"{fname}:{name}: tensor ends at {fh.data_start + off[1]} but the file is "
            f"{fh.file_size} bytes (incomplete download)")
    return TensorMeta(name=name, dtype=dt, shape=shape, file=fname, offsets=off,
                      nbytes=nbytes)


def _check_layout(fh: _FileHeader, fname: str, strict: bool) -> None:
    """sum(nbytes) == max(data_offsets): the payloads must tile the data section."""
    total = 0
    for v in fh.names.values():
        dt = v["dtype"]
        if dt not in SAFETENSORS_DTYPES:
            raise ValueError(f"{fname}: unsupported dtype {dt!r}")
        total += _prod(v["shape"]) * SAFETENSORS_DTYPES[dt].itemsize
    end = fh.data_end
    if total != end:
        raise ValueError(
            f"{fname}: tensor payloads sum to {total} bytes but the data section is {end} "
            "bytes (non-contiguous or overlapping layout)")
    if strict and fh.data_start + end != fh.file_size:
        raise ValueError(
            f"{fname}: data section ends at {fh.data_start + end} but the file is "
            f"{fh.file_size} bytes (incomplete or padded file)")


def read_safetensors_headers(path_or_dir, strict: bool = True) -> Dict[str, TensorMeta]:
    """Parse one ``*.safetensors`` file or an artifact directory into ``{name: TensorMeta}``.

    Directory case: when ``model.safetensors.index.json`` exists, its ``weight_map`` decides
    which shard holds which tensor and every mapped tensor must be present; otherwise every
    ``*.safetensors`` file in the directory is read.  ``strict`` additionally requires each
    file to be exactly as long as its declared data section (an in-progress download fails).
    """
    p = Path(path_or_dir)
    out: Dict[str, TensorMeta] = {}
    if p.is_dir():
        idx = p / INDEX_NAME
        names_by_file: Optional[Dict[str, List[str]]] = None
        if idx.is_file():
            wm = json.loads(idx.read_text())["weight_map"]
            names_by_file = defaultdict(list)
            for name, fname in wm.items():
                names_by_file[str(fname)].append(str(name))
            if not names_by_file:
                raise ValueError(f"{idx}: empty weight_map")
            files = sorted(names_by_file)
        else:
            files = sorted(x.name for x in p.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"{p}: no *.safetensors and no {INDEX_NAME}")
        for fname in files:
            fp = p / fname
            if not fp.is_file():
                raise FileNotFoundError(f"{p}: index references missing shard {fname}")
            fh = _read_file_header(fp)
            _check_layout(fh, fname, strict)
            want = set(names_by_file[fname]) if names_by_file else set(fh.names)
            missing = want - set(fh.names)
            if missing:
                raise ValueError(f"{fname}: {len(missing)} indexed tensors absent, e.g. "
                                 f"{sorted(missing)[:3]}")
            for name in want:
                out[name] = _meta_from_entry(name, fh.names[name], fname, fh, strict)
        return out
    if p.is_file():
        fh = _read_file_header(p)
        _check_layout(fh, p.name, strict)
        for name, entry in fh.names.items():
            out[name] = _meta_from_entry(name, entry, p.name, fh, strict)
        return out
    raise FileNotFoundError(f"{p}: not a file or directory")


def build_index(root: str, strict: bool = True) -> Dict[str, TensorMeta]:
    """``local artifact dir -> {name: TensorMeta}``.

    Uses ``model.safetensors.index.json`` when present and *additionally* indexes every other
    ``*.safetensors`` file in the directory (``ngram_embedding.safetensors``,
    ``mtp_hyper_connection_mixer_patch.safetensors``, ...), so the name set is complete.
    """
    p = Path(root)
    out = read_safetensors_headers(p, strict=strict)
    covered = {m.file for m in out.values()}
    for f in sorted(p.glob("*.safetensors")):
        if f.name in covered:
            continue
        for name, meta in read_safetensors_headers(f, strict=strict).items():
            if name in out:
                raise ValueError(f"{name}: present in {out[name].file} and {f.name}")
            out[name] = meta
    return out


# ---------------------------------------------------------------------------- readers


# Linux caps one read()/pread() at MAX_RW_COUNT = INT_MAX & PAGE_MASK; larger requests
# return short.  Tensors bigger than this (embed_tokens, the ngram table) must be chunked.
MAX_READ = 0x7FFFF000


class _LocalFile:
    """Byte-range reader over one local file (``os.pread``; no mmap, no whole-file read)."""

    remote = False

    def __init__(self, path: Path):
        self.path = Path(path)
        self.size = os.path.getsize(self.path)
        self._fd: Optional[int] = None
        self._lock = threading.Lock()

    def _getfd(self) -> int:
        with self._lock:
            if self._fd is None:
                self._fd = os.open(self.path, os.O_RDONLY)
            return self._fd

    def read_range(self, off: int, nbytes: int) -> bytes:
        if nbytes == 0:
            return b""
        if off < 0 or nbytes < 0 or off + nbytes > self.size:
            raise ValueError(
                f"{self.path}: range {off}+{nbytes} outside a {self.size}-byte file "
                "(incomplete download?)")
        fd = self._getfd()
        if nbytes <= MAX_READ:
            buf = os.pread(fd, nbytes, off)
            if len(buf) != nbytes:
                raise ValueError(f"{self.path}: short pread {len(buf)}/{nbytes} at {off}")
            return buf
        out = bytearray(nbytes)
        view = memoryview(out)
        done = 0
        while done < nbytes:
            n = min(MAX_READ, nbytes - done)
            got = os.preadv(fd, [view[done:done + n]], off + done)
            if got <= 0:
                raise ValueError(f"{self.path}: preadv returned {got} at {off + done}")
            done += got
        return out

    def read_ranges(self, ranges: Sequence[Tuple[int, int]],
                    workers: int = 1) -> List[bytes]:
        if len(ranges) <= 1 or workers <= 1:
            return [self.read_range(o, n) for o, n in ranges]
        with ThreadPoolExecutor(max_workers=min(workers, HTTP_WORKERS_MAX)) as ex:
            return list(ex.map(lambda r: self.read_range(r[0], r[1]), ranges))

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None


class _HTTPFile:
    """Ranged reader over one Hub file; the CDN URL is resolved once and cached.

    The resolved CDN URL is a *signed* URL with an ``Expires`` (hours), so a long run can
    outlive it. On 401/403/410 the reader re-resolves at the same pinned revision and
    retries that one range, leaving bytes, revision and Range header unchanged.

    Every ranged read verifies ``Content-Range`` (and the body length) before the bytes are
    trusted, and retries with backoff before raising.
    """

    remote = True

    def __init__(self, repo: str, rev: str, path: str, token: Optional[str],
                 stats: dict, stats_lock: threading.Lock, retries: int = 5,
                 metadata=None):
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        self.repo, self.rev, self.path = repo, rev, path
        self.stats, self._lock, self.retries = stats, stats_lock, retries
        self.hub_url = hf_hub_url(repo, path, revision=rev)
        md = metadata if metadata is not None else self._head(token, f"HEAD {path}")
        self.location = getattr(md, "location", None) or self.hub_url
        self.size = int(getattr(md, "size", 0) or 0)
        lfs = getattr(md, "lfs", None)
        self.sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        self.etag = getattr(md, "etag", None)
        self._token = token
        self._tl = threading.local()
        self.pool = None            # set by BF16Source: parallel chunk reads (<= 4 threads)

    def _head(self, token: Optional[str], what: str):
        from huggingface_hub import get_hf_file_metadata

        def once():
            with self._lock:
                self.stats["http_requests"] += 1
                self.stats["http_heads"] += 1
            return get_hf_file_metadata(self.hub_url, token=token)

        return self._retry(once, what)

    def _retry(self, fn, what: str):
        last: Optional[BaseException] = None
        for i in range(self.retries):
            try:
                return fn()
            except BaseException as exc:            # noqa: BLE001 - retried
                last = exc
                with self._lock:
                    self.stats["retries"] += 1
                time.sleep(min(2 ** i, 15) * 0.5)
        raise RuntimeError(f"{what}: failed after {self.retries} attempts: {last}")

    def _session(self):
        import requests

        s = getattr(self._tl, "session", None)
        if s is None:
            s = requests.Session()
            self._tl.session = s
        return s

    def read_range(self, off: int, nbytes: int) -> bytes:
        """Read ``nbytes`` at ``off``; ranges above ``HTTP_MAX_RANGE`` are chunked so a
        single request never asks a CDN for more than 256 MiB.  The chunks are assembled
        into one buffer (one copy of the tensor, plus at most ``workers`` in-flight
        chunks), never two full copies."""
        if nbytes <= HTTP_MAX_RANGE:
            return self._read_range_once(off, nbytes)
        chunks = [(off + d, min(HTTP_MAX_RANGE, nbytes - d))
                  for d in range(0, nbytes, HTTP_MAX_RANGE)]
        out = bytearray(nbytes)
        in_pool = threading.current_thread().name.startswith("fnfetch")
        if self.pool is not None and len(chunks) > 1 and not in_pool:
            parts = list(self.pool.map(lambda r: self._read_range_once(r[0], r[1]), chunks))
        else:
            parts = [self._read_range_once(o, n) for o, n in chunks]
        done = 0
        for (o, n), part in zip(chunks, parts):
            if len(part) != n:
                raise RuntimeError(f"{self.path}: short chunk {len(part)}/{n} at {o}")
            out[o - off:o - off + n] = part
            done += n
        return out

    def _refresh_location(self) -> bool:
        """Re-resolve the signed CDN URL for the pinned revision (stats-counted)."""
        try:
            md = self._head(self._token, f"HEAD {self.path} (url refresh)")
        except Exception:
            return False
        loc = getattr(md, "location", None)
        if not loc:
            return False
        changed = loc != self.location
        self.location = loc
        if changed:
            with self._lock:
                self.stats["url_refreshes"] = int(self.stats.get("url_refreshes", 0)) + 1
        return True

    def _read_range_once(self, off: int, nbytes: int) -> bytes:
        if nbytes == 0:
            return b""
        if self.size and off + nbytes > self.size:
            raise ValueError(f"{self.path}: range {off}+{nbytes} beyond size {self.size}")

        def once() -> bytes:
            headers = {"Range": f"bytes={off}-{off + nbytes - 1}",
                       "Accept-Encoding": "identity"}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            with self._lock:
                self.stats["http_requests"] += 1
                self.stats["http_ranges"] += 1
            r = self._session().get(self.location, headers=headers, timeout=180)
            if r.status_code in (401, 403, 410):
                # The cached CDN URL is signed and expires; re-resolve at the pinned revision
                # and retry this exact range once. Same bytes, same revision, same Range header.
                if self._refresh_location():
                    r = self._session().get(self.location, headers=headers, timeout=180)
            if r.status_code != 206:
                if r.status_code == 200:
                    raise RuntimeError(
                        f"{self.path}: server ignored Range (wanted {nbytes} bytes, got the "
                        f"whole {len(r.content)}-byte file)")
                raise RuntimeError(f"{self.path}: HTTP {r.status_code}")
            cr = r.headers.get("Content-Range", "")
            exp = f"bytes {off}-{off + nbytes - 1}/"
            if not cr.startswith(exp):
                raise RuntimeError(f"{self.path}: Content-Range {cr!r} does not start "
                                   f"with {exp!r}")
            total = cr[len(exp):]
            if not total.isdigit() or (self.size and int(total) != self.size):
                raise RuntimeError(f"{self.path}: Content-Range total {total!r} != "
                                   f"{self.size}")
            if len(r.content) != nbytes:
                raise RuntimeError(f"{self.path}: short body {len(r.content)}/{nbytes}")
            return r.content

        return self._retry(once, f"GET {self.path} bytes={off}-{off + nbytes - 1}")

    def read_ranges(self, ranges: Sequence[Tuple[int, int]],
                    workers: int = 1) -> List[bytes]:
        if len(ranges) <= 1 or workers <= 1:
            return [self.read_range(o, n) for o, n in ranges]
        with ThreadPoolExecutor(max_workers=min(workers, HTTP_WORKERS_MAX)) as ex:
            return list(ex.map(lambda r: self.read_range(r[0], r[1]), ranges))

    def close(self) -> None:
        pass


def _resolve_token(token: Optional[str]) -> Tuple[Optional[str], str]:
    """Token value or path-to-file.  Returns (value, label); the value is never logged."""
    if token:
        try:
            p = Path(token)
            if os.sep in token and p.is_file():
                return p.read_text().strip(), f"file:{p}"
        except OSError:
            pass
        return token.strip(), "constructor"
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        v = os.environ.get(env)
        if v:
            return v.strip(), f"env:{env}"
    p = Path.home() / ".cache" / "huggingface" / "token"
    if p.is_file():
        try:
            return p.read_text().strip(), f"file:{p}"
        except OSError:
            pass
    return None, "none"


# -------------------------------------------------------------------------- base class


class BaseSource:
    """Counters, name resolution and the shared row/column plumbing."""

    _readers: Dict[str, object] = {}

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.stats: dict = {
            "tensors_read": 0,
            "rows_read": 0,
            "bytes_read": 0,
            "http_requests": 0,
            "http_heads": 0,
            "http_ranges": 0,
            "local_ranges": 0,
            "get_many_calls": 0,
            "get_many_coalesced_ranges": 0,
            "get_many_spans_before_merge": 0,
            "seconds": 0.0,
            "calls": 0,
            "calls_dropped": 0,
            "retries": 0,
            "names_cached": False,
            "meta_cached": False,
            "ref_module": None,
            "ref_sha256": None,
            "incomplete_files": [],
        }
        self._calls: List[dict] = []
        self._max_call_records = 4096
        self._names: Optional[set] = None
        self._stems: Optional[Dict[str, str]] = None
        self._meta_cache: Dict[str, TensorMeta] = {}

    # -- accounting ---------------------------------------------------------
    def _record(self, op: str, name: str, nbytes: int, ranges: int, requests: int,
                seconds: float, extra: Optional[dict] = None) -> None:
        rec = {"op": op, "name": name, "bytes": int(nbytes), "ranges": int(ranges),
               "requests": int(requests), "seconds": round(float(seconds), 4)}
        if extra:
            rec.update(extra)
        with self._lock:
            st = self.stats
            st["calls"] += 1
            st["seconds"] = round(st["seconds"] + float(seconds), 4)
            if op in ("get", "get_many", "fetch"):
                st["tensors_read"] += 1
            if op == "get_rows":
                st["rows_read"] += 1
            if len(self._calls) < self._max_call_records:
                self._calls.append(rec)
            else:
                st["calls_dropped"] += 1

    def _add(self, key: str, value: int) -> None:
        with self._lock:
            self.stats[key] = self.stats.get(key, 0) + int(value)

    def _note_incomplete(self, fname: str) -> None:
        with self._lock:
            if fname not in self.stats["incomplete_files"]:
                self.stats["incomplete_files"].append(fname)

    def call_records(self, limit: int = 0, op: Optional[str] = None) -> List[dict]:
        recs = self._calls if not op else [r for r in self._calls if r["op"] == op]
        return recs[-limit:] if limit else list(recs)

    def reset_stats(self) -> None:
        with self._lock:
            for k in list(self.stats):
                if isinstance(self.stats[k], int):
                    self.stats[k] = 0 if k != "calls_cap" else self.stats[k]
            self.stats["names_cached"] = self._names is not None
            self._calls = []

    def stats_summary(self) -> dict:
        out = {k: (list(v) if isinstance(v, list) else v)
               for k, v in self.stats.items()}
        out["calls_recorded"] = len(self._calls)
        return out

    def close(self) -> None:
        for r in self._readers.values():
            try:
                r.close()
            except Exception:                           # noqa: BLE001
                pass

    # -- names / meta -------------------------------------------------------
    def names(self) -> set:
        if self._names is None:
            self._names = set(self._build_names())
            self._stems = {_canon(n): n for n in self._names}
            self.stats["names_cached"] = True
            self.stats["n_names"] = len(self._names)
        return self._names

    def _build_names(self) -> Iterable[str]:
        raise NotImplementedError

    def resolve(self, name: str) -> str:
        """Accept the artifact spelling, a ``.weight``-suffixed spelling, or the stem."""
        names = self.names()
        if name in names:
            return name
        cand = name + ".weight"
        if cand in names:
            return cand
        if name.endswith(".weight"):
            bare = name[:-len(".weight")]
            if bare in names:
                return bare
        got = (self._stems or {}).get(_canon(name))
        if got is None:
            raise KeyError(f"{type(self).__name__}: unknown tensor {name!r}")
        return got

    def meta(self, name: str) -> TensorMeta:
        name = self.resolve(name)
        m = self._meta_cache.get(name)
        if m is None:
            m = self._build_meta(name)
            self._meta_cache[name] = m
            self.stats["meta_cached"] = True
        return m

    def _build_meta(self, name: str) -> TensorMeta:
        raise NotImplementedError

    # -- access -------------------------------------------------------------
    def get(self, name: str) -> torch.Tensor:
        raise NotImplementedError

    def get_rows(self, name: str, row_ids) -> torch.Tensor:
        raise NotImplementedError

    def get_columns(self, name: str, col0: int, col1: int) -> torch.Tensor:
        raise NotImplementedError

    def get_many(self, names: Sequence[str], max_gap_bytes: int = 1 << 20) -> dict:
        raise NotImplementedError

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _ids(row_ids) -> List[int]:
        if torch.is_tensor(row_ids):
            return [int(x) for x in row_ids.reshape(-1).tolist()]
        return [int(x) for x in row_ids]

    @staticmethod
    def _order(ids: List[int]) -> Tuple[List[int], List[int]]:
        """Sorted unique ids + the position of each original id among them."""
        uniq = sorted(set(ids))
        pos = {v: i for i, v in enumerate(uniq)}
        return uniq, [pos[v] for v in ids]

    @staticmethod
    def _read_rows(reader, base: int, runs: Sequence[Tuple[int, int]], row_bytes: int,
                   dt: torch.dtype, tail_shape: Sequence[int], max_gap_bytes: int = 0,
                   workers: int = 1):
        """Read ``[start, stop)`` row runs and stack them; near-adjacent runs are merged
        into one read when ``max_gap_bytes`` > 0.  Returns ``(rows, spans, bytes_read)``."""
        ranges = [(base + r0 * row_bytes, (r1 - r0) * row_bytes) for r0, r1 in runs]
        spans = (_coalesce([(o, n, "") for o, n in ranges], max_gap_bytes)
                 if max_gap_bytes and len(ranges) > 1 else ranges)
        bufs = reader.read_ranges(spans, workers=workers)
        starts = _span_starts(spans)
        out = torch.empty((sum(r1 - r0 for r0, r1 in runs),) + tuple(tail_shape), dtype=dt)
        idx = 0
        for (r0, r1), (off, nbytes) in zip(runs, ranges):
            i = _slice_within(spans, off, nbytes, starts)
            if i < 0:
                raise RuntimeError(f"row range {off}+{nbytes} not covered by a span")
            soff, sn = spans[i]
            piece = memoryview(bufs[i])[off - soff:off - soff + nbytes]
            out[idx:idx + (r1 - r0)] = _tensor_from_buffer(piece, dt,
                                                           [r1 - r0] + list(tail_shape))
            idx += r1 - r0
        return out, len(spans), sum(n for _, n in spans)


# ------------------------------------------------------------------------ EXL3 source


def _weight_orig_from_inner(inner_f16, suh, svh, ref):
    """ref.weight_orig tail VERBATIM (same ops, same order, CPU, fp32): inner fp16
    [k,n] Hadamard-domain reconstruction -> original-basis fp32 weight [k,n].
    Copied from ~/B70-DOCS/scripts/tmp/fmt_screen_20261011.py:687."""
    w = inner_f16.float()
    k, n = w.shape
    h = ref.hadamard128(w.device, torch.float32)
    w = (h @ w.view(k // 128, 128, n)).view(k, n)
    w = w * suh.float().unsqueeze(1)
    w = (w.view(k, n // 128, 128) @ h).view(k, n)
    w = w * svh.float().unsqueeze(0)
    return w


class EXL3Source(BaseSource):
    """Local exllamav3 artifact; quantized groups decoded through the pinned ``ref``.

    root          artifact directory (``model.safetensors.index.json`` + shards)
    refsrc        directory holding the pinned ``ref.py`` (``sys.path.insert(0, refsrc)``)
    decode_cache  keep decoded fp32 blocks keyed by tensor (off by default; GBs per large
                  tensor, only useful for repeated small reads)
    """

    def __init__(self, root: str, refsrc: str, decode_cache: bool = False):
        super().__init__()
        self.root = Path(root)
        self.refsrc = str(refsrc)
        self.decode_cache = bool(decode_cache)
        self._ref = None
        self._wm: Optional[Dict[str, str]] = None
        self._files: Dict[str, _FileHeader] = {}
        self._readers = {}
        self._ngram: Optional[dict] = None
        self._block_cache: Dict[str, torch.Tensor] = {}
        self._workers = min(HTTP_WORKERS_MAX, 4)   # parallel local preads for row runs
        self.stats.update({"root": str(self.root), "refsrc": self.refsrc,
                           "ngram_file": str(self.root / NGRAM_FILE)})

    # -- index --------------------------------------------------------------
    @property
    def wm(self) -> Dict[str, str]:
        if self._wm is None:
            p = self.root / INDEX_NAME
            if not p.is_file():
                raise FileNotFoundError(f"{p}: EXL3 artifact without a weight index")
            wm = json.loads(p.read_text())["weight_map"]
            self._wm = {str(k): str(v) for k, v in wm.items()}
            self.stats["index_tensors"] = len(self._wm)
            self.stats["quantized_groups"] = sum(
                1 for k in self._wm if k.endswith(".trellis"))
        return self._wm

    def _file_header(self, fname: str) -> _FileHeader:
        fh = self._files.get(fname)
        if fh is None:
            fp = self.root / fname
            if not fp.is_file():
                raise FileNotFoundError(f"{fp}: shard referenced by the index is missing")
            fh = _read_file_header(fp)
            self._files[fname] = fh
            if not fh.complete:
                self._note_incomplete(fname)
        return fh

    def _ngram_info(self) -> Optional[dict]:
        """Header-only description of ``ngram_embedding.safetensors`` (outside the index)."""
        if self._ngram is not None:
            return self._ngram or None
        p = self.root / NGRAM_FILE
        if not p.is_file():
            self._ngram = {}
            self.stats["ngram_present"] = False
            return None
        fh = _read_file_header(p)
        self._files[NGRAM_FILE] = fh
        stems = sorted({k[:-len(".trellis")] for k in fh.names if k.endswith(".trellis")})
        plain = sorted(k for k in fh.names
                       if not any(k.endswith(s) for s in COMPONENT_SUFFIXES))
        info: dict = {"file": NGRAM_FILE, "keys": sorted(fh.names), "complete": fh.complete,
                      "file_size": fh.file_size, "data_end": fh.data_end,
                      "quantized_stems": stems, "plain_names": plain,
                      "metadata": dict(fh.metadata or {}),
                      "shape": None, "dtype": None, "table": None}
        if len(stems) == 1 and all(
                k in fh.names for k in (stems[0] + ".suh", stems[0] + ".svh")):
            info["layout"] = "exl3_trellis"
            info["table"] = stems[0]
            t = fh.names[info["table"] + ".trellis"]
            info["stored_shape"] = [int(x) for x in t["shape"]]
            # logical table: [rows, dim] == the trellis' (n, k)
            info["shape"] = [int(t["shape"][1]) * 16, int(t["shape"][0]) * 16]
            info["dtype"] = "BF16"
        elif stems:
            # a sharded quantized table: many <stem>.trellis keys, no sign vectors
            info["layout"] = "exl3_ngram_trellis"
            info["shard_stems"] = stems
            t = fh.names[stems[0] + ".trellis"]
            info["stored_shape"] = [int(x) for x in t["shape"]]
            info["shard_rows"] = int(t["shape"][0])
            info["row_words"] = int(t["shape"][1])
        elif plain:
            info["layout"] = "plain_rowmajor"
            two_d = [n for n in plain if len(fh.names[n]["shape"]) == 2]
            info["table"] = two_d[0] if len(two_d) == 1 else None
            if info["table"]:
                info["shape"] = [int(x) for x in fh.names[info["table"]]["shape"]]
                info["dtype"] = fh.names[info["table"]]["dtype"]
        else:
            info["layout"] = "unknown"
        self._ngram = info
        self.stats.update({"ngram_present": True, "ngram_layout": info["layout"],
                           "ngram_table": info["table"], "ngram_shape": info["shape"],
                           "ngram_dtype": info["dtype"], "ngram_complete": info["complete"],
                           "ngram_metadata": dict(info["metadata"]),
                           "ngram_shards": len(info.get("shard_stems") or []),
                           "ngram_shard_rows": info.get("shard_rows"),
                           "ngram_row_words": info.get("row_words")})
        if not fh.complete:
            self._note_incomplete(NGRAM_FILE)
        return info

    def _build_names(self) -> Iterable[str]:
        names = set()
        for k in self.wm:
            if k.endswith(".trellis"):
                names.add(k[:-len(".trellis")])
            elif not any(k.endswith(s) for s in COMPONENT_SUFFIXES):
                names.add(k)
        ngram = self._ngram_info()
        if ngram:
            # the n-gram file is outside the index: take every logical tensor it holds
            fh = self._file_header(NGRAM_FILE)
            for k in fh.names:
                if k.endswith(".trellis"):
                    names.add(k[:-len(".trellis")])
                    if ngram.get("layout") == "exl3_ngram_trellis":
                        # sharded table without sign vectors: the raw key is a usable
                        # spelling too (callers see it in the file)
                        names.add(k)
                elif not any(k.endswith(s) for s in COMPONENT_SUFFIXES):
                    names.add(k)
        return names

    def _ngram_shard(self, name: str) -> Optional[str]:
        """Shard stem when ``name`` addresses one shard of the sharded n-gram table."""
        info = self._ngram_info()
        if not info or info.get("layout") != "exl3_ngram_trellis":
            return None
        stem = name[:-len(".trellis")] if name.endswith(".trellis") else name
        if stem in (info.get("shard_stems") or ()):
            return stem
        return None

    def _ngram_row_decode_unsupported(self, name: str, stem: str) -> RuntimeError:
        info = self._ngram_info() or {}
        md = info.get("metadata") or {}
        return RuntimeError(
            f"{name}: the n-gram table of this artifact uses the custom layout "
            f"'exl3_ngram_trellis' (metadata={md}) — one row per table row, "
            f"{info.get('row_words')} int16 per row = 1 fp16 scale word + "
            f"{640} bits of state data for {md.get('row_dim', '?')} dims, K=4, "
            f"codebook mul1.  The bit packing / element order could NOT be resolved from "
            f"the pinned ref decoder (ref.py has no n-gram path) and no tested hypothesis "
            f"reproduced the official BF16 rows (see logs/sources_selftest.log section I).  "
            f"Refusing to return guessed values: read the rows from the official BF16 "
            f"checkpoint ('...ngram_embedding.shard_<N>.weight', BF16Source.get_rows) or "
            f"obtain the quantizer's decoder.  (stem={stem})")

    # -- meta ---------------------------------------------------------------
    def _group(self, name: str) -> Optional[dict]:
        """Quantized group descriptor for ``name``, or None when stored plain.

        Groups normally come from ``model.safetensors.index.json``; the n-gram table lives
        outside the index, so its own header is consulted for an EXL3 group as well.
        """
        wm = self.wm
        if name + ".trellis" in wm:
            extra = name + ".mul1" if name + ".mul1" in wm else name + ".mcg"
            if extra not in wm:
                raise KeyError(f"{name}: quantized group without a .mul1/.mcg flag")
            keys = {"trellis": name + ".trellis", "suh": name + ".suh",
                    "svh": name + ".svh", "extra": extra}
            shards = {wm[k] for k in keys.values()}
            if len(shards) != 1:
                raise ValueError(f"{name}: quantized group spans shards {sorted(shards)}")
            return {"keys": keys, "file": shards.pop()}
        info = self._ngram_info()
        if info and info.get("layout") == "exl3_trellis":
            fh = self._file_header(NGRAM_FILE)
            if name + ".trellis" in fh.names:
                extra = (name + ".mul1" if name + ".mul1" in fh.names
                         else name + ".mcg")
                keys = {"trellis": name + ".trellis", "suh": name + ".suh",
                        "svh": name + ".svh", "extra": extra}
                missing = [k for k in keys.values() if k not in fh.names]
                if missing:
                    raise KeyError(f"{NGRAM_FILE}: {name} group missing {missing}")
                return {"keys": keys, "file": NGRAM_FILE}
        return None

    def _ngram_hint(self, name: str) -> Optional[str]:
        p = self.root / NGRAM_FILE
        if "ngram" in name.lower() and not p.is_file():
            return (f"{name}: {p} is not present yet (24.44 GiB download still running); "
                    "ngram rows are unavailable until it lands")
        if "ngram" in name.lower() and p.is_file():
            info = self._ngram_info() or {}
            if not info.get("complete"):
                return (f"{name}: {p} is incomplete "
                        f"({info.get('file_size')} of "
                        f"{8 + (info.get('data_end') or 0)} bytes); refusing to read it")
        return None

    def resolve(self, name: str) -> str:
        """Accept the artifact spelling, a ``.weight``-suffixed spelling, or the stem."""
        try:
            return super().resolve(name)
        except KeyError:
            hint = self._ngram_hint(name)
            if hint:
                raise FileNotFoundError(hint) from None
            raise

    def _entry_for(self, name: str) -> Tuple[str, dict]:
        """(shard file, header entry) for a stored name (index first, then ngram file)."""
        wm = self.wm
        fname = wm.get(name)
        if fname is not None:
            fh = self._file_header(fname)
            if name not in fh.names:
                raise KeyError(f"{fname}: {name} missing from the shard header")
            return fname, fh.names[name]
        info = self._ngram_info()
        if info:
            fh = self._file_header(NGRAM_FILE)
            if name in fh.names:
                return NGRAM_FILE, fh.names[name]
        raise KeyError(f"{name}: neither in the index nor in {NGRAM_FILE}")

    def _build_meta(self, name: str) -> TensorMeta:
        stem = self._ngram_shard(name)
        if stem is not None:
            info = self._ngram_info() or {}
            md = info.get("metadata") or {}
            fh = self._file_header(NGRAM_FILE)
            t = fh.names[stem + ".trellis"]
            rows = int(info.get("shard_rows") or t["shape"][0])
            dim = int(md.get("row_dim") or 0)
            off = [int(x) for x in t["data_offsets"]]
            return TensorMeta(name=name, dtype="BF16", shape=[rows, dim], file=NGRAM_FILE,
                              offsets=off, nbytes=rows * dim * 2, quant=True,
                              ks=int(md.get("K") or 0), stored_nbytes=off[1] - off[0],
                              stored_shape=[int(x) for x in t["shape"]])
        group = self._group(name)
        if group is not None:
            fh = self._file_header(group["file"])
            tre = fh.names[group["keys"]["trellis"]]
            suh = fh.names[group["keys"]["suh"]]
            svh = fh.names[group["keys"]["svh"]]
            tshape = [int(x) for x in tre["shape"]]
            if len(tshape) != 3 or tshape[2] % 16:
                raise ValueError(f"{name}: unexpected trellis shape {tshape}")
            k_bits = tshape[2] // 16
            if not 2 <= k_bits <= 8:
                raise ValueError(f"{name}: invalid trellis K={k_bits}")
            k, n = tshape[0] * 16, tshape[1] * 16
            if int(suh["shape"][0]) != k or int(svh["shape"][0]) != n:
                raise ValueError(f"{name}: sign vector / trellis mismatch")
            payload = sum(int(e["data_offsets"][1]) - int(e["data_offsets"][0])
                          for e in (fh.names[kk] for kk in group["keys"].values()))
            return TensorMeta(name=name, dtype="BF16", shape=[n, k],
                              file=group["file"],
                              offsets=[int(x) for x in tre["data_offsets"]],
                              nbytes=n * k * SAFETENSORS_DTYPES["BF16"].itemsize,
                              quant=True, ks=k_bits, stored_nbytes=payload,
                              stored_shape=tshape)
        fname, entry = self._entry_for(name)
        fh = self._file_header(fname)
        return _meta_from_entry(name, entry, fname, fh, strict=False)

    # -- ref ----------------------------------------------------------------
    @property
    def ref(self):
        if self._ref is None:
            if self.refsrc not in sys.path:
                sys.path.insert(0, self.refsrc)
            import ref as _ref
            got = os.path.abspath(str(_ref.__file__))
            if not got.startswith(os.path.abspath(self.refsrc)):
                raise RuntimeError(
                    f"ref resolved to {got}, outside the pinned refsrc {self.refsrc}")
            with open(got, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            self._ref = _ref
            self.stats["ref_module"] = got
            self.stats["ref_sha256"] = digest
            print(f"[fn_sources] ref decoder {got} sha256={digest}", flush=True)
        return self._ref

    # -- decode -------------------------------------------------------------
    def _codebook(self, group: dict) -> Tuple[int, int, int]:
        """(cb, cb_value, K) selected exactly as the 27B harness does."""
        ref = self.ref
        fname = group["file"]
        e = self._file_header(fname).names[group["keys"]["extra"]]
        off, end = (int(x) for x in e["data_offsets"])
        raw = self._local_reader(fname).read_range(
            self._file_header(fname).data_start + off, end - off)
        v = torch.frombuffer(bytearray(raw), dtype=SAFETENSORS_DTYPES[e["dtype"]])
        cb_val = int(v.item()) if v.numel() == 1 else int(v[0])
        key = group["keys"]["extra"]
        cb = ref.CB_MUL1 if (key.endswith(".mul1") and cb_val) else (
            ref.CB_MCG if (key.endswith(".mcg") and cb_val) else ref.CB_3INST)
        tre = self._file_header(fname).names[group["keys"]["trellis"]]
        return cb, cb_val, int(tre["shape"][2]) // 16

    def _local_reader(self, fname: str) -> _LocalFile:
        r = self._readers.get(fname)
        if r is None:
            r = _LocalFile(self.root / fname)
            self._readers[fname] = r
        return r

    def _read_trellis(self, fname: str, tre_e: dict, c0i: int, c1i: int):
        tshape = [int(x) for x in tre_e["shape"]]
        dt = SAFETENSORS_DTYPES[tre_e["dtype"]]
        tile_bytes = tshape[2] * dt.itemsize
        fh = self._file_header(fname)
        base = fh.data_start + int(tre_e["data_offsets"][0])
        reader = self._local_reader(fname)
        if c0i == 0 and c1i == tshape[1]:
            buf = reader.read_range(base, int(tre_e["data_offsets"][1]) -
                                    int(tre_e["data_offsets"][0]))
            self._add("local_ranges", 1)
            self._add("bytes_read", len(buf))
            return _tensor_from_buffer(buf, dt, tshape)
        # trellis[i, j, :] is contiguous in j: one read per i row
        want = (c1i - c0i) * tile_bytes
        row_stride = tshape[1] * tile_bytes
        parts = []
        for i in range(tshape[0]):
            off = base + i * row_stride + c0i * tile_bytes
            parts.append(reader.read_range(off, want))
        self._add("local_ranges", len(parts))
        self._add("bytes_read", want * len(parts))
        t = torch.cat([_tensor_from_buffer(p, dt, [c1i - c0i, tshape[2]]) for p in parts])
        return t.reshape(tshape[0], c1i - c0i, tshape[2]).contiguous()

    def _decode_cols(self, name: str, n0: int, n1: int) -> torch.Tensor:
        """Decode output columns ``[n0, n1)`` -> bf16 ``[n1-n0, k]``."""
        group = self._group(name)
        if group is None:
            return self.get(name)[n0:n1]
        if self.decode_cache:
            full = self._block_cache.get(name)
            if full is None:
                full = self._decode_cols_uncached(name, 0, self.meta(name).shape[0])
                self._block_cache[name] = full
            return full[n0:n1]
        return self._decode_cols_uncached(name, n0, n1)

    def _decode_cols_uncached(self, name: str, n0: int, n1: int) -> torch.Tensor:
        t0 = time.perf_counter()
        ref = self.ref
        group = self._group(name)
        assert group is not None
        fname = group["file"]
        fh = self._file_header(fname)
        tre_e = fh.names[group["keys"]["trellis"]]
        suh_e = fh.names[group["keys"]["suh"]]
        svh_e = fh.names[group["keys"]["svh"]]
        tshape = [int(x) for x in tre_e["shape"]]
        k, n = tshape[0] * 16, tshape[1] * 16
        if k % 128 or n % 128:
            raise ValueError(f"{name}: k={k} n={n} is not Hadamard128 aligned")
        if not (0 <= n0 < n1 <= n):
            raise IndexError(f"{name}: column range {n0}:{n1} outside n={n}")
        cb, cb_val, k_bits = self._codebook(group)
        b0, b1 = (n0 // 128) * 128, ((n1 + 127) // 128) * 128
        c0i, c1i = b0 // 16, b1 // 16
        reader = self._local_reader(fname)
        tre = self._read_trellis(fname, tre_e, c0i, c1i)
        suh_off = [int(x) for x in suh_e["data_offsets"]]
        suh = _tensor_from_buffer(
            reader.read_range(fh.data_start + suh_off[0], suh_off[1] - suh_off[0]),
            SAFETENSORS_DTYPES[suh_e["dtype"]], suh_e["shape"])
        sv_dt = SAFETENSORS_DTYPES[svh_e["dtype"]]
        sv_off = fh.data_start + int(svh_e["data_offsets"][0]) + b0 * sv_dt.itemsize
        svh = _tensor_from_buffer(reader.read_range(sv_off, (b1 - b0) * sv_dt.itemsize),
                                  sv_dt, [b1 - b0])
        self._add("local_ranges", 2)
        self._add("bytes_read", (suh_off[1] - suh_off[0]) + (b1 - b0) * sv_dt.itemsize)
        inner = ref.reconstruct_inner(tre, k_bits, cb)
        w = _weight_orig_from_inner(inner, suh, svh, ref)      # (k, block) fp32
        del inner, tre
        out = w.t().contiguous().to(torch.bfloat16)            # (block, k)
        del w
        if b0 != n0 or b1 != n1:
            out = out[n0 - b0:n1 - b0].contiguous()
        self._record("decode", name, 0, 0, 0, time.perf_counter() - t0,
                     {"k_bits": k_bits, "codebook": cb, "cb_value": cb_val,
                      "cols": [int(n0), int(n1)], "shape": list(out.shape)})
        return out

    def _plain(self, name: str) -> torch.Tensor:
        fname, e = self._entry_for(name)
        fh = self._file_header(fname)
        off, end = (int(x) for x in e["data_offsets"])
        buf = self._local_reader(fname).read_range(fh.data_start + off, end - off)
        if len(buf) != end - off:
            raise RuntimeError(f"{name}: short read {len(buf)}/{end - off} bytes")
        self._add("local_ranges", 1)
        self._add("bytes_read", end - off)
        t = _tensor_from_buffer(buf, SAFETENSORS_DTYPES[e["dtype"]], e["shape"])
        if int(t.numel() * t.element_size()) != end - off:
            raise RuntimeError(f"{name}: {t.numel()} elements != {end - off} bytes")
        # floating-point tensors come back as bf16 (harness convention); integer/bool
        # tensors (ngram bookkeeping: offsets, vocab sizes, multipliers) stay as stored
        return t.to(torch.bfloat16) if t.is_floating_point() else t

    def _plain_rows(self, name: str, ids: List[int], max_gap_bytes: int = 0
                    ) -> Tuple[torch.Tensor, str]:
        """Contiguous multi-row read for a plain tensor; returns (rows, stored dtype)."""
        fname, e = self._entry_for(name)
        fh = self._file_header(fname)
        dt = SAFETENSORS_DTYPES[e["dtype"]]
        shape = [int(x) for x in e["shape"]]
        row_bytes = _prod(shape[1:]) * dt.itemsize
        base = fh.data_start + int(e["data_offsets"][0])
        out, spans, nbytes = self._read_rows(
            self._local_reader(fname), base, _runs(ids), row_bytes, dt, shape[1:],
            max_gap_bytes=max_gap_bytes, workers=self._workers)
        self._add("local_ranges", spans)
        self._add("bytes_read", nbytes)
        return out, e["dtype"]

    # -- public access ------------------------------------------------------
    def get(self, name: str) -> torch.Tensor:
        name = self.resolve(name)
        stem = self._ngram_shard(name)
        if stem is not None:
            raise self._ngram_row_decode_unsupported(name, stem)
        t0 = time.perf_counter()
        if self._group(name) is None:
            t = self._plain(name)
        else:
            t = self._decode_cols(name, 0, self.meta(name).shape[0])
        self._record("get", name, int(t.numel() * t.element_size()), 0, 0,
                     time.perf_counter() - t0,
                     {"shape": list(t.shape), "dtype": str(t.dtype)})
        return t

    def _ngram_rows_upstream(self, stem: str, ids: List[int]) -> torch.Tensor:
        """Rows of the custom ``exl3_ngram_trellis`` table, decoded as the engine decodes them.

        Layout and semantics are the vendored exllamav3's: each row is (1 + 10K) little-endian
        int16 words, word 0 is the fp16 row scale, the rest is the tail-biting mul1 ring;
        ``row = codebook(state) * scale + head_bias[head]``
        (modules/quant/exl3_lib/ngram_codec.py, the torch mirror of the ngram_dequant kernel).
        This reader was verified bit-exact against the extension's own ``ngram_dequant`` on real
        artifact rows (logs/ngram_dequant_oracle.json); it is NOT fitted to the official BF16
        table, and the deviation from that table is reported separately as a diagnostic
        (logs/ngram_realbyte_oracle.json: rfn 0.9927-0.9932, std ratio ~2^7).
        """
        import importlib.util
        codec_path = os.environ.get("FN_NGRAM_CODEC") or (
            "/data/nvme/engines/exl3xpu-mixedk/flashnext/build/exllamav3/exllamav3/"
            "modules/quant/exl3_lib/ngram_codec.py")
        spec = importlib.util.spec_from_file_location("fn_ngram_codec", codec_path)
        codec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codec)
        info = self._ngram_info() or {}
        md = info.get("metadata") or {}
        K = int(md.get("K") or 4)
        shard_rows = int(md.get("shard_rows") or 0)
        row_dim = int(md.get("row_dim") or 0)
        if not shard_rows or not row_dim:
            raise RuntimeError(f"{stem}: n-gram metadata lacks shard_rows/row_dim ({md})")
        fname, e = self._entry_for(stem + ".trellis")
        fh = self._file_header(fname)
        dt = SAFETENSORS_DTYPES[e["dtype"]]
        words = _prod([int(x) for x in e["shape"][1:]])
        base = fh.data_start + int(e["data_offsets"][0])
        # per-head bias + offsets live beside the shards in the same file
        prefix = stem[: -len(stem.split(".")[-1])]
        try:
            _bf, be = self._entry_for(prefix + "head_bias")
            _of, oe = self._entry_for(prefix + "head_offsets")
            hb = _tensor_from_buffer(self._local_reader(_bf).read_range(
                self._file_header(_bf).data_start + int(be["data_offsets"][0]),
                _prod([int(x) for x in be["shape"]]) * SAFETENSORS_DTYPES[be["dtype"]].itemsize),
                SAFETENSORS_DTYPES[be["dtype"]], [int(x) for x in be["shape"]]).float()
            ho = _tensor_from_buffer(self._local_reader(_of).read_range(
                self._file_header(_of).data_start + int(oe["data_offsets"][0]),
                _prod([int(x) for x in oe["shape"]]) * SAFETENSORS_DTYPES[oe["dtype"]].itemsize),
                SAFETENSORS_DTYPES[oe["dtype"]], [int(x) for x in oe["shape"]]).long()
        except KeyError:
            hb, ho = None, None
        cb = codec.mul1_codebook(torch.device("cpu"))
        row_bytes = words * dt.itemsize
        out = torch.empty((len(ids), row_dim), dtype = torch.bfloat16)
        for i, gid in enumerate(ids):
            shard, local = divmod(int(gid), shard_rows)
            f2, e2 = self._entry_for(f"{prefix}shard_{shard}.trellis")
            base2 = self._file_header(f2).data_start + int(e2["data_offsets"][0])
            buf = self._local_reader(f2).read_range(base2 + local * row_bytes, row_bytes)
            self._add("local_ranges", 1)
            self._add("bytes_read", len(buf))
            packed = torch.frombuffer(bytearray(buf), dtype = torch.int16).unsqueeze(0)
            if hb is not None and ho is not None:
                h = int(torch.searchsorted(ho, torch.tensor([int(gid)]), right = True)) - 1
                h = max(0, min(h, int(hb.shape[0]) - 1))
                dec = codec.dequant_rows(packed, K, cb, hb[h].float()[None, :])
            else:
                dec = codec.dequant_rows(packed, K, cb, None)
            out[i] = dec.to(torch.float32).reshape(-1).to(torch.bfloat16)
        self._record("ngram_rows", stem, int(out.numel() * out.element_size()), 0, len(ids),
                     0.0,
                     {"decode": "upstream_consistent (ngram_codec.dequant_rows)",
                      "rows": len(ids)})
        return out

    def get_rows(self, name: str, row_ids, max_gap_bytes: int = 0) -> torch.Tensor:
        """Gather output rows (axis 0), row-major; 128-block granular for quantized.

        ``max_gap_bytes`` > 0 merges near-adjacent row runs into one read (useful over a
        network); it never changes the returned rows."""
        name = self.resolve(name)
        stem = self._ngram_shard(name)
        if stem is not None:
            # Default and only path for this artifact: the custom 'exl3_ngram_trellis' table is
            # decoded through the vendored codec, which is the same algorithm as the engine's
            # ngram_dequant kernel (verified bit-exact on real artifact rows,
            # logs/ngram_dequant_oracle.json). It is not an env-selected variant and not fitted
            # to the official BF16 table; the deviation from that table is reported separately
            # (logs/ngram_realbyte_oracle.json). Provenance is recorded per call.
            return self._ngram_rows_upstream(stem, self._ids(row_ids))
        t0 = time.perf_counter()
        ids = self._ids(row_ids)
        meta = self.meta(name)
        if ids and (min(ids) < 0 or max(ids) >= meta.shape[0]):
            bad = min(ids) if min(ids) < 0 else max(ids)
            raise IndexError(f"{name}: row id {bad} outside 0..{meta.shape[0]}")
        if not ids:
            return torch.empty((0,) + tuple(meta.shape[1:]), dtype=torch.bfloat16)
        uniq, pos = self._order(ids)
        ordered = pos == list(range(len(ids)))      # ascending, no duplicates
        contig = ordered and len(uniq) == uniq[-1] - uniq[0] + 1   # ... and gapless
        if self._group(name) is None:
            out, _dt = self._plain_rows(name, uniq, max_gap_bytes=max_gap_bytes)
            if out.is_floating_point():
                out = out.to(torch.bfloat16)
        else:
            lo, hi = uniq[0], uniq[-1] + 1
            out = self._decode_cols(name, lo, hi)
            if not contig:
                out = out[[i - lo for i in uniq]]
        if not ordered:
            out = out[[*pos]].contiguous()
        self._record("get_rows", name, int(out.numel() * out.element_size()), 0, 0,
                     time.perf_counter() - t0,
                     {"rows": len(ids), "unique_rows": len(uniq)})
        return out

    def get_columns(self, name: str, col0: int, col1: int) -> torch.Tensor:
        """Aligned output-axis block; equals ``get(name)[col0:col1]``."""
        name = self.resolve(name)
        t0 = time.perf_counter()
        meta = self.meta(name)
        col0, col1 = int(col0), int(col1)
        if col0 < 0 or col1 > meta.shape[0] or col1 <= col0:
            raise IndexError(f"{name}: column block {col0}:{col1} outside {meta.shape[0]}")
        stem = self._ngram_shard(name)
        if stem is not None:
            raise self._ngram_row_decode_unsupported(name, stem)
        if self._group(name) is not None:
            out = self._decode_cols(name, col0, col1)
        else:
            fname, e = self._entry_for(name)
            fh = self._file_header(fname)
            dt = SAFETENSORS_DTYPES[e["dtype"]]
            shape = [int(x) for x in e["shape"]]
            row_bytes = _prod(shape[1:]) * dt.itemsize
            off = fh.data_start + int(e["data_offsets"][0]) + col0 * row_bytes
            buf = self._local_reader(fname).read_range(off, (col1 - col0) * row_bytes)
            self._add("local_ranges", 1)
            self._add("bytes_read", len(buf))
            out = _tensor_from_buffer(buf, dt, [col1 - col0] + shape[1:]).to(
                torch.bfloat16)
        self._record("get_columns", name, int(out.numel() * out.element_size()), 1, 0,
                     time.perf_counter() - t0, {"cols": [col0, col1]})
        return out

    def get_many(self, names: Sequence[str], max_gap_bytes: int = 1 << 20) -> dict:
        """Coalesced multi-tensor read: adjacent/close stored ranges become one read."""
        t0 = time.perf_counter()
        resolved = [self.resolve(n) for n in names]
        self._add("get_many_calls", 1)
        per_file: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
        entries: Dict[str, Tuple[str, dict, Optional[dict]]] = {}
        for name in resolved:
            stem = self._ngram_shard(name)
            if stem is not None:
                raise self._ngram_row_decode_unsupported(name, stem)
            group = self._group(name)
            if group is None:
                fname, e = self._entry_for(name)
                entries[name] = (fname, e, None)
                fh = self._file_header(fname)
                per_file[fname].append((fh.data_start + int(e["data_offsets"][0]),
                                        int(e["data_offsets"][1]) -
                                        int(e["data_offsets"][0]), name))
                continue
            fname = group["file"]
            fh = self._file_header(fname)
            entries[name] = (fname, fh.names[group["keys"]["trellis"]], group)
            for key in group["keys"].values():
                e = fh.names[key]
                per_file[fname].append((fh.data_start + int(e["data_offsets"][0]),
                                        int(e["data_offsets"][1]) -
                                        int(e["data_offsets"][0]), key))
        before = sum(len(v) for v in per_file.values())
        spans_read, bytes_read = 0, 0
        payload: Dict[Tuple[str, int, int], memoryview] = {}
        for fname, rngs in per_file.items():
            spans = _coalesce(rngs, max_gap_bytes)
            bufs = self._local_reader(fname).read_ranges(spans, workers=self._workers)
            spans_read += len(spans)
            bytes_read += sum(n for _, n in spans)
            self._add("local_ranges", len(spans))
            starts = _span_starts(spans)
            for off, nbytes, _key in rngs:
                i = _slice_within(spans, off, nbytes, starts)
                if i < 0:
                    raise RuntimeError(f"{fname}: range {off}+{nbytes} uncovered")
                soff, sn = spans[i]
                payload[(fname, off, nbytes)] = memoryview(bufs[i])[off - soff:
                                                                     off - soff + nbytes]
        out: Dict[str, torch.Tensor] = {}
        for name in resolved:
            fname, e, group = entries[name]
            fh = self._file_header(fname)
            if group is None:
                off = fh.data_start + int(e["data_offsets"][0])
                nbytes = int(e["data_offsets"][1]) - int(e["data_offsets"][0])
                tv = _tensor_from_buffer(payload[(fname, off, nbytes)],
                                         SAFETENSORS_DTYPES[e["dtype"]], e["shape"])
                out[name] = tv.to(torch.bfloat16) if tv.is_floating_point() else tv
                continue
            if self.decode_cache and name in self._block_cache:
                out[name] = self._block_cache[name]
                continue
            parts = {}
            for part, key in group["keys"].items():
                e2 = fh.names[key]
                off = fh.data_start + int(e2["data_offsets"][0])
                nbytes = int(e2["data_offsets"][1]) - int(e2["data_offsets"][0])
                parts[part] = _tensor_from_buffer(payload[(fname, off, nbytes)],
                                                  SAFETENSORS_DTYPES[e2["dtype"]],
                                                  e2["shape"])
            cb, cb_val, k_bits = self._codebook(group)
            inner = self.ref.reconstruct_inner(parts["trellis"], k_bits, cb)
            w = _weight_orig_from_inner(inner, parts["suh"], parts["svh"], self.ref)
            del inner
            out[name] = w.t().contiguous().to(torch.bfloat16)
            del w
        self._add("get_many_coalesced_ranges", spans_read)
        self._add("get_many_spans_before_merge", before)
        self._add("bytes_read", bytes_read)
        self._record("get_many", f"<{len(resolved)} names>",
                     sum(int(t.numel() * t.element_size()) for t in out.values()),
                     spans_read, 0, time.perf_counter() - t0,
                     {"names": len(resolved), "spans": spans_read, "spans_before": before,
                      "bytes_fetched": bytes_read})
        return out

    _workers = 1


# ------------------------------------------------------------------------ BF16 source


class BF16Source(BaseSource):
    """Official (or local-copy) bf16 checkpoint.

    local_dir  directory of safetensors shards (+ optional index json)
    repo/rev   Hugging Face repo + revision; every tensor byte range is fetched with a
               ranged HTTP GET, the per-file CDN URL resolved once and cached
    token      raw token value *or* a path to a file holding it (read once, in the
               constructor; never logged).  Falls back to HF_TOKEN /
               ~/.cache/huggingface/token
    prefetch   concurrent ranged-request threads (0 -> default), clamped to 4
    """

    def __init__(self, local_dir: Optional[str] = None, repo: Optional[str] = None,
                 rev: Optional[str] = None, token: Optional[str] = None,
                 prefetch: int = 0):
        super().__init__()
        if not local_dir and not repo:
            raise ValueError("BF16Source needs local_dir or repo")
        self.local_dir = Path(local_dir) if local_dir else None
        self.repo = repo
        self.rev = rev
        self.prefetch = max(1, min(int(prefetch or HTTP_WORKERS_MAX), HTTP_WORKERS_MAX))
        self._workers = self.prefetch
        self._token, token_source = _resolve_token(token)
        self._readers = {}
        self._manifest_cache: Optional[Dict[str, str]] = None
        self._remote_files: Dict[str, _HTTPFile] = {}
        self._meta_raw: Dict[str, object] = {}
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_lock = threading.Lock()
        self._fh_cache: Dict[str, _FileHeader] = {}
        self.stats.update({"mode": "local" if self.local_dir else "http",
                           "repo": repo or "", "rev": rev or "",
                           "local_dir": str(self.local_dir or ""),
                           "prefetch_workers": self._workers,
                           "token_present": bool(self._token),
                           "token_source": token_source})

    # -- files --------------------------------------------------------------
    def _reader(self, fname: str):
        r = self._readers.get(fname)
        if r is None:
            if self.local_dir is not None:
                r = _LocalFile(self.local_dir / fname)
            else:
                r = self._remote_files.get(fname)
                if r is None:
                    r = _HTTPFile(self.repo, self.rev, fname, self._token, self.stats,
                                  self._lock, retries=max(1, int(os.environ.get("FN_HTTP_RETRIES", "5"))),
                                  metadata=self._meta_raw.get(fname))
                    r.pool = self._pool_get()
                    self._remote_files[fname] = r
            self._readers[fname] = r
        return r

    def _pool_get(self) -> ThreadPoolExecutor:
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=self._workers,
                                                thread_name_prefix="fnfetch")
            return self._pool

    def _read_ranges(self, fname: str, ranges: Sequence[Tuple[int, int]]) -> List[bytes]:
        return self._reader(fname).read_ranges(ranges, workers=self._workers)

    def _fh(self, fname: str) -> _FileHeader:
        fh = self._fh_cache.get(fname)
        if fh is not None:
            return fh
        if self.local_dir is not None:
            fp = self.local_dir / fname
            if not fp.is_file():
                raise FileNotFoundError(f"{fp}: shard referenced by the index is missing")
            fh = _read_file_header(fp)
            if not fh.complete:
                self._note_incomplete(fname)
        else:
            reader = self._reader(fname)
            raw = reader.read_range(0, 8)
            n = int.from_bytes(raw, "little")
            if n <= 0 or n > HEADER_MAX_BYTES:
                raise ValueError(f"{fname}: implausible safetensors header length {n}")
            body = reader.read_range(8, n)
            hdr, n = _parse_header_bytes(raw + body, fname)
            fh = _header_from_parsed(hdr, n, Path(fname), reader.size, remote=True)
        self._fh_cache[fname] = fh
        return fh

    def _manifest(self) -> Dict[str, str]:
        """name -> shard file (index json when present, else every local shard header)."""
        if self._manifest_cache is not None:
            return self._manifest_cache
        man: Dict[str, str] = {}
        raw: Optional[bytes] = None
        if self.local_dir is not None:
            p = self.local_dir / INDEX_NAME
            if p.is_file():
                raw = p.read_bytes()
            else:
                for f in sorted(self.local_dir.glob("*.safetensors")):
                    for name in self._fh(f.name).names:
                        if name in man:
                            raise ValueError(f"{name}: in {man[name]} and {f.name}")
                        man[name] = f.name
        else:
            raw = self.fetch_file_bytes(INDEX_NAME)
        if raw is not None:
            wm = json.loads(raw.decode("utf-8"))["weight_map"]
            man = {str(k): str(v) for k, v in wm.items()}
        self._manifest_cache = man
        self.stats["manifest_tensors"] = len(man)
        return man

    def _build_names(self) -> Iterable[str]:
        return set(self._manifest())

    def _lookup(self, name: str) -> Tuple[str, dict]:
        man = self._manifest()
        fname = man.get(name)
        if fname is None:
            raise KeyError(f"{type(self).__name__}: unknown tensor {name!r}")
        fh = self._fh(fname)
        if name not in fh.names:
            raise ValueError(f"{fname}: {name} missing from the shard header")
        return fname, fh.names[name]

    def _build_meta(self, name: str) -> TensorMeta:
        fname, e = self._lookup(name)
        return _meta_from_entry(name, e, fname, self._fh(fname), strict=False)

    # -- public access ------------------------------------------------------
    def get(self, name: str) -> torch.Tensor:
        name = self.resolve(name)
        t0 = time.perf_counter()
        fname, e = self._lookup(name)
        fh = self._fh(fname)
        off, end = (int(x) for x in e["data_offsets"])
        buf = self._read_ranges(fname, [(fh.data_start + off, end - off)])[0]
        if len(buf) != end - off:
            raise RuntimeError(f"{name}: short read {len(buf)}/{end - off}")
        t = _tensor_from_buffer(buf, SAFETENSORS_DTYPES[e["dtype"]], e["shape"])
        self._add("bytes_read", end - off)
        if self.local_dir is not None:
            self._add("local_ranges", 1)
        self._record("get", name, int(t.numel() * t.element_size()),
                     1, 1 if self.local_dir is None else 0,
                     time.perf_counter() - t0,
                     {"file": fname, "shape": list(t.shape), "dtype": str(t.dtype)})
        return t

    def get_rows(self, name: str, row_ids, max_gap_bytes: int = 0) -> torch.Tensor:
        """Gather rows along axis 0 (row-major).  ``max_gap_bytes`` > 0 merges
        near-adjacent row runs into one read; the returned rows never change."""
        name = self.resolve(name)
        t0 = time.perf_counter()
        ids = self._ids(row_ids)
        meta = self.meta(name)
        if not ids:
            return torch.empty((0,) + tuple(meta.shape[1:]),
                               dtype=SAFETENSORS_DTYPES[meta.dtype])
        if min(ids) < 0 or max(ids) >= meta.shape[0]:
            bad = min(ids) if min(ids) < 0 else max(ids)
            raise IndexError(f"{name}: row id {bad} outside 0..{meta.shape[0]}")
        uniq, pos = self._order(ids)
        ordered = pos == list(range(len(ids)))          # already ascending, no duplicates
        fname, e = self._lookup(name)
        fh = self._fh(fname)
        dt = SAFETENSORS_DTYPES[e["dtype"]]
        shape = [int(x) for x in e["shape"]]
        row_bytes = _prod(shape[1:]) * dt.itemsize
        base = fh.data_start + int(e["data_offsets"][0])
        out, spans, nbytes = self._read_rows(
            self._reader(fname), base, _runs(uniq), row_bytes, dt, shape[1:],
            max_gap_bytes=max_gap_bytes, workers=self._workers)
        if not ordered:
            out = out[torch.tensor(pos, dtype=torch.long)].contiguous()
        self._add("bytes_read", nbytes)
        if self.local_dir is not None:
            self._add("local_ranges", spans)
        self._record("get_rows", name, int(out.numel() * out.element_size()),
                     spans, spans if self.local_dir is None else 0,
                     time.perf_counter() - t0,
                     {"rows": len(ids), "unique_rows": len(uniq), "spans": spans,
                      "file": fname})
        return out

    def get_columns(self, name: str, col0: int, col1: int) -> torch.Tensor:
        """Output-axis block ``get(name)[col0:col1]`` (HF/safetensors layout)."""
        name = self.resolve(name)
        t0 = time.perf_counter()
        meta = self.meta(name)
        col0, col1 = int(col0), int(col1)
        if col0 < 0 or col1 > meta.shape[0] or col1 <= col0:
            raise IndexError(f"{name}: column block {col0}:{col1} outside {meta.shape[0]}")
        fname, e = self._lookup(name)
        fh = self._fh(fname)
        dt = SAFETENSORS_DTYPES[e["dtype"]]
        shape = [int(x) for x in e["shape"]]
        row_bytes = _prod(shape[1:]) * dt.itemsize
        off = fh.data_start + int(e["data_offsets"][0]) + col0 * row_bytes
        buf = self._read_ranges(fname, [(off, (col1 - col0) * row_bytes)])[0]
        if len(buf) != (col1 - col0) * row_bytes:
            raise RuntimeError(f"{name}: short column read {len(buf)}")
        out = _tensor_from_buffer(buf, dt, [col1 - col0] + shape[1:])
        self._add("bytes_read", len(buf))
        if self.local_dir is not None:
            self._add("local_ranges", 1)
        self._record("get_columns", name, int(out.numel() * out.element_size()), 1,
                     1 if self.local_dir is None else 0, time.perf_counter() - t0,
                     {"cols": [col0, col1], "file": fname})
        return out

    def get_many(self, names: Sequence[str], max_gap_bytes: int = 1 << 20) -> dict:
        """Coalesced multi-tensor read: adjacent/close byte ranges become ONE request."""
        t0 = time.perf_counter()
        resolved = [self.resolve(n) for n in names]
        self._add("get_many_calls", 1)
        per_file: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
        for name in resolved:
            fname, e = self._lookup(name)
            fh = self._fh(fname)
            per_file[fname].append((fh.data_start + int(e["data_offsets"][0]),
                                    int(e["data_offsets"][1]) -
                                    int(e["data_offsets"][0]), name))
        before = sum(len(v) for v in per_file.values())
        spans_read, bytes_read = 0, 0
        payload: Dict[Tuple[str, int, int], memoryview] = {}
        for fname, rngs in per_file.items():
            spans = _coalesce(rngs, max_gap_bytes)
            bufs = self._read_ranges(fname, spans)
            spans_read += len(spans)
            bytes_read += sum(n for _, n in spans)
            if self.local_dir is not None:
                self._add("local_ranges", len(spans))
            starts = _span_starts(spans)
            for off, nbytes, _name in rngs:
                i = _slice_within(spans, off, nbytes, starts)
                if i < 0:
                    raise RuntimeError(f"{fname}: range {off}+{nbytes} uncovered")
                soff, sn = spans[i]
                payload[(fname, off, nbytes)] = memoryview(bufs[i])[off - soff:
                                                                     off - soff + nbytes]
        out: Dict[str, torch.Tensor] = {}
        for name in resolved:
            fname, e = self._lookup(name)
            fh = self._fh(fname)
            off = fh.data_start + int(e["data_offsets"][0])
            nbytes = int(e["data_offsets"][1]) - int(e["data_offsets"][0])
            out[name] = _tensor_from_buffer(payload[(fname, off, nbytes)],
                                            SAFETENSORS_DTYPES[e["dtype"]], e["shape"])
        self._add("get_many_coalesced_ranges", spans_read)
        self._add("get_many_spans_before_merge", before)
        self._add("bytes_read", bytes_read)
        self._record("get_many", f"<{len(resolved)} names>",
                     sum(int(t.numel() * t.element_size()) for t in out.values()),
                     spans_read, spans_read if self.local_dir is None else 0,
                     time.perf_counter() - t0,
                     {"names": len(resolved), "spans": spans_read, "spans_before": before,
                      "bytes_fetched": bytes_read})
        return out

    # -- raw file helpers ---------------------------------------------------
    def file_metadata(self, path: str) -> dict:
        """Size / LFS sha256 / resolved CDN URL for a repo (or local) file."""
        if self.local_dir is not None:
            p = self.local_dir / path
            body = p.read_bytes()
            return {"local": True, "size": len(body),
                    "sha256": hashlib.sha256(body).hexdigest()}
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        md = self._meta_raw.get(path)
        if md is None:
            self._add("http_requests", 1)
            self._add("http_heads", 1)
            md = get_hf_file_metadata(hf_hub_url(self.repo, path, revision=self.rev),
                                      token=self._token)
            self._meta_raw[path] = md
        lfs = getattr(md, "lfs", None)
        return {"local": False, "size": int(getattr(md, "size", 0) or 0),
                "sha256": getattr(lfs, "sha256", None) if lfs is not None else None,
                "location": getattr(md, "location", None),
                "etag": getattr(md, "etag", None)}

    def fetch_file_bytes(self, path: str, sha256_check: bool = False) -> bytes:
        """Fetch a whole (small) file with ranged reads; verify length and, if asked, sha256."""
        t0 = time.perf_counter()
        md = self.file_metadata(path)
        size = int(md["size"])
        if size <= 0:
            raise RuntimeError(f"{path}: no declared size")
        step = 8 << 20
        parts = [self._read_ranges(path, [(off, min(step, size - off))])[0]
                 for off in range(0, size, step)]
        body = b"".join(parts)
        if len(body) != size:
            raise RuntimeError(f"{path}: {len(body)} bytes != declared {size}")
        digest = hashlib.sha256(body).hexdigest()
        self._record("fetch", path, size, len(parts),
                     len(parts) if self.local_dir is None else 0,
                     time.perf_counter() - t0,
                     {"sha256": digest, "declared_size": size})
        self._add("bytes_read", size)
        if sha256_check and md.get("sha256") and md["sha256"] != digest:
            raise RuntimeError(f"{path}: sha256 {digest} != hub {md['sha256']}")
        return body

    def close(self) -> None:
        super().close()
        with self._pool_lock:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None
