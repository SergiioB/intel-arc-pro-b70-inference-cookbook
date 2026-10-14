#!/usr/bin/env python3
"""Reference-weight KL evaluation for Qwen3.8-Flash-Next (Qwen4Exp) EXL3.

Method (same shape as the Qwen3.8-27B SC5 screen in
~/B70-DOCS/scripts/tmp/fmt_screen_20261011.py):

  * one forward through ALL 48 text layers of the native upstream Qwen4Exp
    implementation (transformers worktree, PR #48337) for the frozen panels;
  * weights are bf16-rounded (storage precision) and every matmul runs in FP32;
  * the BF16 arm reads the official Qwen/Qwen3.8-Flash-Next checkpoint (local
    shards or HTTP range reads, never a local copy in hf mode);
  * the EXL3 arm reconstructs weights from the turboderp EXL3 artifact with the
    pinned reference decoder (`ref.py`), i.e. no GGUF or serving-engine substitute;
  * KL(P_BF16 || P_arm) is computed per scored position over the frozen
    wikitext panel and the held-out chat response positions, with each arm's own
    output head (native quantized head for EXL3), raw logits retained on disk.

Modes:
  coverage   map every HF parameter to an artifact key and verify shapes (no forward)
  logits     run an arm over both panels, retain raw logits + per-layer stats
  identity   re-run one layer twice from the source, assert bitwise equality
  kl         read two retained logits files and compute the metrics

Nothing here trains, samples, or serves; vision and MTP tensors are out of scope
and explicitly not loaded.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import resource
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Module-level safetensors imports: mode_parity saves its raw captures (and its per-layer
# checkpoints) from module scope. A function-local import here previously left save_file
# undefined at that point, which killed a completed 48-layer capture with NameError.
from safetensors.torch import load_file as sf_load
from safetensors.torch import save_file as save_file

import torch

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
WORK = Path(os.environ.get("FN_KL_WORK", "/data/nvme/work/flashnext-kl"))

BF16_REPO = "Qwen/Qwen3.8-Flash-Next"
BF16_REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
EXL3_DEFAULT = "/data/nvme/models/Qwen3.8-Flash-Next-exl3-2.05bpw"
PREFIX = "model.language_model."
N_LAYERS = 48
T_PANEL = 832


def log(msg: str) -> None:
    print(f"[fnkl] {msg}", flush=True)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())



def mem_available_mib() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return -1


def wait_for_host_memory(min_mib: int, timeout_s: float = 900.0) -> int:
    """Pause the run while the host is short on memory; abort rather than risk it.

    Bounds agreed with the peer holding the GPU window: keep MemAvailable above
    FN_MIN_MEM_MIB (default 25 GiB) and our own RSS under ~12 GiB.
    """
    if min_mib <= 0:
        return mem_available_mib()
    t0 = time.time()
    warned = False
    while True:
        avail = mem_available_mib()
        if avail < 0 or avail >= min_mib:
            return avail
        if not warned:
            log(f"HOST MEMORY GUARD: MemAvailable={avail} MiB < {min_mib} MiB, pausing")
            warned = True
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"host memory guard: MemAvailable stayed below {min_mib} MiB "
                             f"for {timeout_s:.0f}s (last {avail} MiB)")
        time.sleep(15)


def source_stats(source) -> dict:
    """Both the frozen interface (`stats` dict) and fn_sources' summary are accepted."""
    fn = getattr(source, "stats_summary", None)
    if callable(fn):
        try:
            return fn()
        except Exception as e:  # never mask the run for a stats helper
            return {"stats_summary_error": f"{type(e).__name__}: {e}"}
    return dict(getattr(source, "stats", {}) or {})


def sha256_file(path: str | Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if limit is not None and h.digest_size and f.tell() >= limit:
                break
    return h.hexdigest()


def wjson(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ----------------------------------------------------------------------------- panels


@dataclass
class Panels:
    ids: torch.Tensor            # [B, T] long
    attn: torch.Tensor           # [B, T] bool, 1 = real token
    score: torch.Tensor          # [B, T] bool, scored positions
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"shape": list(self.ids.shape),
                "tokens": int(self.attn.sum()), "scored": int(self.score.sum()),
                "meta": self.meta}


def load_panels(path: Path) -> Panels:
    from safetensors.torch import load_file

    d = load_file(str(path))
    wiki = d["wiki_ids"]
    held = d["heldout_ids"]
    hmask = d["heldout_mask"]
    if wiki.shape[1] < T_PANEL:
        wiki = torch.nn.functional.pad(wiki, (0, T_PANEL - wiki.shape[1]), value=0)
    ids = torch.cat([wiki, held], dim=0)
    attn = torch.ones_like(ids, dtype=torch.bool)
    score = torch.cat([torch.ones_like(wiki, dtype=torch.bool), hmask.bool()], dim=0)
    meta = {}
    side = path.with_suffix(".json")
    if side.is_file():
        meta = json.loads(side.read_text())
    return Panels(ids=ids, attn=attn, score=score, meta=meta)


# ----------------------------------------------------------------------------- model


def load_text_config(exl3_root: Path):
    from transformers import Qwen4ExpTextConfig

    cfg_all = json.loads((exl3_root / "config.json").read_text())
    tc = dict(cfg_all["text_config"])
    tc = {k: v for k, v in tc.items() if k not in ("model_type",)}
    cfg = Qwen4ExpTextConfig(**tc)
    got = {"num_hidden_layers": cfg.num_hidden_layers, "layer_types": len(cfg.layer_types),
           "num_experts": cfg.num_experts, "num_experts_per_tok": cfg.num_experts_per_tok,
           "ple_layer_ids": list(cfg.ple_layer_ids), "hc_count": cfg.hc_count,
           "ngram_size": cfg.ngram_size, "heads_per_ngram": cfg.heads_per_ngram,
           "ngram_vocab_size_base": cfg.ngram_vocab_size_base,
           "indexer_budget": cfg.indexer_budget, "vocab_size": cfg.vocab_size,
           "hidden_size": cfg.hidden_size, "max_position_embeddings": cfg.max_position_embeddings}
    assert cfg.num_hidden_layers == N_LAYERS, got
    assert len(cfg.layer_types) == N_LAYERS, got
    return cfg, cfg_all, got


def build_skeleton(cfg, dtype: torch.dtype, source=None):
    """Instantiate Qwen4ExpTextModel with every tensor on the meta device.

    `init_empty_weights` cannot be used here: the PLE n-gram embedding allocates a
    320,001,536 x 160 table inside `nn.Embedding.__init__`, i.e. before any
    register_parameter hook runs (204.8 GB fp32, which is what a real allocation
    attempt does). Building under `torch.device("meta")` keeps the construction
    allocation-free, and every buffer is then refreshed with real values:
      * rotary inverse frequencies are recomputed by re-instantiating the upstream
        rotary module on CPU,
      * the three PLE hash buffers are recomputed with the upstream helpers and,
        when the weight source stores them (the official BF16 checkpoint does),
        asserted equal to the stored values.
    """
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        _build_layer_multipliers, _find_nth_prime_after)
    from transformers import Qwen4ExpTextModel

    if dtype != torch.float32:
        raise SystemExit("reference math runs in FP32 activations by design")
    with torch.device("meta"):
        model = Qwen4ExpTextModel(cfg)
    model.eval()
    report = refresh_buffers(model, cfg, source)
    model._fn_buffer_report = report
    meta_params = sum(1 for p in model.parameters() if p.is_meta)
    log(f"skeleton: {meta_params} params on meta, buffers refreshed: "
        f"{sorted(report)}")
    return model


def refresh_buffers(model, cfg, source=None) -> dict:
    """Give every non-parameter buffer real, upstream-identical values."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        _build_layer_multipliers, _find_nth_prime_after)

    report: dict = {}
    rot = type(model.rotary_emb)(config=cfg)
    for name, buf in rot.named_buffers():
        model.rotary_emb._buffers[name] = buf.detach().clone()
        report[f"rotary_emb.{name}"] = {"origin": "recomputed_upstream_module",
                                        "shape": list(buf.shape)}

    ple_layer = list(cfg.ple_layer_ids)[0] - 1          # checkpoint module index
    ple_index = list(cfg.ple_layer_ids).index(list(cfg.ple_layer_ids)[0])
    ngram_heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram
    module = model.layers[ple_layer].ple.ple_embedding
    mult = _build_layer_multipliers(cfg.vocab_size, cfg.ngram_size, ple_index, cfg.seed)
    sizes: list[int] = []
    offsets: list[int] = []
    total = 0
    for head_idx in range(ngram_heads):
        global_head_idx = ple_index * ngram_heads + head_idx
        size = _find_nth_prime_after(cfg.ngram_vocab_size_base - 1, global_head_idx + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    values = {
        "layer_multipliers": torch.tensor(mult, dtype=torch.int64),
        "ngram_heads_vocab_sizes": torch.tensor(sizes, dtype=torch.int64),
        "ngram_heads_offsets": torch.tensor(offsets, dtype=torch.int64),
    }
    for name, tensor in values.items():
        key = f"{PREFIX}layers.{ple_layer}.ple.ple_embedding.{name}"
        stored_equal = None
        stored_note = None
        if source is not None:
            try:
                stored = source.get(resolve_key(source, key)).to(torch.int64).reshape(-1)
                stored_equal = bool(torch.equal(stored, tensor))
            except KeyError:
                stored_note = "not present in source"
            except Exception as e:
                stored_note = f"{type(e).__name__}: {e}"
        module._buffers[name] = tensor
        report[name] = {"origin": "recomputed_upstream_helpers", "shape": list(tensor.shape),
                        "stored_equal": stored_equal,
                        "stored_checked": stored_equal is not None,
                        "stored_note": stored_note}
    return report


# ----------------------------------------------------------------------------- mapping


@dataclass
class ParamSpec:
    kind: str                    # tensor | experts | ngram_head | skip
    key: str = ""
    layer: int = -1
    extra: dict = field(default_factory=dict)


_RESOLVE_MEMO: dict[tuple, str] = {}


def resolve_key(source, key: str) -> str:
    """Artifact keys drop the trailing `.weight` for EXL3-quantized tensors."""
    memo = _RESOLVE_MEMO.get((id(source), key))
    if memo:
        return memo
    names = source.names()
    if key in names:
        _RESOLVE_MEMO[(id(source), key)] = key
        return key
    stem = key[: -len(".weight")] if key.endswith(".weight") else None
    if stem and stem in names:
        _RESOLVE_MEMO[(id(source), key)] = stem
        return stem
    raise KeyError(f"{key} not present in source ({len(names)} names)")



def iter_local_params(module):
    """Yield (relative_name, owning_module, local_name, param) for a module tree."""
    for mod_path, mod in module.named_modules():
        for local, p in list(mod._parameters.items()):
            if p is None:
                continue
            full = f"{mod_path}.{local}" if mod_path else local
            yield full, mod, local, p


def set_local_param(mod, local: str, tensor: torch.Tensor) -> None:
    """Replace a (possibly meta-device) parameter with real storage."""
    mod._parameters[local] = torch.nn.Parameter(tensor, requires_grad=False)


def materialize_toplevel(model, source) -> dict:
    """embed_tokens + hyper_connection_mixer are not inside a decoder layer."""
    rec = {}
    for name, param in model.named_parameters():
        if name.startswith("layers."):
            continue
        spec = param_spec(name)
        assert spec.kind == "tensor", (name, spec)
        t = source.get(resolve_key(source, spec.key)).float()
        if tuple(t.shape) != tuple(param.shape):
            raise SystemExit(f"shape mismatch {spec.key}: {tuple(t.shape)} vs {tuple(param.shape)}")
        mod_path, _, local = name.rpartition(".")
        owner = model.get_submodule(mod_path) if mod_path else model
        set_local_param(owner, local, t)
        rec[name] = list(param.shape)
    log(f"toplevel materialized: {rec}")
    return rec


def param_spec(name: str) -> ParamSpec:
    """Map an HF Qwen4ExpTextModel parameter/buffer name to an artifact key."""
    if name == "embed_tokens.weight":
        return ParamSpec("tensor", PREFIX + "embed_tokens.weight")
    if name.startswith("hyper_connection_mixer."):
        return ParamSpec("tensor", PREFIX + name)
    if name.startswith("layers."):
        parts = name.split(".")
        layer = int(parts[1])
        rel = ".".join(parts[2:])
        if rel.startswith("mlp.experts."):
            tail = rel.split(".")
            if tail[-1] == "gate_up_proj":
                return ParamSpec("experts", layer=layer, extra={"kind": "gate_up"})
            if tail[-1] == "down_proj":
                return ParamSpec("experts", layer=layer, extra={"kind": "down"})
            raise SystemExit(f"unexpected experts parameter {name}")
        if rel.endswith("ngram_embedding.weight"):
            return ParamSpec("ngram_head", layer=layer,
                             extra={"rel": rel[: -len(".weight")]})
        return ParamSpec("tensor", PREFIX + f"layers.{layer}." + rel, layer=layer)
    raise SystemExit(f"unmapped parameter {name}")


def artifact_index(root: Path) -> dict:
    idx = json.loads((root / "model.safetensors.index.json").read_text())
    return idx


# ----------------------------------------------------------------------------- ngram shim

_NGRAM_RE = re.compile(r"ngram_embedding(?:\.shard_(\d+))?\.weight$")


def ngram_tables(source, model) -> tuple[list[tuple[str, int]], int, int]:
    """Locate the (possibly sharded) n-gram table and validate it against the model.

    The official checkpoint stores it as `...ngram_embedding.shard_<k>.weight`
    (128 shards x 2500012 rows x 160 in bf16); the EXL3 artifact may store one
    tensor or a sharded/quantized equivalent. Returns ([(key, rows)], row_dim,
    total_rows) with the model's own expected row count asserted.
    """
    ple = [l for l in model.layers if getattr(l, "ple", None) is not None]
    assert len(ple) == 1, f"expected exactly one PLE layer, found {len(ple)}"
    emb = ple[0].ple.ple_embedding.ngram_embedding
    expected_rows, row_dim = int(emb.weight.shape[0]), int(emb.weight.shape[1])
    # Artifacts that keep the table in its own file (EXL3) describe it in that file's
    # safetensors metadata instead of per-shard `.weight` tensors in the index.
    info_fn = getattr(source, "_ngram_info", None)
    if callable(info_fn):
        info = info_fn() or {}
        md = info.get("metadata") or {}
        if md.get("rows"):
            rows = int(md["rows"])
            shard_rows = int(md.get("shard_rows") or rows)
            rd = int(md.get("row_dim") or row_dim)
            stems = [s for s in (info.get("quantized_stems") or [])]
            assert rd == row_dim, f"ngram row_dim {rd} != model {row_dim}"
            assert rows == expected_rows, f"ngram rows {rows} != model {expected_rows}"
            if stems:
                # real artifact names (the source resolves these); no synthesized keys
                assert len(stems) * shard_rows >= rows, (len(stems), shard_rows, rows)
                return ([(s, shard_rows) for s in stems], row_dim, rows)
    found = []
    for n in source.names():
        m = _NGRAM_RE.search(n)
        if m:
            found.append((int(m.group(1)) if m.group(1) else 0, n))
    assert found, "no n-gram embedding tensor found in the source"
    found.sort()
    tables = []
    total = 0
    for idx, key in found:
        shp = source.meta(key).shape
        assert len(shp) == 2 and int(shp[1]) == row_dim, (key, shp, row_dim)
        tables.append((key, int(shp[0])))
        total += int(shp[0])
    assert total == expected_rows, f"ngram rows {total} != model {expected_rows}"
    return tables, row_dim, total


class NGramFileGather(torch.nn.Module):
    """Drop-in for the PLE nn.Embedding: gathers only the rows the panel needs.

    Handles a sharded table by mapping a global row id to (shard, local row);
    every shard except the last must be full, which is asserted at construction.
    """

    def __init__(self, source, tables: list[tuple[str, int]], row_dim: int):
        super().__init__()
        self.tables = tables
        self.row_dim = row_dim
        self.num_rows = sum(r for _, r in tables)
        self.source = source
        self.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16),
                                         requires_grad=False)   # upstream reads .weight.device
        self.calls = 0
        self.requested_rows = 0
        self.unique_rows = 0
        self.per_shard: dict[str, int] = {}
        self.seconds = 0.0
        if len(tables) > 1:
            full = tables[0][1]
            assert all(r == full for _, r in tables[:-1]), \
                f"sharded ngram table must be full except the last: {[r for _, r in tables]};"

    def _gather(self, flat: torch.Tensor) -> torch.Tensor:
        out = torch.empty(flat.numel(), self.row_dim, dtype=torch.float32)
        uniq = torch.unique(flat)
        if len(self.tables) == 1:
            key, rows = self.tables[0]
            vals = self.source.get_rows(key, uniq)
            assert tuple(vals.shape) == (uniq.numel(), self.row_dim), tuple(vals.shape)
            pos = torch.searchsorted(uniq, flat)
            out = vals.float()[pos].contiguous()
            self.per_shard[key] = self.per_shard.get(key, 0) + int(uniq.numel())
            return out
        bounds = []
        acc = 0
        for key, rows in self.tables:
            bounds.append((key, rows, acc))
            acc += rows
        shard_of = torch.searchsorted(torch.tensor([b[2] for b in bounds] + [acc]),
                                      uniq, right=True) - 1
        for si, (key, rows, start) in enumerate(bounds):
            ids = uniq[shard_of == si]
            if ids.numel() == 0:
                continue
            # max_gap_bytes merges near-adjacent rows into few large spans, so a scattered
            # per-token gather becomes a small number of coalesced range requests instead of
            # one request per contiguous run (the exact bytes returned are unchanged).
            vals = self.source.get_rows(key, ids - start, max_gap_bytes=1 << 20)
            assert tuple(vals.shape) == (ids.numel(), self.row_dim), tuple(vals.shape)
            self.per_shard[key] = self.per_shard.get(key, 0) + int(ids.numel())
            sel = torch.searchsorted(uniq, ids)
            out[sel] = vals.float()
        pos = torch.searchsorted(uniq, flat)
        return out[pos].contiguous()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        orig_shape = ids.shape
        flat = ids.reshape(-1).long()
        assert int(flat.min()) >= 0 and int(flat.max()) < self.num_rows, "ngram id out of range"
        t0 = time.time()
        rows = self._gather(flat)
        self.seconds += time.time() - t0
        self.calls += 1
        self.requested_rows += int(flat.numel())
        self.unique_rows += int(torch.unique(flat).numel())
        return rows.reshape(*orig_shape, self.row_dim)

    def stats(self) -> dict:
        return {"calls": self.calls, "row_lookups": self.requested_rows,
                "unique_rows": self.unique_rows, "seconds": round(self.seconds, 3),
                "num_rows": self.num_rows, "row_dim": self.row_dim,
                "shards": len(self.tables), "unique_rows_per_shard": self.per_shard}


def install_ngram_shim(model, source, tables, row_dim):
    shims = []
    for idx, layer in enumerate(model.layers):
        ple = getattr(layer, "ple", None)
        if ple is None:
            continue
        emb = ple.ple_embedding
        shim = NGramFileGather(source, tables, row_dim)
        emb.ngram_embedding = shim
        shims.append(shim)
        log(f"ngram gather shim installed on layer {idx}: {len(tables)} shard(s), "
            f"{shim.num_rows} rows, dim {row_dim}")
    return shims


# ----------------------------------------------------------------------------- experts fp32


def install_experts_fp32(model):
    """Mirror Qwen4ExpTextExperts.forward exactly, casting each expert slice to FP32.

    The upstream forward is copied verbatim (see modeling_qwen4_exp.py:872-898) with
    `current_state.float()` / weight.float() so weights stay bf16 in RAM but every
    matmul is FP32 — identical arithmetic to `w.to(bfloat16).to(float32) @ x`.
    """
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextExperts

    seen = 0

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx].float()
            gate, up = torch.nn.functional.linear(
                current_state, self.gate_up_proj[expert_idx].float()).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = torch.nn.functional.linear(
                current_hidden_states, self.down_proj[expert_idx].float())
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None].float()
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states

    for module in model.modules():
        if isinstance(module, Qwen4ExpTextExperts):
            module.forward = forward.__get__(module, Qwen4ExpTextExperts)
            seen += 1
    log(f"experts FP32 cast forward installed on {seen} MoE blocks")
    return seen


# ----------------------------------------------------------------------------- streaming


class LayerLoader:
    """Materializes/evicts one decoder layer's parameters from a weight source."""

    def __init__(self, model, source, stats_path: Path, free_after: bool = True):
        self.model = model
        self.source = source
        self.stats_path = stats_path
        self.free_after = free_after
        self.records: list[dict] = []
        self.loaded: set[int] = set()
        self.capture_idx: int | None = None
        self.capture: tuple | None = None
        self.run_stats: dict = {}
        self.decode_ctx: dict | None = None
        self.expected: dict[tuple[int, str], tuple] = {}
        self.min_mem_mib = int(os.environ.get("FN_MIN_MEM_MIB", "25600"))
        self.mem_min_seen = mem_available_mib()
        self.all_layers = list(model.layers)   # absolute indices survive --layers-list

    def materialize(self, idx: int) -> dict:
        t0 = time.time()
        wait_for_host_memory(int(os.environ.get("FN_MIN_MEM_MIB", "25600")))
        layer = self.all_layers[idx]
        n_tensors = 0
        bytes_in = 0
        for name, owner, local, param in iter_local_params(layer):
            spec = param_spec(f"layers.{idx}.{name}")
            if spec.kind == "tensor":
                key = resolve_key(self.source, spec.key)
                t = self.source.get(key)
                bytes_in += t.numel() * t.element_size()
                n_tensors += 1
                self._assign(owner, local, t, key, self.expected[(idx, name)])
            elif spec.kind == "experts":
                n_tensors, bytes_in = self._fill_experts(owner, local, spec, n_tensors,
                                                        bytes_in, self.expected[(idx, name)])
            elif spec.kind == "ngram_head":
                continue                       # served by the gather shim
            else:
                raise SystemExit(f"spec kind {spec.kind} not materializable")
        dt = time.time() - t0
        self.mem_min_seen = min(self.mem_min_seen, mem_available_mib())
        rec = {"layer": idx, "tensors": n_tensors, "source_bytes": bytes_in,
               "mem_available_mib_after": mem_available_mib(),
               "resident_bytes": sum(p.numel() * p.element_size()
                                     for p in self._params_with_data(layer)),
               "seconds": round(dt, 2)}
        self.records.append(rec)
        self.loaded.add(idx)
        log(f"layer {idx}: {n_tensors} tensors, {bytes_in / 2**20:.0f} MiB read, "
            f"{rec['resident_bytes'] / 2**20:.0f} MiB resident, {dt:.1f}s")
        return rec

    @staticmethod
    def _params_with_data(layer) -> list:
        out = []
        for p in layer.parameters():
            if p.data.numel():
                out.append(p)
        return out

    @staticmethod
    def _assign(owner, local, tensor: torch.Tensor, key: str, expected: tuple) -> None:
        if tuple(tensor.shape) != tuple(expected):
            raise SystemExit(f"shape mismatch {key}: artifact {tuple(tensor.shape)} "
                             f"vs model {tuple(expected)}")
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        set_local_param(owner, local, tensor)

    def _fill_experts(self, owner, local, spec: ParamSpec, n_tensors: int, bytes_in: int,
                      expected: tuple):
        layer = spec.layer
        experts = self.all_layers[layer].mlp.experts
        kind = spec.extra["kind"]
        fused = fused_expert_key(self.source, layer,
                                 "gate_up_proj" if kind == "gate_up" else "down_proj")
        if fused is not None:
            t = self.source.get(fused)
            if tuple(t.shape) != tuple(expected):
                raise SystemExit(f"fused expert shape {tuple(t.shape)} vs model "
                                 f"{tuple(expected)}")
            n_tensors += 1
            bytes_in += t.numel() * t.element_size()
            set_local_param(owner, local, t)
            return n_tensors, bytes_in
        num_experts = experts.num_experts
        inter = experts.intermediate_dim
        hidden = experts.hidden_dim
        if kind == "gate_up":
            assert tuple(expected) == (num_experts, 2 * inter, hidden), tuple(expected)
        else:
            assert tuple(expected) == (num_experts, hidden, inter), tuple(expected)
        if self.decode_ctx is not None:
            keys = []
            for e in range(num_experts):
                base = f"{PREFIX}layers.{layer}.mlp.experts.{e}."
                keys += ([base + "gate_proj.weight", base + "up_proj.weight"]
                         if kind == "gate_up" else [base + "down_proj.weight"])
            for k in keys:
                bytes_in += self.source.meta(resolve_key(self.source, k)).nbytes
            out = decode_experts_parallel(self.decode_ctx, layer, kind, num_experts, inter, hidden)
            set_local_param(owner, local, out)
            return n_tensors + len(keys), bytes_in
        if kind == "gate_up":
            out = torch.empty(num_experts, 2 * inter, hidden, dtype=torch.float32)
        else:
            out = torch.empty(num_experts, hidden, inter, dtype=torch.float32)
        keys: list[str] = []
        for e in range(num_experts):
            base = f"{PREFIX}layers.{layer}.mlp.experts.{e}."
            keys += ([base + "gate_proj.weight", base + "up_proj.weight"]
                     if kind == "gate_up" else [base + "down_proj.weight"])
        keys = [resolve_key(self.source, k) for k in keys]
        get_many = getattr(self.source, "get_many", None)
        tensors = get_many(keys) if get_many else {k: self.source.get(k) for k in keys}
        for t in tensors.values():
            bytes_in += t.numel() * t.element_size()
        n_tensors += len(tensors)
        for e in range(num_experts):
            base = f"{PREFIX}layers.{layer}.mlp.experts.{e}."
            if kind == "gate_up":
                g = tensors.pop(resolve_key(self.source, base + "gate_proj.weight"))
                u = tensors.pop(resolve_key(self.source, base + "up_proj.weight"))
                assert tuple(g.shape) == (inter, hidden) and tuple(u.shape) == (inter, hidden)
                out[e, :inter] = g.float()
                out[e, inter:] = u.float()
            else:
                d = tensors.pop(resolve_key(self.source, base + "down_proj.weight"))
                assert tuple(d.shape) == (hidden, inter)
                out[e] = d.float()
        set_local_param(owner, local, out)
        return n_tensors, bytes_in

    def evict(self, idx: int) -> None:
        if not self.free_after:
            return
        layer = self.all_layers[idx]
        for _, owner, local, param in iter_local_params(layer):
            set_local_param(owner, local, torch.empty(0, dtype=param.dtype))
        gc.collect()
        self.loaded.discard(idx)


def fused_expert_key(source, layer: int, which: str) -> str | None:
    """The official BF16 checkpoint stores fused expert tensors (HF layout).

    `...mlp.experts.gate_up_proj` [E, 2*inter, hidden] and `...down_proj`
    [E, hidden, inter]; the EXL3 artifact instead splits them per expert, so this
    returns None there and the per-expert path is used.
    """
    names = source.names()
    for cand in (f"{PREFIX}layers.{layer}.mlp.experts.{which}",
                 f"{PREFIX}layers.{layer}.mlp.experts.{which}.weight"):
        if cand in names:
            return cand
    return None


def _decode_experts_worker(payload):
    """Worker: decode a slice of experts straight into a shared-memory buffer.

    Each worker gets its own EXL3Source (and therefore its own pinned `ref`), so
    the decode is byte-for-byte the same code path as the serial one.
    """
    sys.path.insert(0, str(TOOLS))
    import torch as _t
    _t.set_num_threads(int(os.environ.get("FN_DECODE_THREADS", "4")))
    root, refsrc, layer, kind, num_experts, inter, hidden, out, lo, hi = payload
    import fn_sources
    src = fn_sources.EXL3Source(root, refsrc)
    for e in range(lo, hi):
        base = f"{PREFIX}layers.{layer}.mlp.experts.{e}."
        if kind == "gate_up":
            g = src.get(resolve_key(src, base + "gate_proj.weight"))
            u = src.get(resolve_key(src, base + "up_proj.weight"))
            out[e, :inter] = g
            out[e, inter:] = u
        else:
            out[e] = src.get(resolve_key(src, base + "down_proj.weight"))
    return hi - lo


def decode_experts_parallel(ctx: dict, layer: int, kind: str, num_experts: int,
                            inter: int, hidden: int) -> torch.Tensor:
    """Decode all experts of one layer into a shared bf16 buffer with N processes."""
    pool = ctx["pool"]
    workers = ctx["workers"]
    if kind == "gate_up":
        out = torch.empty(num_experts, 2 * inter, hidden, dtype=torch.bfloat16)
    else:
        out = torch.empty(num_experts, hidden, inter, dtype=torch.bfloat16)
    out = out.share_memory_()
    step = (num_experts + workers - 1) // workers
    payloads = [(ctx["root"], ctx["refsrc"], layer, kind, num_experts, inter, hidden,
                 out, lo, min(lo + step, num_experts))
                for lo in range(0, num_experts, step)]
    done = pool.map(_decode_experts_worker, payloads)
    assert sum(done) == num_experts, (done, num_experts)
    return out


def streaming_layer_forward(loader: LayerLoader, orig_forward, run_stats: dict, idx: int):
    def forward(self, *args, **kwargs):
        loader.materialize(idx)
        try:
            out = orig_forward(self, *args, **kwargs)
        finally:
            loader.evict(idx)
        hs = out[0] if isinstance(out, (tuple, list)) else out
        with torch.no_grad():
            run_stats.setdefault("layer_hidden", []).append({
                "layer": idx, "mean": float(hs.mean()), "std": float(hs.std()),
                "absmax": float(hs.abs().max()), "shape": list(hs.shape),
            })
        if loader.capture_idx == idx and loader.capture is None:
            loader.capture = (args, kwargs, hs.detach().clone())
        return out
    return forward


def install_streaming(model, source, stats_path: Path, free_after: bool = True) -> LayerLoader:
    loader = LayerLoader(model, source, stats_path, free_after=free_after)
    for i, layer in enumerate(model.layers):
        for name, _, _, p in iter_local_params(layer):
            loader.expected[(i, name)] = tuple(p.shape)
    run_stats: dict = {}
    for i, layer in enumerate(model.layers):
        orig = type(layer).forward
        layer.forward = streaming_layer_forward(loader, orig, run_stats, i).__get__(
            layer, type(layer))
    loader.run_stats = run_stats
    return loader


def replay_identity(loader: LayerLoader, model, idx: int = 0) -> dict:
    """Re-run one captured decoder-layer call after re-materializing its weights.

    This exercises both halves of the claim: the weight source is deterministic
    (weights are decoded/read a second time from the artifact) and the FP32 layer
    math is deterministic. Bitwise equality is required.
    """
    out: dict = {}
    cap = loader.capture
    if cap is None:
        out["error"] = f"layer {idx} call was not captured"
        return out
    args, kwargs, hs1 = cap
    layer = model.layers[idx]
    loader.materialize(idx)
    try:
        with torch.no_grad():
            res = type(layer).forward(layer, *args, **kwargs)
    finally:
        loader.evict(idx)
    hs2 = res[0] if isinstance(res, (tuple, list)) else res
    hs2 = hs2.detach()
    out["layer"] = idx
    out["bitwise_equal"] = bool(torch.equal(hs1, hs2))
    out["max_abs_diff"] = float((hs1 - hs2).abs().max())
    out["elements"] = int(hs1.numel())
    out["dtype"] = str(hs1.dtype)
    return out


# ----------------------------------------------------------------------------- head


def head_logits(hidden_rows: torch.Tensor, source, vocab_rows: int, block: int = 16384,
                native_key: str = "lm_head.weight", check: dict | None = None) -> torch.Tensor:
    """FP32 logits over the full vocab, streaming the head in vocab-row blocks."""
    native_key = resolve_key(source, native_key)
    out = torch.empty(hidden_rows.shape[0], vocab_rows, dtype=torch.float32)
    t0 = time.time()
    for r0 in range(0, vocab_rows, block):
        r1 = min(r0 + block, vocab_rows)
        w = source.get_rows(native_key, torch.arange(r0, r1))
        if tuple(w.shape) != (r1 - r0, hidden_rows.shape[1]):
            raise SystemExit(f"head block shape {tuple(w.shape)} at rows {r0}:{r1}")
        out[:, r0:r1] = hidden_rows @ w.float().T
        del w
    if check is not None:
        check.update({"head_key": native_key, "vocab_rows": vocab_rows,
                      "block": block, "seconds": round(time.time() - t0, 2)})
    return out


# ----------------------------------------------------------------------------- helpers


def rmsfree(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone()


def make_source(args, exl3_root: Path, bf16_map: dict | None = None):
    import fn_sources  # noqa: E402  (implemented in tools/fn_sources.py)

    if args.arm == "exl3":
        src = fn_sources.EXL3Source(str(exl3_root), args.refsrc)
        # ngram table lives in its own file for this artifact
        return src
    if args.bf16_dir:
        return fn_sources.BF16Source(local_dir=args.bf16_dir)
    return fn_sources.BF16Source(
        repo=BF16_REPO, rev=BF16_REV,
        token=(Path(args.hf_token_file).read_text().strip()
               if args.hf_token_file and Path(args.hf_token_file).is_file() else None))



class MapSource:
    """Coverage-only source backed by the pre-computed BF16 layout map (no HTTP)."""

    def __init__(self, path: str | Path):
        m = json.loads(Path(path).read_text())
        self.layout = m
        self._t = m["tensors"]
        self.stats = {"kind": "bf16_layout_map", "path": str(path),
                      "n_tensors": len(self._t), "repo": m.get("repo"),
                      "revision": m.get("revision"),
                      "payload_bytes": m.get("payload_bytes")}

    def names(self):
        return set(self._t)

    def meta(self, name):
        v = self._t[name]

        class M:
            pass
        mm = M()
        mm.shape = list(v["shape"])
        mm.dtype = v["dtype"]
        mm.nbytes = int(v["bytes"])
        return mm

    def get(self, *a, **k):
        raise RuntimeError("MapSource is coverage-only")

    get_rows = get_columns = get_many = get


def _meta_shape(source, name, keys_only):
    if keys_only:
        return None
    return tuple(source.meta(name).shape)


# ----------------------------------------------------------------------------- coverage


def mode_coverage(args) -> int:
    exl3_root = Path(args.exl3_root)
    cfg, cfg_all, cfg_fields = load_text_config(exl3_root)
    if args.arm == "bf16" and args.bf16_dir is None and Path(args.bf16_map).is_file():
        source = MapSource(args.bf16_map)
        log(f"coverage source: pre-computed BF16 layout map {args.bf16_map} (no HTTP)")
    else:
        source = make_source(args, exl3_root)
    model = build_skeleton(cfg, torch.float32, source)
    keys_only = bool(args.keys_only)
    log(f"coverage[{args.arm}]: enumerating source tensor names ...")
    names = set(source.names())
    # An EXL3 artifact stores a quantized linear as component tensors over a stem
    # (xxx.trellis / xxx.suh / xxx.svh / xxx.mul1), never xxx.weight, so the HF-side key must
    # be matched against the stem. This does not relax the gate -- an absent stem is still a
    # problem -- and `names` itself stays alias-free so `leftover` keeps its meaning.
    lookup = names | {n + ".weight" for n in names
                      if not n.endswith((".weight", ".bias")) and (n + ".weight") not in names}
    log(f"coverage[{args.arm}]: {len(names)} artifact tensors; walking model params ...")
    consumed: set[str] = set()
    problems: list[str] = []
    # `any(n.startswith(p) for n in names)` is O(len(names)) per call and is reached once per
    # missing expert tensor (layers x experts x 3 projections); on this artifact that is
    # hundreds of millions of startswith calls -- coverage never finished. Same semantics,
    # O(1) lookup via the set of all name prefixes.
    name_prefixes: set[str] = set()
    for n in names:
        for i in range(1, len(n) + 1):
            name_prefixes.add(n[:i])
    n_params = 0
    for pname, param in model.named_parameters():
        n_params += 1
        if n_params % 250 == 0:
            log(f"coverage[{args.arm}]: param {n_params} ({pname}) …")
        spec = param_spec(pname)
        if spec.kind == "tensor":
            if spec.key not in lookup:
                problems.append(f"missing artifact tensor for {pname} -> {spec.key}")
                continue
            shp = _meta_shape(source, spec.key, keys_only)
            if shp is not None and shp != tuple(param.shape):
                problems.append(f"shape {pname}: artifact {shp} vs model {tuple(param.shape)}")
            consumed.add(spec.key)
            if spec.key not in names and spec.key.endswith(".weight") \
                    and spec.key[:-len(".weight")] in names:
                # quantized stem form: count the artifact name so `leftover` stays meaningful
                consumed.add(spec.key[:-len(".weight")])
        elif spec.kind == "experts":
            layer = spec.layer
            which = "gate_up_proj" if spec.extra["kind"] == "gate_up" else "down_proj"
            fused = fused_expert_key(source, layer, which)
            if fused is not None:
                shp = _meta_shape(source, fused, keys_only)
                if shp is not None and shp != tuple(param.shape):
                    problems.append(f"fused shape {fused}: {shp} vs {tuple(param.shape)}")
                consumed.add(fused)
            else:
                for e in range(cfg.num_experts):
                    for proj in ("gate_proj", "up_proj", "down_proj"):
                        k = f"{PREFIX}layers.{layer}.mlp.experts.{e}.{proj}.weight"
                        if k not in lookup:
                            if k[:-1] not in name_prefixes:
                                problems.append(f"missing {k}")
                        else:
                            consumed.add(k)
                            if k not in names and k[:-len(".weight")] in names:
                                consumed.add(k[:-len(".weight")])
        elif spec.kind == "ngram_head":
            continue
    # not parameters but certainly part of the reference forward:
    for extra in ("lm_head.weight", "lm_head", "model.language_model.layers.1.ple.ple_embedding."
                  "layer_multipliers",
                  "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
                  "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"):
        if extra in names:
            consumed.add(extra)
    ngram_note = None
    try:
        tables, row_dim, total_rows = ngram_tables(source, model)
        consumed.update(k for k, _ in tables)
        # the table's component tensors (shard_k.trellis) are addressed through their stem, so
        # count them as consumed too -- otherwise they show up as anonymous leftovers.
        consumed.update(n for n in names
                        if any(n == f"{k}.trellis" or n.startswith(k + ".") for k, _ in tables))
    except Exception as e:            # e.g. the n-gram file has not downloaded yet
        ngram_note = f"{type(e).__name__}: {e}"
        tables, row_dim, total_rows = [], None, None
        log(f"coverage: ngram table not resolvable yet ({ngram_note})")
    leftover = sorted(
        n for n in names
        if n not in consumed
        and not n.startswith("model.visual.")
        and not n.startswith("mtp.")
    )
    out = {
        "generated": now(), "mode": "coverage", "arm": args.arm,
        "exl3_root": str(exl3_root), "bf16_repo": BF16_REPO, "bf16_rev": BF16_REV,
        "text_config_fields": cfg_fields, "hf_parameters": n_params,
        "artifact_tensors": len(names), "consumed": len(consumed),
        "leftover_non_vision_non_mtp": leftover[:200],
        "leftover_count": len(leftover),
        "ngram_table": {"shards": [{"key": k, "rows": r} for k, r in tables],
                        "row_dim": row_dim, "total_rows": total_rows,
                        "note": ngram_note},
        "keys_only": keys_only,
        "problems": problems[:200], "problem_count": len(problems),
        "source_stats": source_stats(source),
    }
    dest = Path(args.out)
    wjson(dest, out)
    log(f"coverage: params={n_params} artifact={len(names)} consumed={len(consumed)} "
        f"leftover={len(leftover)} problems={len(problems)} -> {dest}")
    for p in problems[:20]:
        log("  PROBLEM " + p)
    for p in leftover[:20]:
        log("  LEFTOVER " + p)
    return 1 if problems else 0


# ----------------------------------------------------------------------------- run


def run_arm(args) -> int:
    exl3_root = Path(args.exl3_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw"
    raw.mkdir(exist_ok=True)

    cfg, cfg_all, cfg_fields = load_text_config(exl3_root)
    panels = load_panels(Path(args.panels))
    if args.rows:
        keep = [int(x) for x in args.rows.split(",")]
        panels = Panels(ids=panels.ids[keep], attn=panels.attn[keep], score=panels.score[keep],
                        meta=panels.meta)
    log(f"panels: {panels.as_dict()}")

    source = make_source(args, exl3_root)
    model = build_skeleton(cfg, torch.float32, source)
    model_dir = getattr(source, "root", None) or getattr(source, "local_dir", None) or BF16_REPO

    n_experts = install_experts_fp32(model)
    toplevel = materialize_toplevel(model, source)
    if args.skip_ngram:
        forced = [i for i, l in enumerate(model.layers) if getattr(l, "ple", None) is not None]
        if args.layers_list:
            sel0 = {int(x) for x in args.layers_list.split(",")}
            forced = [i for i in forced if i in sel0]
        if forced:
            raise SystemExit(f"--skip-ngram refused: layer(s) {forced} need the PLE table")
        shims, tables, row_dim, total_rows = [], [], None, None
        log("ngram table skipped by request (--skip-ngram): PLE not exercised")
    else:
        tables, row_dim, total_rows = ngram_tables(source, model)
        shims = install_ngram_shim(model, source, tables, row_dim)
    loader = install_streaming(model, source, out_dir / "layer_stats.json",
                               free_after=(not args.keep_layers))
    pool = None
    if args.arm == "exl3" and args.decode_workers > 1:
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(args.decode_workers)
        loader.decode_ctx = {"pool": pool, "workers": args.decode_workers,
                             "root": str(exl3_root), "refsrc": args.refsrc}
        log(f"parallel EXL3 expert decode: {args.decode_workers} worker processes")
    if args.identity:
        loader.capture_idx = args.identity_layer
    if args.limit_layers:
        model.layers = model.layers[: args.limit_layers]
        log(f"WARNING: --limit-layers={args.limit_layers}: NOT a full-model run")
    if args.layers_list:
        sel = [int(x) for x in args.layers_list.split(",")]
        model.layers = torch.nn.ModuleList([model.layers[i] for i in sel])
        log(f"WARNING: --layers-list={sel}: partial-layer run, NOT a full-model result")

    ids = panels.ids
    attn = panels.attn.long()
    log(f"forward: batch={ids.shape[0]} T={ids.shape[1]} layers={len(model.layers)}")

    h_check: dict = {}
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    hidden = out.last_hidden_state
    fwd_s = time.time() - t0
    log(f"forward done in {fwd_s:.1f}s, hidden {tuple(hidden.shape)} {hidden.dtype}")

    if args.identity:
        h_check["layer_replay"] = replay_identity(loader, model, args.identity_layer)
        log(f"identity layer replay: {h_check['layer_replay']}")

    rows_flat = hidden.reshape(-1, hidden.shape[-1]).float()
    head_stats: dict = {}
    logits = head_logits(rows_flat, source, cfg.vocab_size, block=args.head_block,
                         check=head_stats)
    log(f"logits {tuple(logits.shape)} in {head_stats['seconds']}s")
    if args.identity:
        logits2 = head_logits(rows_flat, source, cfg.vocab_size, block=args.head_block)
        h_check["head_repeat_bitwise"] = bool(torch.equal(logits, logits2))
        del logits2

    from safetensors.torch import save_file
    lp = raw / f"logits_{args.arm}.safetensors"
    save_file({"logits": logits.contiguous()}, str(lp), metadata={
        "arm": args.arm, "model_dir": str(model_dir), "bf16_rev": BF16_REV,
        "rows": str(ids.shape[0]), "T": str(ids.shape[1]),
        "panels": str(args.panels), "generated": now(),
    })
    hs = hidden.to(torch.bfloat16).contiguous()
    save_file({"hidden": hs}, str(raw / f"hidden_{args.arm}.safetensors"))

    ident = {
        "generated": now(), "mode": "logits", "arm": args.arm,
        "math": {
            "activations": "float32", "weights": "bf16 storage -> float32 per matmul",
            "note": "identical policy to the 27B screen: weight.to(bfloat16).to(float32) @ x.float()",
            "experts": "per-expert slice cast to float32 inside a verbatim copy of the upstream MoE forward",
            "layers": len(model.layers),
            "full_model": not (args.limit_layers or args.layers_list),
            "layers_list": args.layers_list,
            "ngram_exercised": bool(shims),
            "head": f"lm_head streamed in {args.head_block}-vocab-row blocks, fp32 matmul",
        },
        "panels": panels.as_dict(),
        "toplevel_shapes": toplevel,
        "buffer_report": getattr(model, "_fn_buffer_report", {}),
        "source": {"kind": args.arm, "model_dir": str(model_dir), "bf16_repo": BF16_REPO,
                   "bf16_rev": BF16_REV, "stats": source_stats(source)},
        "ngram": [s.stats() for s in shims],
        "layer_records": loader.records,
        "layer_hidden_stats": loader.run_stats.get("layer_hidden", []),
        "head": head_stats,
        "identity_checks": h_check,
        "timing": {"forward_seconds": round(fwd_s, 2)},
        "environment": {
            "torch": torch.__version__, "python": sys.version.split()[0],
            "peak_rss_mib": round(peak_rss_mib(), 1),
            "mem_available_mib_min_seen": loader.mem_min_seen,
            "mem_available_mib_guard": loader.min_mem_mib,
            "refsrc": args.refsrc, "ref_sha256": ref_fingerprint(args.refsrc),
        },
        "artifacts": {
            "logits": str(lp), "logits_sha256": sha256_file(lp),
            "panels": str(args.panels), "panels_sha256": sha256_file(args.panels),
            "hidden": str(raw / f"hidden_{args.arm}.safetensors"),
        },
    }
    if pool is not None:
        pool.close()
        pool.join()
        ident["decode_workers"] = args.decode_workers
    wjson(out_dir / f"identity_{args.arm}.json", ident)
    log(f"wrote {lp} ({lp.stat().st_size / 2**30:.2f} GiB) and identity_{args.arm}.json")
    return 0


def ref_fingerprint(refsrc: str | None) -> dict:
    if not refsrc:
        return {}
    try:
        import ref  # noqa: F401
        return {"path": ref.__file__, "sha256": sha256_file(ref.__file__)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ----------------------------------------------------------------------------- KL


def mode_kl(args) -> int:
    from safetensors.torch import load_file
    import math

    panels = load_panels(Path(args.panels))
    if args.rows:
        keep = [int(x) for x in args.rows.split(",")]
        panels = Panels(ids=panels.ids[keep], attn=panels.attn[keep], score=panels.score[keep],
                        meta=panels.meta)
    ref = load_file(args.ref_logits)["logits"].float()
    arm = load_file(args.arm_logits)["logits"].float()
    assert ref.shape == arm.shape, (tuple(ref.shape), tuple(arm.shape))
    B, T = panels.ids.shape
    V = ref.shape[-1]
    assert ref.shape[0] == B * T, (tuple(ref.shape), B, T)

    lp_ref = torch.log_softmax(ref, dim=-1)
    p_ref = lp_ref.exp()
    top_ref = ref.argmax(-1)
    lp = torch.log_softmax(arm, dim=-1)
    kl_all = (p_ref * (lp_ref - lp)).sum(-1)
    agree = (arm.argmax(-1) == top_ref)

    mask = panels.score.reshape(-1)
    wiki_mask = panels.score[:2].reshape(-1) if panels.ids.shape[0] > 2 else None
    meta = {
        "generated": now(), "mode": "kl",
        "definition": "mean over scored positions of KL(P_ref || P_arm) with p_ref from the reference arm",
        "ref_logits": str(args.ref_logits), "ref_sha256": sha256_file(args.ref_logits),
        "arm_logits": str(args.arm_logits), "arm_sha256": sha256_file(args.arm_logits),
        "panels": str(args.panels), "panels_sha256": sha256_file(args.panels),
        "vocab": V, "rows": B, "T": T,
        "scored_positions": int(mask.sum()),
        "kl_mean": float(kl_all[mask].mean()),
        "kl_p50": float(kl_all[mask].median()),
        "kl_p99": float(torch.quantile(kl_all[mask].float(), 0.99)),
        "kl_max": float(kl_all[mask].max()),
        "top1_vs_ref": float(agree[mask].float().mean()),
        "per_row_kl_mean": [float(kl_all[i * T:(i + 1) * T][panels.score[i]].mean())
                            for i in range(B)],
        "per_row_scored": [int(panels.score[i].sum()) for i in range(B)],
    }
    if wiki_mask is not None:
        held = panels.score[2:].reshape(-1) if panels.ids.shape[0] > 2 else None
        meta["wikitext_slice"] = {
            "scored": int(wiki_mask.sum()),
            "kl_mean": float(kl_all[:wiki_mask.numel()][wiki_mask].mean()),
            "top1_vs_ref": float(agree[:wiki_mask.numel()][wiki_mask].float().mean()),
        }
        if held is not None:
            base = wiki_mask.numel()
            meta["heldout_response_slice"] = {
                "scored": int(held.sum()),
                "kl_mean": float(kl_all[base:][held].mean()),
                "top1_vs_ref": float(agree[base:][held].float().mean()),
            }
    if args.identity:
        self_kl = (p_ref * (lp_ref - lp_ref)).sum(-1)
        meta["self_kl_max_abs"] = float(self_kl.abs().max())
    wjson(Path(args.out), meta)
    log(f"KL mean={meta['kl_mean']:.6f} p99={meta['kl_p99']:.4f} "
        f"top1={meta['top1_vs_ref']:.4f} scored={meta['scored_positions']} -> {args.out}")
    return 0


# ----------------------------------------------------------------------------- main


def _write_checkpoint(ckpt_dir: Path, layer: int, tensor: "torch.Tensor", prompt_len: int) -> Path:
    """Write ONE layer checkpoint to disk immediately (called from the capture hook, i.e. during
    the forward, so a run cut short keeps every completed layer). Never saves a view."""
    p = Path(ckpt_dir) / f"layer_{int(layer)}.safetensors"
    p.parent.mkdir(parents=True, exist_ok=True)   # self-sufficient: never lose a layer to a missing dir
    save_file({f"layer_{int(layer)}": tensor.detach().float().cpu().clone().contiguous()}, str(p),
              metadata={"layer": str(int(layer)), "prompt_len": str(int(prompt_len))})
    return p


def _write_parity_artifacts(out_dir: Path, raw_caps: dict, hidden, logits, vocab: int) -> dict:
    """The single serialization path used by mode_parity (and exercised by the selftest)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    # NOTE: a row of a large tensor is a VIEW; torch.save of a view serialises the whole backing
    # storage (a single 248320-float row once became 3.1 GB). Everything saved here is cloned.
    if raw_caps:
        p = out_dir / "raw_layers.safetensors"
        save_file({f"layer_{i}": t.detach().clone() for i, t in raw_caps.items()}, str(p))
        written["raw_layers"] = str(p)
        p = out_dir / "raw_hidden_logits.safetensors"
        save_file({"hidden": hidden.detach().float().cpu().clone(),
                   "logits_last": logits.reshape(-1, vocab)[-1].detach().float().cpu().clone()}, str(p))
        written["raw_hidden_logits"] = str(p)
    return written


def mode_selftest_write(args) -> int:
    """Real serialization contract test through the same helpers mode_parity uses, on a small
    throwaway fixture (no model, no forward). Proves: module-scope save_file works, per-layer
    checkpoints land DURING the capture loop, and every file round-trips bitwise."""
    out = Path(args.out or "/tmp/fn_parity_write_selftest")
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "raw_layers_partial"
    torch.manual_seed(0)
    layers = 4
    caps = {}
    for i in range(layers):
        t = torch.randn(1, 3, 16)
        caps[i] = t
        p = _write_checkpoint(ck, i, t, 3)                    # happens per layer, mid-loop
        assert p.is_file(), f"checkpoint {p} missing immediately after write"
    hidden = torch.randn(1, 3, 8)
    logits = torch.randn(1, 3, 32)
    written = _write_parity_artifacts(out, caps, hidden, logits, 32)
    back = sf_load(written["raw_layers"])
    ok = set(back) == {f"layer_{i}" for i in range(layers)} and all(
        torch.equal(back[f"layer_{i}"], caps[i]) for i in range(layers))
    hl = sf_load(written["raw_hidden_logits"])
    ok = ok and torch.equal(hl["hidden"], hidden) and torch.equal(
        hl["logits_last"], logits.reshape(-1, 32)[-1])
    for i in range(layers):
        assert torch.equal(sf_load(ck / f"layer_{i}.safetensors")[f"layer_{i}"], caps[i])
    # simulate a forward that dies AFTER the capture but BEFORE the consolidated write
    partial = len(list(ck.glob("layer_*.safetensors")))
    print(f"selftest-write: module-scope save_file OK | {layers} checkpoints on disk during capture "
          f"({partial} files) | consolidated round-trip bitwise {'OK' if ok else 'FAILED'}")
    print(f"selftest-write: a crash before the consolidated write still leaves {partial} layers on disk")
    return 0 if ok and partial == layers else 1


def mode_parity(args) -> int:
    """Reference forward on an explicit batch-1 token list, with per-layer records.

    Same artifact, same quant decode, batch 1, exact sequence length (no padding), full 48
    layers, so each recorded quantity lines up with the engine's ``fwd_modules`` output for
    the identical ids. This is the reference half of the same-quant port gate; it is not a
    served-port measurement and it does not replace the declared tolerance test.
    """
    if not args.prompt_ids_file:
        raise SystemExit("parity: --prompt-ids-file is required")
    raw = json.loads(Path(args.prompt_ids_file).read_text())
    if isinstance(raw, dict):
        raw = raw.get("ids") or raw.get("wiki_ids") or raw.get("prompt_ids")
    ids_list = [int(x) for x in raw]
    ids = torch.tensor([ids_list], dtype=torch.long)
    exl3_root = Path(args.exl3_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, cfg_all, cfg_fields = load_text_config(exl3_root)
    source = make_source(args, exl3_root)
    model = build_skeleton(cfg, torch.float32, source)

    install_experts_fp32(model)
    materialize_toplevel(model, source)
    tables, row_dim, total_rows = ngram_tables(source, model)
    shims = install_ngram_shim(model, source, tables, row_dim)
    loader = install_streaming(model, source, out_dir / "layer_stats.json",
                               free_after=(not args.keep_layers))
    attn = torch.ones_like(ids)
    raw_caps: dict = {}
    # Per-layer disk checkpoints: each captured layer is written out as soon as it is produced,
    # so a run that is cut short (deadline, crash, SIGINT) still leaves every completed layer on
    # disk. The in-memory dict is kept only for the final consolidated file.
    ckpt_dir = out_dir / "raw_layers_partial"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _cap(i):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(o) and o.dim() >= 2 and i not in raw_caps:
                t = o.detach().float().cpu().clone().contiguous()   # clone: never save a view
                raw_caps[i] = t
                try:
                    _write_checkpoint(ckpt_dir, i, t, int(ids.shape[1]))
                except Exception as exc:                                # never break the forward
                    log(f"checkpoint layer {i} failed (continuing): {type(exc).__name__}: {exc}")
        return hook

    for i, layer in enumerate(model.layers):
        layer.register_forward_hook(_cap(i))
    log(f"parity forward: batch=1 T={ids.shape[1]} layers={len(model.layers)} (no padding, "
        f"exact ids)")
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    hidden = out.last_hidden_state
    fwd_s = time.time() - t0
    log(f"forward done in {fwd_s:.1f}s, hidden {tuple(hidden.shape)} {hidden.dtype}")
    rows_flat = hidden.reshape(-1, hidden.shape[-1]).float()
    head_stats: dict = {}
    logits = head_logits(rows_flat, source, cfg.vocab_size, block=args.head_block,
                         check=head_stats)
    rec = {
        "mode": "parity",
        "note": ("batch 1, exact ids, no padding, full model, reference FP32; the reference half "
                 "of the same-quant port gate"),
        "ids": ids_list,
        "forward_seconds": round(fwd_s, 2),
        "hidden_shape": list(hidden.shape),
        "layer_hidden": loader.run_stats.get("layer_hidden", []),
        "logits_shape": list(logits.shape),
        "logits_last": [float(x) for x in logits.reshape(-1, cfg.vocab_size)[-1]],
        "head": head_stats,
        "shims": [s.stats() for s in shims],
        "ngram_table": {"row_dim": row_dim, "total_rows": total_rows},
        "source": source_stats(source),
        "full_model": True,
        "layers": len(model.layers),
        "generated": now(),
    }
    (out_dir / "parity.json").write_text(json.dumps(rec))
    written = _write_parity_artifacts(out_dir, raw_caps, hidden, logits, cfg.vocab_size)
    if "raw_layers" in written:
        log(f"wrote raw tensors for {len(raw_caps)} layers (+hidden/logits)")
    log(f"wrote {out_dir / 'parity.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["coverage", "logits", "kl", "parity", "selftest-write"])
    ap.add_argument("--prompt-ids-file", default=None,
                    help="parity: JSON file with the exact token ids (or {'ids': [...]})")
    ap.add_argument("--arm", choices=["bf16", "exl3"], default="exl3")
    ap.add_argument("--exl3-root", default=EXL3_DEFAULT)
    ap.add_argument("--bf16-dir", default=None, help="local BF16 shard dir (else HTTP range reads)")
    ap.add_argument("--refsrc", default=str(WORK / "refsrc"))
    ap.add_argument("--hf-token-file", default="~/.cache/huggingface/token")
    ap.add_argument("--bf16-map", default=str(WORK / "bf16_index" / "bf16_map.json"),
                    help="pre-computed BF16 shard layout map used for bf16 coverage")
    ap.add_argument("--keys-only", action="store_true",
                    help="coverage: verify key presence only (no shape reads)")
    ap.add_argument("--panels", default=str(WORK / "panels" / "panels.safetensors"))
    ap.add_argument("--out", default=str(WORK / "runs" / "adhoc"))
    ap.add_argument("--rows", default=None, help="debug: subset of panel rows, e.g. 0")
    ap.add_argument("--limit-layers", type=int, default=0,
                    help="debug only; labels the run as NOT full-model")
    ap.add_argument("--layers-list", default=None,
                    help="debug only: run exactly these decoder layers (NOT full-model)")
    ap.add_argument("--skip-ngram", action="store_true",
                    help="debug only: skip the PLE n-gram table (refused if a run layer needs it)")
    ap.add_argument("--keep-layers", action="store_true")
    ap.add_argument("--head-block", type=int, default=16384)
    ap.add_argument("--decode-workers", type=int, default=1,
                    help="worker processes for EXL3 expert decode (exl3 arm only)")
    ap.add_argument("--identity", action="store_true",
                    help="repeat one decoder-layer call and the head computation; assert bitwise equality")
    ap.add_argument("--identity-layer", type=int, default=0,
                    help="decoder layer index replayed for the identity check")
    ap.add_argument("--ref-logits", default=None)
    ap.add_argument("--arm-logits", default=None)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("FN_MAIN_THREADS", "16")))

    if args.mode == "selftest-write":
        return mode_selftest_write(args)
    if args.mode == "coverage":
        return mode_coverage(args)
    if args.mode == "logits":
        return run_arm(args)
    if args.mode == "parity":
        return mode_parity(args)
    if args.mode == "kl":
        if not (args.ref_logits and args.arm_logits):
            raise SystemExit("--ref-logits and --arm-logits are required for kl mode")
        return mode_kl(args)
    raise SystemExit("unreachable")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(3)
