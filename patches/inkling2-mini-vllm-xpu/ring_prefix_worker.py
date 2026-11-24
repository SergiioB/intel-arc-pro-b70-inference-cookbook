# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker side of the opt-in single-slot exact-tail prefix cache.

Every piece of recurrent state this stack keeps outside vLLM's block-hash
reuse -- the sliding-window short-conv caches, the sliding-window local
K/V, and the DSpark draft caches -- is a manager-owned paged tensor bound to
an ``AttentionLayerBase`` in ``static_forward_context``. Hash-verified reuse
restores the deep prefix; this module owns the last-``slack``-token frontier:

* ``ring_high`` -- the live high-water mark: the end position of the latest
  forward in eager mode. Prefix hits admitted within ``[ring_high - slack,
  ring_high]`` read live retained pages.
* ``saved_frontier`` -- the end position of the last forward of at least
  ``min_snapshot_rows`` rows (a real prefill chunk; decode and DSpark
  verification forwards are smaller). Its tail is captured as a
  ``PrefillTail`` so a resend ending exactly at it replays the original
  final chunk byte-for-byte instead of recomputing at a different chunk
  size (chunk-size-dependent numerics).

The scheduler-side ``RingPrefix`` admits only hits these two markers can
cover; this module fails closed with ``RuntimeError`` if a forward ever
arrives at an uncovered position, so no state is silently reused.
"""

import os

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context

from .prefix_tail import PrefillTail
from .ring_prefix import RingPrefix

# `<|unused_200054|>` rows in input_ids mark positions where the runner merged
# vision embeddings in place of text embeddings (IMAGE_PAD_ID in
# vllm.models.inkling2.processing).
IMAGE_PAD_ID = 200054


class RingPrefixWorker:
    min_rows = RingPrefix.min_snapshot_rows

    def __init__(self, static_forward_context: dict, vllm_config) -> None:
        self.static_ctx = static_forward_context
        self._vllm_config = vllm_config
        # Must cover every hit the scheduler's RingPrefix can admit. Admitted
        # hits land on cache_hit_alignment_tokens (resolved after profiling,
        # before warmup); start at --block-size and upgrade once it exists.
        self.slack = RingPrefix.slack_for(vllm_config.cache_config.block_size)
        self.ring_high = 0
        self.saved_frontier = 0
        self.saved_tail: PrefillTail | None = None
        self._fwd: tuple[int, int, dict] | None = None
        self._debug = os.environ.get("INK2_RING_PREFIX_DEBUG", "0") == "1"
        self._seen: set[str] = set()
        # First physical page of the request's block table; forwards of one
        # request (including spec-decode re-prefills that restart inside the
        # computed range) share it, so it separates a fresh request arriving
        # on a prefix hit from an in-request re-prefill.
        self._req_fp: int | None = None
        # Per-cache logical block size (positions per block_table column),
        # learned by checking which candidate reproduces a captured
        # slot_mapping; 0 means unresolved and disables prefix-row restore.
        self._slot_L: dict[str, int] = {}
        # Extra captured rows beyond the replay tail so hits anywhere in the
        # slack ring still find their conv-tap rows inside the snapshot.
        w = 1
        for obj in static_forward_context.values():
            if type(obj).__name__ == "InklingConvState":
                w = max(w, getattr(obj, "sliding_window", 1))
        self._state_extra = w - 1

    def _sync_slack(self) -> None:
        hit = getattr(
            self._vllm_config.cache_config, "cache_hit_alignment_tokens", None
        )
        if hit:
            self.slack = max(self.slack, RingPrefix.slack_for(hit))

    def _fp(self, md_map: dict) -> int | None:
        for md in md_map.values():
            table = getattr(md, "block_table", None)
            if table is not None and table.ndim == 2 and table.shape[0] >= 1:
                return int(table[0, 0])
        return None

    def _log_once(self, msg: str) -> None:
        if self._debug and msg not in self._seen:
            self._seen.add(msg)
            self._log(msg)

    @staticmethod
    def _log(msg: str) -> None:
        if os.environ.get("INK2_RING_PREFIX_DEBUG", "0") == "1":
            import sys

            print(f"INK2 ring prefix: {msg}", file=sys.stderr, flush=True)

    def _metadata_map(self) -> dict | None:
        ctx = get_forward_context()
        # Bookkeeping only runs for eager forwards; graph capture/replay
        # neither reads nor mutates these markers.
        if ctx.cudagraph_runtime_mode is not CUDAGraphMode.NONE:
            self._log_once(f"cg_mode={ctx.cudagraph_runtime_mode}")
            return None
        md_map = ctx.attn_metadata
        if not isinstance(md_map, dict):
            self._log_once(f"md_type={type(md_map)}")
        return md_map if isinstance(md_map, dict) else None

    def _paged_items(self, md_map: dict, rows: slice) -> list[tuple] | None:
        """(name, cache, slots) for every bound 4-D paged cache."""
        items = []
        for name, obj in self.static_ctx.items():
            cache = getattr(obj, "kv_cache", None)
            if not (
                isinstance(cache, torch.Tensor)
                and cache.ndim == 4
                and cache.numel() > 0
            ):
                continue
            md = md_map.get(name)
            slots = getattr(md, "slot_mapping", None) if md is not None else None
            if slots is None:
                return None
            items.append((name, cache, slots[rows]))
        return items

    def _num_reqs(self, md_map: dict) -> int:
        for md in md_map.values():
            table = getattr(md, "block_table", None)
            if table is not None:
                return table.shape[0]
        return 0

    def _learn_slot_map(
        self, name: str, md, cache: torch.Tensor, start_pos: int, slots
    ) -> int:
        """Learn the positions-per-block_table-column for a paged cache by
        finding the candidate that reproduces its captured slot_mapping."""
        known = self._slot_L.get(name)
        if known is not None:
            return known
        table = getattr(md, "block_table", None)
        ps = cache.shape[2]
        n = slots.numel()
        L = 0
        if table is not None and table.ndim == 2 and table.shape[0] >= 1:
            pos = torch.arange(start_pos, start_pos + n, device=table.device)
            cmp_slots = slots.to(table.device)
            for cand in (ps, getattr(md, "block_size", 0) or 0, 4, 8, 16, 32):
                if not cand:
                    continue
                blk = torch.div(pos, cand, rounding_mode="floor")
                if int(blk.max()) >= table.shape[1]:
                    continue
                est = table[0, blk] * ps + pos % ps
                if torch.equal(est, cmp_slots):
                    L = cand
                    break
        self._slot_L[name] = L
        if not L:
            self._log(f"slot map unresolved for {name}; state restore off")
        return L

    def _prefix_items(
        self, md_map: dict, start: int, count: int
    ) -> list[tuple] | None:
        """(name, cache, slots) for absolute positions [start, start+count),
        computed from each group's block table (prefix region: outside this
        forward's own slot_mapping)."""
        items = []
        for name, obj in self.static_ctx.items():
            cache = getattr(obj, "kv_cache", None)
            if not (
                isinstance(cache, torch.Tensor)
                and cache.ndim == 4
                and cache.numel() > 0
            ):
                continue
            md = md_map.get(name)
            table = getattr(md, "block_table", None) if md is not None else None
            L = self._slot_L.get(name, 0)
            if table is None or not L:
                return None
            pos = torch.arange(start, start + count, device=table.device)
            blk = torch.div(pos, L, rounding_mode="floor")
            if int(blk.max()) >= table.shape[1]:
                return None
            slots = table[0, blk] * cache.shape[2] + pos % cache.shape[2]
            items.append((name, cache, slots))
        return items

    def _restore_state(self, md_map: dict, first: int) -> None:
        """Rewrite the saved rows for positions [frontier-n_state, first) into
        this request's slots so the admitted hit reads byte-exact state."""
        tail = self.saved_tail
        assert tail is not None
        n_state = tail.cache_rows[0].shape[0] if tail.cache_rows else 0
        start = self.saved_frontier - n_state
        count = first - start
        if count <= 0:
            return
        items = self._prefix_items(md_map, start, count)
        if items is None:
            self._log(f"state restore skipped at {first}: slots unresolved")
            return
        tail.restore_state([(cache, slots) for _, cache, slots in items], count)
        self._log(f"restored {count} state rows at {first}")

    def entry(self, input_ids, positions, inputs_embeds):
        """Return a replayed forward output for an exact-tail resend, else
        None after enforcing the retained-state coverage guard."""
        self._sync_slack()
        self._fwd = None
        self._log_once(
            f"entry-reached ctx={type(get_forward_context().attn_metadata)}"
        )
        md_map = self._metadata_map()
        if md_map is None:
            self._log_once("md_map_none")
            return None
        n_reqs = self._num_reqs(md_map)
        if n_reqs != 1:
            self._log_once(f"num_reqs={n_reqs}")
            return None
        size = positions.numel()
        first = int(positions[0])
        total = first + size
        new_req = self._fp(md_map) != self._req_fp
        if new_req:
            self._req_fp = self._fp(md_map)
        self._log(
            f"entry size={size} first={first} total={total} "
            f"high={self.ring_high} saved={self.saved_frontier} "
            f"tail={self.saved_tail is not None} new={new_req}"
        )
        if (
            self.saved_tail is not None
            and input_ids is not None
            and 0 < first < self.ring_high
            and total == self.saved_frontier
            and size >= self.min_rows
        ):
            items = self._paged_items(md_map, slice(0, size))
            if items is not None:
                # vLLM already matched the preceding block hashes (including
                # salt); the tail verify guards the unsuffixed remainder.
                res = self.saved_tail.restore(
                    input_ids[:size],
                    [(cache, slots) for _, cache, slots in items],
                )
                if res is not None:
                    self.ring_high = total
                    self._log(f"replay {size} tail tokens at {total}")
                    hidden, aux = res
                    return hidden if aux is None else (hidden, aux)
                self._log(f"replay skipped: tail mismatch at {first}..{total}")
            else:
                self._log("replay skipped: no paged metadata")
        # The coverage guard applies only to the first forward of a fresh
        # request arriving on a prefix hit. In-request re-prefills (spec
        # decode re-running an interior range) always read state they wrote
        # themselves, so they must not trip the fail-closed path.
        if new_req:
            # A divergent branch invalidates the association between the
            # saved outputs and the live request's prefix.
            if first < self.saved_frontier or first == 0:
                self.saved_tail = None
            if first == 0:
                self.ring_high = self.saved_frontier = 0
            elif first < self.ring_high - self.slack:
                if (
                    self.saved_tail is not None
                    and self.saved_frontier - self.slack
                    <= first
                    <= self.saved_frontier
                ):
                    self._restore_state(md_map, first)
                    self.ring_high = first
                else:
                    raise RuntimeError(
                        "INK2 ring prefix: admitted prefix hit exceeds both "
                        "retained ring histories"
                    )
        self.ring_high = max(self.ring_high, total)
        self._fwd = (size, total, md_map)
        return None

    def exit(self, input_ids, inputs_embeds, output) -> None:
        """After a real forward: checkpoint the frontier and its tail."""
        fwd, self._fwd = self._fwd, None
        if fwd is None:
            return
        size, total, md_map = fwd
        if size < self.min_rows:
            return
        self.saved_frontier = total
        self.saved_tail = None
        # Only forwards whose tail holds no vision-pad rows are captured:
        # those positions embed encoder features that token ids alone cannot
        # re-derive, so a captured tail could never verify on a resend.
        if input_ids is None or bool((input_ids == IMAGE_PAD_ID).any()):
            return
        if isinstance(output, tuple):
            hidden, aux_list = output
            aux = list(aux_list[:-1])
        else:
            hidden, aux = output, None
        n = min(self.slack, size)
        n_state = min(self.slack + self._state_extra, size)
        items = self._paged_items(md_map, slice(size - n_state, size))
        if items is not None:
            for name, cache, slots in items:
                md = md_map.get(name)
                self._learn_slot_map(
                    name, md, cache, total - n_state, slots
                )
            self.saved_tail = PrefillTail.capture(
                input_ids[-n:],
                hidden[-n:],
                None if aux is None else [a[-n:] for a in aux],
                [(cache, slots) for _, cache, slots in items],
            )
            self._log(
                f"captured tail n={n} state={n_state} at {total} "
                f"caches={len(items)}"
            )
        else:
            self._log(f"capture skipped at {total}: no paged metadata")
