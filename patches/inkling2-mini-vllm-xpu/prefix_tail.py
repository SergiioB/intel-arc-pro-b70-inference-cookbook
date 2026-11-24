# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact outputs and cache rows for the final tokens of a saved text prefill.

Every state a later forward needs that vLLM's block-hash reuse does not
restore lives in a manager-owned paged cache on this stack: target K/V
(full and sliding-window), the per-layer/per-pass short-conv state caches,
and the DSpark draft K/V. Each cache group owns its own slot mapping, so the
tail is captured per (cache, slots) pair at that cache's page granularity.
"""

from dataclasses import dataclass

import torch


def _rows(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Byte copy of `slots` rows out of a `(pages, heads, page, width)` cache."""
    page_tokens = cache.shape[2]
    return cache.view(torch.uint8)[slots // page_tokens, :, slots % page_tokens]


@dataclass
class PrefillTail:
    token_ids: torch.Tensor
    hidden: torch.Tensor
    aux: list[torch.Tensor] | None
    cache_rows: list[torch.Tensor]

    @classmethod
    def capture(cls, ids, hidden, aux, items):
        """items: list of (cache, slots) for the captured positions.

        ``items`` may cover more positions than ``ids``: the cache rows span
        the state window needed by hits landing anywhere in the slack ring,
        while ``ids``/``hidden``/``aux`` cover only the replayable tail.
        """
        rows = [_rows(cache, slots).clone() for cache, slots in items]
        return cls(
            ids.clone(),
            hidden.clone(),
            None if aux is None else [a.clone() for a in aux],
            rows,
        )

    def restore_state(self, items, count: int) -> None:
        """Write the first ``count`` saved rows of each cache into ``items``.

        ``items`` holds (cache, slots) pairs whose slots address the same
        absolute positions as the saved rows' leading edge -- i.e. the prefix
        region a resend will read but never rewrite.
        """
        for (cache, slots), rows in zip(items, self.cache_rows, strict=True):
            if slots.numel() < count:
                return
            cache.view(torch.uint8)[
                slots[:count] // cache.shape[2], :, slots[:count] % cache.shape[2]
            ] = rows[:count]

    def restore(self, ids, items):
        """Write the saved rows into this request's own slots and return the
        saved `(hidden, aux)` suffix, or None when the suffix does not match.

        `ids` and every slots tensor cover this forward's tokens in order;
        `n = ids.numel()` is the replay length (the uncached tail).
        """
        n = ids.numel()
        if n > self.token_ids.numel() or not torch.equal(ids, self.token_ids[-n:]):
            return None
        if len(items) != len(self.cache_rows):
            return None
        for (cache, slots), rows in zip(items, self.cache_rows, strict=True):
            if slots.numel() != n:
                return None
            cache.view(torch.uint8)[
                slots // cache.shape[2], :, slots % cache.shape[2]
            ] = rows[-n:]
        hidden = self.hidden[-n:]
        return hidden, None if self.aux is None else [
            *[a[-n:] for a in self.aux], hidden
        ]
