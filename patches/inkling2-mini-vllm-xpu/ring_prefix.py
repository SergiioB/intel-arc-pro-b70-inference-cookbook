# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard vLLM prefix hits with the single slot's resident or saved ring history."""

import os


class RingPrefix:
    # Smaller forwards are also used by single-token decode and DSpark verification.
    # Leave a prefill tail that both the scheduler and worker will checkpoint.
    min_snapshot_rows = 9

    @classmethod
    def slack_for(cls, hit_tokens: int) -> int:
        """Extra positions each ring retains below its frontier.

        Hits admitted to a request land on ``hit_tokens`` boundaries
        (``cache_hit_alignment_tokens``: the scheduler block size = LCM of the
        KV group block sizes, or the finer hash unit when partial-hash hits
        are enabled). Both the convolution and local-attention rings keep
        ``slack`` extra positions beyond their receptive fields; one frontier
        represents live decode state, the other a snapshot from the last
        sizeable prompt chunk. ``limit()`` caps admitted hits at
        ``request_tokens - min_snapshot_rows`` so every hit leaves a
        replayable tail, so when a frontier ends at remainder ``r`` in
        ``[1, min_snapshot_rows)`` the nearest admissible boundary sits
        ``hit_tokens + r`` positions back. ``hit_tokens +
        min_snapshot_rows - 1`` therefore covers *every* remainder: for any
        ``end >= hit_tokens`` some admissible boundary lands inside the ring.
        """
        return max(32, hit_tokens + cls.min_snapshot_rows - 1)

    def __init__(self, hash_tokens, hit_tokens=None):
        # `hash_tokens` is the granularity of Request.block_hashes (the GCD
        # of prefix-cacheable group block sizes). `hit_tokens` is the
        # alignment the coordinator returns for num_computed_tokens; it is
        # what `covers()` must hold and what `limit()` may admit.
        self.hash_tokens = hash_tokens
        self.hit_tokens = hit_tokens or hash_tokens
        self.slack = self.slack_for(self.hit_tokens)
        self.owner = None
        self.high = 0
        self.snapshot = 0
        self.hashes = ()
        if os.environ.get("INK2_RING_PREFIX_DEBUG", "0") == "1":
            import sys

            print(
                f"INK2 ring prefix: hash_tokens={hash_tokens} "
                f"hit_tokens={self.hit_tokens} slack={self.slack}",
                file=sys.stderr,
                flush=True,
            )

    def scheduled(self, owner, end, rows):
        if self.owner != owner:
            self.high = 0
            # An exact checkpoint remains valid when a new request continues
            # directly from it, even if its final prefill chunk is tiny.
            if end - rows != self.snapshot:
                self.snapshot = 0
        self.owner = owner
        self.high = max(self.high, end)
        if rows >= self.min_snapshot_rows:
            self.snapshot = end
        self.hashes = ()

    def finish(self, owner, committed, hashes):
        if owner == self.owner:
            self.hashes = tuple(hashes[: min(committed, self.high) // self.hash_tokens])

    def covers(self, length):
        return length > 0 and any(
            end - self.slack <= length <= end for end in (self.high, self.snapshot) if end > 0
        )

    def limit(self, hashes, allowed, request_tokens):
        """Block hashes include the full prefix and the caller's cache salt."""
        # A short *original* final chunk can start exactly at a saved, aligned
        # checkpoint. Replay that chunk at its original size on every resend.
        saved = self.snapshot
        if (saved and saved <= allowed and saved % self.hit_tokens == 0
                and 0 < request_tokens - saved < self.min_snapshot_rows):
            blocks = saved // self.hash_tokens
            if blocks <= min(len(self.hashes), len(hashes)) and self.hashes[blocks - 1] == hashes[blocks - 1]:
                return saved
        allowed = min(allowed, request_tokens - self.min_snapshot_rows)
        available = min(len(self.hashes), len(hashes), allowed // self.hash_tokens)
        # The coordinator floors num_computed_tokens to `hit_tokens`; admit
        # only hit-aligned boundaries so `covers()` sees what it checks.
        stride = max(1, self.hit_tokens // self.hash_tokens)
        for end in (self.high, self.snapshot):
            latest = min(available, end // self.hash_tokens)
            latest -= latest % stride
            for index in range(latest, 0, -stride):
                if index * self.hash_tokens < end - self.slack:
                    break
                if self.hashes[index - 1] == hashes[index - 1]:
                    return index * self.hash_tokens
        return 0
