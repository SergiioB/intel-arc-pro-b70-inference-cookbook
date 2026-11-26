APPROVED FOR v0.2.0 RELEASE (WITH DOCUMENTED SCOPE)
Wave 9G blocker closed: Granite lookup requires `(True, member_name)`; a literal `block_sparse_moe:` prefix on an ordinary member cannot satisfy it.
The regression test accepts a genuine raw Granite bank and requires rejection of the prefixed lookalike with valid BF16 shapes.
The README distinguishes the live raw-text Gemma checkpoint from the chat-calibrated quality result and limits the Had64 claim to the tested model.
Diff-only review: no kernel changes in the patch. The 17/17 GPU0 result was reported, not rerun; no GPU or Docker commands were used.
