# Wave 9H test evidence: what the "700-test release chain" is made of

1. One full run: the Wave 9F suite on GPU0, 699 passed, 0 failed. Log: `exp17_moe_plugin_pytest_wave9f_gpu0.log`.
2. Wave 9H then added one admission regression test, `test_literal_block_sparse_moe_prefix_member_cannot_spoof_granite_bank`.
3. After that change the author reran the Wave 9F test file on GPU0 and reported 17 of 17 passed (16 tests from the 9F log plus the new one).
4. No log file of the 17/17 run exists in this study. The result is author-reported. Neither final reviewer reran it: see `opus55_wave9h_v020_final_signoff.md` and `astra_wave9h_v020_final_signoff.md`.
5. "700 tests" therefore means 699 in one logged run plus 1 added test with an author-reported pass. It is not one logged 700-test run.

This note was written on 2026-11-26 for the V11 bundle. It is a statement of provenance, not a test log.
