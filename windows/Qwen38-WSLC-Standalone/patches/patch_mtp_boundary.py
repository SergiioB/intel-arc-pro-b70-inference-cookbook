#!/usr/bin/env python3
"""Patch vLLM GDN metadata for a partial final speculative group.

At an exact max-model-length boundary, vLLM may schedule fewer than
num_speculative_tokens + 1 query tokens for the last speculative step. The XPU
GDN kernel requires complete speculative groups. Reclassify only that partial
pure-spec step as stateful non-spec prefill so the request can finish without
padding beyond its configured sequence length.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

MARKER = "B70_MTP_PARTIAL_FINAL_GROUP"

OLD = '''            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
'''

NEW = '''            if num_prefills == 0 and num_decodes == 0:
                expected_spec_token_size = num_spec_decodes * (self.num_spec + 1)
                actual_spec_token_size = query_start_loc_cpu[-1].item()
                if actual_spec_token_size < expected_spec_token_size:
                    # B70_MTP_PARTIAL_FINAL_GROUP: The max-sequence boundary can
                    # truncate the final speculative group. The XPU GDN kernel
                    # requires complete groups, so process this final partial
                    # group through the existing stateful non-spec prefill path.
                    spec_sequence_masks = None
                    spec_sequence_masks_cpu = None
                    num_prefills = num_spec_decodes
                    num_prefill_tokens = actual_spec_token_size
                    num_spec_decodes = 0
                    num_spec_decode_tokens = 0
                    spec_token_indx = None
                    non_spec_token_indx = None
                    spec_state_indices_tensor = None
                    non_spec_state_indices_tensor = block_table_tensor[:, 0]
                    spec_query_start_loc = None
                    non_spec_query_start_loc = query_start_loc
                    non_spec_query_start_loc_cpu = query_start_loc_cpu
                    num_accepted_tokens = None
                else:
                    spec_token_indx = torch.arange(
                        expected_spec_token_size,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                    non_spec_token_indx = torch.empty(
                        0, dtype=torch.int32, device=query_start_loc.device
                    )
                    # Filter by spec_sequence_masks to exclude padded sequences
                    spec_state_indices_tensor = block_table_tensor[
                        spec_sequence_masks_cpu, : self.num_spec + 1
                    ]
                    non_spec_state_indices_tensor = None
                    # Padded sequences are always at the back, so the first
                    # num_spec_decodes + 1 entries of query_start_loc already
                    # contain the correct cumulative token counts.
                    spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                    non_spec_query_start_loc = None
                    non_spec_query_start_loc_cpu = None
            else:
'''

OLD_FINALIZE = '''            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
'''

NEW_FINALIZE = '''            if spec_sequence_masks_cpu is not None:
                assert num_accepted_tokens is not None
                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
'''


def patch_text(text: str) -> str:
    if MARKER in text:
        return text
    if text.count(OLD) != 1:
        raise RuntimeError("GDN pure-spec anchor changed; refusing to patch")
    if text.count(OLD_FINALIZE) != 1:
        raise RuntimeError("GDN finalize anchor changed; refusing to patch")
    text = text.replace(OLD, NEW, 1)
    return text.replace(OLD_FINALIZE, NEW_FINALIZE, 1)


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm package not found")
    path = Path(next(iter(spec.submodule_search_locations))) / "v1/attention/backends/gdn_attn.py"
    original = path.read_text()
    patched = patch_text(original)
    if patched == original:
        print(f"already patched {path}")
        return
    compile(patched, str(path), "exec")
    path.write_text(patched)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
