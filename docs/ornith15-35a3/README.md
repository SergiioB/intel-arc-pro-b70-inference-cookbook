# Ornith-1.5-35B-A3B on the B70

Family index. Keep Qwen, Nemotron, and Muse numbers on their own pages.

| Path | What it is |
|---|---|
| [ORNITH-VLLM-XPU.md](ORNITH-VLLM-XPU.md) | Recipe: download, launch, measured results |
| [MIXEDCAL-V2.md](MIXEDCAL-V2.md) | Why this GPTQ exists: experts-only, mixed-domain cal, RTN 24.76% → 10.37% |
| [CLAIMS.md](CLAIMS.md) | Full measured tables and LocalMaxxing ids |
| `benchmarks/ornith15-35a3/launch-ornith-mtp1.sh` | Default serve: MTP1 + DraftINT4, cache off |
| [Dashboard SVG](../assets/b70-ornith15-dashboard.svg) | Decode, cold-input, capacity, and LocalMaxxing card |

## Artifact

Published under **`SergiioB`** (two i's):

- https://huggingface.co/SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2

Local experts-only GPTQ INT4 G128 with MTP left in BF16. Not an official Ornith GPTQ.

## Warnings

1. Same public image digest as Qwen3.8-27B (`f01e24f6`). Not the Qwen3.6 Pi `2c427ef` digest, and not the historical v0.21 native-int4moe image that produced Qwen3.6 MTP4 204.6.
2. Default serve is **MTP1 + DraftINT4**. MTP4 is slower than no-spec on this single-layer head. `DRAFT_INT4=0` is the BF16 draft.
3. Prefill ~9.7k is a **configured 230 W** no-spec cell. Combined MTP1+DraftINT4 230 W LocalMaxxing is `tokSOut` **108.4** / `tokSPrefill` **9073**. Those are different speculation / prompt / cap cells — do not mix them.
4. Do not apply GDN mixed-split v5 on C1. DFlash is not a path: there is no Ornith hidden-2048 draft.
5. LocalMaxxing `APPROVED` means accepted self-report, not independent reproduction.
