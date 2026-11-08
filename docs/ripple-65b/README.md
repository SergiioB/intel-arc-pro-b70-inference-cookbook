# Ripple-65B Preview on two Arc Pro B70 cards (llama.cpp SYCL)

Ricursive's [Ripple-65B Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview)
(65B dense, Apache-2.0, released 2026-11-04) runs on two B70 cards with stock llama.cpp.
Use `-sm tensor`: it is 29 % faster to decode and 70 % faster to prefill than `-sm layer`.
Measured 2026-11-08, status **screen** (two repeats per cell). Evidence:
[`results/ripple-65b/dual-b70-20261108/`](../../results/ripple-65b/dual-b70-20261108/).

| Split | KV | Decode tok/s at 0 / 16K / 32K | Prefill tok/s (pp512) at 0 / 32K |
|---|---|---|---|
| **`-sm tensor`** | f16 | **13.3 / 12.3 / 11.7** | **189 / 176** |
| `-sm layer` | q8_0 | 10.3 / 9.8 / 9.6 | 107 / 106 |
| `-sm layer` | f16 | 10.4 / 9.3 / 8.6 | 112 / 107 |

Quality smoke on the `-sm tensor` server: exact-answer battery 9/9 (6 canaries + 3 prompts
at ~8K tokens), and a `merge_intervals` coding prompt answered with code whose asserts pass
when executed. That is a smoke test, not a quality ranking.

## Serve it

```bash
# llama.cpp b12167 or newer (Ripple support, PR #31692); we used b12195-3-gf477f795c, SYCL build
curl -L -o Ripple-65B-Preview-AD-IQ4_XS.gguf \
  https://huggingface.co/AtomicChat/Ripple-65B-Preview-GGUF/resolve/main/Ripple-65B-Preview-AD-IQ4_XS.gguf
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR='level_zero:0,1' ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
llama-server -m Ripple-65B-Preview-AD-IQ4_XS.gguf -ngl 99 -fa on -sm tensor \
  -ctk f16 -ctv f16 -c 32768 -t 8 --jinja --port 8050
```

- `AD-IQ4_XS` (38.7 GB, 4.78 bpw) is AtomicChat's imatrix file; their card reports 2.5×
  lower KL than the plain `Q4_K_M` at the same size. It needs both cards.
- Keep f16 KV with `-sm tensor`: llama.cpp issue
  [#26409](https://github.com/ggml-org/llama.cpp/issues/26409) reports hangs with quantized
  KV in tensor mode. We did not test quantized KV with `-sm tensor`.
- Four of every five layers use 1,024-token sliding-window attention, so decode loses only
  12 % from empty context to 32K.
- `--jinja` uses the chat template embedded in the GGUF (reasoning and tools).

## Claim boundary

- Two B70 cards on PCIe 4.0 x8/x8 (CPU lanes), caps 180 W and 230 W as found, single
  stream, `llama-bench -ngl 99 -fa 1 -t 8 -p 512 -n 128 -d 0,16384,32768 -r 2`.
  Contexts above 32K were not measured.
- One file (`AD-IQ4_XS`). Ricursive publishes no sampling defaults for the Preview; the smoke
  used temperature 0.
- Card draw during the `-sm tensor` sweep averaged ~367 W for both cards together.
