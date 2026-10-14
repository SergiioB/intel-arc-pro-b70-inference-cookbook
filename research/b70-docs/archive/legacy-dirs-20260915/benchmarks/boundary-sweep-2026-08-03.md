# Boundary Sweep — VRAM/Context Limits for Every Config

**Date:** 2026-08-03
**Build:** b10222 (`build-sycl-0801`)
**Method:** `llama-bench -p 512,4096 -n 128 -r 3 -fa 1 -ctk q8_0 -ctv q4_1 -b 8192 -ub 4096`
**Power:** 150W throughout | **Raw data:** `results/boundary-sweep-20260803-1744.txt`

## Results summary

### ThinkingCap 27B (dense, MTP) — context ceiling per quant

| Quant | Context | VRAM free | pp4096 | tg128 (base) | Fits? |
|-------|---------|-----------|--------|-------------|-------|
| Q4_K_M | **256K** | 930 MiB | 700 t/s | 18.4 t/s | ✅ TIGHT (930 MiB) |
| Q5_K_M | **128K** | 3250 MiB | 706 t/s | 16.2 t/s | ✅ COMFORTABLE |
| Q5_K_M | **200K** | 2677 MiB | 706 t/s | 16.3 t/s | ✅ WORKS (better than predicted!) |
| Q6_K | **128K** | 687 MiB | 715 t/s | 16.0 t/s | ✅ TIGHT (687 MiB — absolute max) |

**Note:** tg128 is llama-bench WITHOUT MTP (base decode). With MTP-4 enabled via
server, these become ~24-32 t/s (MTP gives +35-50% on dense). pp4096 is prefill.

### Qwen 35B MoE — context ceiling per quant

| Quant | Context | VRAM free | pp4096 | tg128 | Fits? |
|-------|---------|-----------|--------|-------|-------|
| Q4_K_XL | **512K** | 1593 MiB | **1659 t/s** | **70.2 t/s** | ✅ WORKS! |
| Q5_K_M | **512K** | — | — | — | ❌ OOM CRASH (server died) |

## Key findings

1. **Q5_K_M @ 200K WORKS** (2677 MiB free) — our linear model predicted only 364 MiB.
   The actual KV overhead is smaller than estimated. Q5 quality at 200K is viable.

2. **Q6_K @ 128K is the absolute quality ceiling** (687 MiB free). It fits but with
   almost zero headroom — any vision projector or extra context would OOM. Usable
   for pure text at exactly 128K, nothing more.

3. **MoE Q4_K_XL @ 512K is the context champion** — 1593 MiB free, 1659 t/s prefill,
   70.2 t/s decode. This is the highest context + highest throughput config available.

4. **MoE Q5_K_M @ 512K = OOM crash** as predicted. The 25 GB weights + 512K KV
   exceeds 32 GB. Q5 MoE maxes at 256K.

5. **Dense 27B decode is ~16-18 t/s base** across all quants (bandwidth-bound on
   weights, quant barely affects speed). The difference is QUALITY, not speed.
   With MTP-4: Q4→~29 t/s, Q5→~24 t/s, Q6→~24 t/s (estimated from +50% MTP gain).

6. **Prefill is flat across dense quants** (~700-715 t/s) — bandwidth-bound on weights.
   MoE prefill is 2.3× faster (1659 t/s) due to tiny active-param attention.

## Updated VRAM boundary map (measured)

```
                    @128K      @200K      @256K      @512K
27B Q4_K_M +MTP    8.1GB ✅   4.2GB ✅   0.9GB ⚠️   OVER
27B Q5_K_M +MTP    3.3GB ✅   2.7GB ✅   OVER       OVER
27B Q6_K +MTP      0.7GB ⚠️   OVER       OVER       OVER
35B MoE Q4_K_XL    8.0GB ✅   ~7GB ✅    6.1GB ✅   1.6GB ✅
35B MoE Q5_K_M     4.0GB ✅   ~3GB ✅    2.2GB ✅   OVER ❌
```
(Values = MiB free after load. ✅ >1GB, ⚠️ <1GB tight, OVER = crash)

## Command (reproducible)

```bash
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
# Example: Q5_K_M @ 200K
llama-bench -m /mnt/<models-disk>/ThinkingCap-Qwen3.6-27B-Q5_K_M-MTP.gguf \
  -ngl 99 -ncmoe 0 -fa 1 -ctk q8_0 -ctv q4_1 \
  -p 512,4096 -n 128 -r 3 -d 0 -b 8192 -ub 4096 -t 8 -dev SYCL0
```
