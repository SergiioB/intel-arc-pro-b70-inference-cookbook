# B70 Engine Comparison — vLLM XPU MTP vs llama.cpp SYCL (2026-08-06)

## Hardware & setup
- Intel Arc Pro B70 32GB, AMD Ryzen 7 5700X3D, Ubuntu 26.04
- **MoE 35B**: Qwen3.6-35B-A3B (3B active). vLLM = GPTQ-Int4 + native XpuFusedMoe + MTP spec (1 layer); llama.cpp = Q4_K_XL GGUF.
- **Dense 27B**: ThinkingCap-Qwen3.6-27B. llama.cpp = Q4_K_M GGUF (no usable vLLM path — FP8 kernel missing on XPU).
- Single-stream (max-num-seqs=1 / serial requests), temperature-controlled (cooldown ≤52°C between runs).
- Decode = best steady-state rep (drops JIT warmup). Prefill = avg of reps.

---

## 1. MoE Decode t/s (best steady-state) — vLLM MTP @150W vs llama.cpp @150W

Format: **vLLM** / llama.cpp (speedup)

| Prompt | g32 | g128 | g256 | g512 |
|--------|------|------|------|------|
| short (45 tok) | **127** / 74 (1.73×) | **118** / 72 (1.64×) | **113** / 67 (1.67×) | **110** / 72 (1.53×) |
| p512 (500 tok) | **121** / 73 (1.66×) | **116** / 72 (1.62×) | **115** / 72 (1.61×) | **113** / 72 (1.58×) |
| p1k (990 tok) | **113** / 73 (1.55×) | **114** / 64 (1.79×) | **114** / 70 (1.62×) | **105** / 70 (1.50×) |
| p2k (1935 tok) | **111** / 70 (1.58×) | **126** / 69 (1.82×) | **116** / 69 (1.67×) | **118** / 63 (1.87×) |
| p4k (3860 tok) | **130** / 66 (1.97×) | **114** / 65 (1.77×) | **116** / 65 (1.80×) | **116** / 65 (1.80×) |
| p8k (7535 tok) | **126** / 59 (2.14×) | **111** / 58 (1.92×) | **114** / 58 (1.97×) | **114** / 58 (1.96×) |

**MoE decode: vLLM MTP is 1.5–2.1× faster than llama.cpp** across every cell. MTP speculative decoding (1.69× over no-spec baseline) is the unlock. The advantage *grows with prompt length* (2.14× at p8k) because MTP amortizes the larger per-token bandwidth cost.

---

## 2. MoE Prefill t/s (avg) — vLLM MTP @150W vs llama.cpp @150W

| Prompt | tokens | vLLM prefill | llama.cpp prefill | vLLM win |
|--------|-------:|-------------:|------------------:|---------:|
| short | 55 | **563** | 104 | 5.4× |
| p512 | 510 | **3406** | 616 | 5.5× |
| p1k | 1000 | **5883** | 695 | 8.5× |
| p2k | 1945 | **6217** | 1498 | 4.2× |
| p4k | 3870 | **6626** | 1728 | 3.8× |
| p8k | 7545 | **7526** | 1662 | 4.5× |

**MoE prefill: vLLM is 3.8–8.5× faster.** This is vLLM's signature MoE win — the native int4 XpuFusedMoe kernel + continuous batching prefill path. llama.cpp's prefill is compute-bound and far behind at scale.

---

## 3. MoE Power sweep — 150W vs 230W (find the sweet spot)

Fixed cell: p2k prompt, g128 gen, single-stream.

| Wattage | vLLM MTP decode | vLLM prefill | llama.cpp decode (p2k/g128) | temp peak |
|---------|----------------:|-------------:|---------------------------:|----------:|
| **150W** | **125.7** | 7308 | 69 | ~58°C |
| 230W | 115.4 | 7345 | — | ~58°C |

**MoE sweet spot = 150W.** Δ 150→230W = -10.3 t/s (-8.2%). MoE self-limits power draw to ~140W regardless of cap ([power guide](../docs/POWER-SWEET-SPOTS.md)) — raising to 230W gives ~0 gain and wastes 80W of heat. **Run MoE at 150W.**

---

## 4. Dense 27B Decode t/s — llama.cpp power sweep (150W vs 230W)

Dense scales WITH power (unlike MoE). vLLM dense FP8 has **no XPU kernel** (`KeyError: PlatformEnum.XPU` in `choose_scaled_mm_linear_kernel`) — confirmed 2026-08-06, so llama.cpp is the only working dense path.

| Prompt | 150W g32 | 150W g128 | 230W g32 | 230W g128 | Δ (g128) |
|--------|---------:|---------:|---------:|----------:|---------:|
| short | 22 | 20 | 26 | 24 | +4.1 (+21%) |
| p512 | 19 | 19 | 25 | 24 | +4.9 (+26%) |
| p1k | 20 | 19 | 25 | 24 | +5.0 (+27%) |
| p2k | 19 | 18 | 24 | 23 | +4.8 (+26%) |

**Dense sweet spot = 180W (efficiency) or 230W (max speed).** Dense @230W gives +18-30% over 150W (21.5→25.5 t/s at short/g32). But dense runs HOT: sustained load hit **71°C @150W, 79°C @230W** (vs MoE's flat ~58°C). For sustained dense use, **180W is the thermal-safe sweet spot** ([power guide](../docs/POWER-SWEET-SPOTS.md): 0.155 t/s/W efficiency). 230W only for short bursts.

---

## 5. Cross-model summary (single-stream, sweet-spot wattage)

| Model | Engine | Config | Decode (p2k/g128) | Prefill (p2k) | Power | Temp |
|-------|--------|--------|------------------:|--------------:|------:|-----:|
| MoE 35B | **vLLM MTP** | GPTQ-Int4+MTP | **126 t/s** | **6217 t/s** | 150W | 58°C |
| MoE 35B | llama.cpp | Q4_K_XL | 69 t/s | 1498 t/s | 150W | 58°C |
| Dense 27B | llama.cpp | Q4_K_M | 23 t/s | 1007 t/s | 230W | 79°C |
| Dense 27B | llama.cpp | Q4_K_M | 18 t/s | 728 t/s | 150W | 71°C |
| Dense 27B | vLLM | FP8 | ❌ no XPU kernel | — | — | — |

**Key takeaways:**
1. **MoE is 5-6× faster decode than dense** on B70 (126 vs 23 t/s) — MoE reads ~3GB/token, dense reads ~19GB/token (bandwidth-bound both).
2. **vLLM MTP wins MoE** (1.8× decode, 4.2× prefill over llama.cpp) — but needs patched engine (GDN spec assert + BF16 draft).
3. **llama.cpp wins dense by default** — vLLM has no dense XPU kernel (FP8 block kernel missing). llama.cpp dense+MTP (~24-30 t/s; see the [dense gap analysis](../docs/qwen36-27/DENSE-FP8-GAP.md)) is the only path past the ~23 t/s Q4 baseline.
4. **Power: MoE=150W (self-limits), Dense=180W sweet / 230W burst** (scales +18-30%, but thermal cost).

