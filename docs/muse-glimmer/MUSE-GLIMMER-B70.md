# Muse-Glimmer-30B on the B70 (llama.cpp SYCL)

> **Status 2026-08-10:** RUNNING on the Intel Arc Pro B70. Text (reasoning
> channel), vision (mmproj), and DFlash block-diffusion speculative decoding all
> verified.

## Stack that made this work

| Item | Value |
|---|---|
| Engine | llama.cpp master `d2f83055d` (284), SYCL IntelLLVM 2026.0, build `build-sycl-muse-0810` |
| ⚠️ Build flag | **`-DGGML_SYCL_F16=ON` is MANDATORY** — without it prefill loses 3.4× (pp4096 1301 → 293 t/s). Verify `GGML_SYCL_F16:BOOL=ON` in the CMakeCache. |
| Artifacts | `unsloth/Muse-Glimmer-30B-GGUF`: `UD-Q4_K_XL` (15.88 GB) + `mmproj-kquant` (1.40 GB) + `dflash-kquant` (1.63 GB) → `/mnt/models2/muse-glimmer/` |
| Context | 128K fits with `--parallel 1` (~2.3-4 GiB free); 64K with 4 slots |

## Serve command

```bash
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_CACHE_PERSISTENT=0
export SYCL_DEVICE_FILTER=level_zero ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ONEAPI_DEVICE_SELECTOR=level_zero:0 ZE_AFFINITY_MASK=0

llama-server -m /mnt/models2/muse-glimmer/Muse-Glimmer-30B-UD-Q4_K_XL.gguf \
  --mmproj /mnt/models2/muse-glimmer/mmproj-kquant.gguf \
  -c 131072 --parallel 1 -ngl 99 -ncmoe 0 -fa on -ctk q8_0 -ctv q4_1 -t 8 \
  --no-mmap -b 8192 -ub 8192 --host 0.0.0.0 --port 8765 \
  --spec-type draft-dflash --spec-draft-model /mnt/models2/muse-glimmer/dflash-kquant.gguf \
  --spec-draft-n-max 2
```

## Measured numbers (128K ctx, 230 W cap, `-ub 8192`, C1 cold, engine t/s)

**Decode — winner DFlash n_max=2 (n=5):**

| prompt | t/s median | max | TTFT | acceptance |
|---|---:|---:|---:|---:|
| p512/g128 | 26.8 | 28.9 | 1.1 s | 0.85 |
| p8192/g128 | 22.9 | 23.1 | 12.3 s | 0.75 |
| p32768/g128 | 21.1 | 21.1 | 48.1 s | 0.75 (n=3) |

**DFlash gain vs no-spec (no-spec: 22.5 / 17.2 / 13.9 t/s):** +19% p512, +33%
p8192, **+52% p32768** — the DFlash gain grows with context.

**n_max screen (p512/p8192, n=3):** n1 24.8/20.0 · **n2 27.6/21.4** · n3 26.3/21.3
· n4 27.5/18.2 · n5-7 collapse (acceptance 0.30-0.37) · **n8 aborts the server**.
Small draft windows win (same lesson as vLLM spec-N, §15.8).

**Prefill (llama-bench pp):** pp512 858 · pp2048 1278 · pp4096 1301 · pp8192 1172
(auto-ctx, Run 34) · **pp32768 865** (128K ctx, r=3). Long-prompt prefill
degrades as memory access grows.

**Provenance:** engine `timings.predicted_per_second` (llama-server HTTP);
prefill = llama-bench pp; measured draw via `energy1_input` (winner ~83-138 W,
peak 73 °C at 230 W cap); hwmon4 temp; exact prompts (512/8192/32768 + ~50
chat-template tokens verified); 1 warmup + n=3 (n=5 for top two depths).

## Quality caveat

Muse-Glimmer is a reasoning model — generations include a `reasoning_content`
channel; reported decode includes all generated tokens. **Speed does not
establish output/task-quality parity** — a quality check against the BF16
reference (e.g. perplexity or task benchmarks) is a separate, not-yet-done step.

## Failures / operations notes

- **vLLM profile manager steals :8765** — the Pi-bridge automation restarts the
  vLLM container periodically (even with the unit renamed; in-memory systemd
  state is stale after a failed daemon-reload). Kill `b70vllm` before each
  benchmark phase or use a different port.
- **llama-bench hangs after printing** (known §9.5) — kill after the table
  appears. Long llama-bench runs buffer all output until the end; give generous
  timeouts or the file comes out empty.
- `n8` DFlash aborts at launch at 128K — unsupported.
- A VRAM-overflow crash earlier (f16-KV probe on a dirty GPU) hard-rebooted the
  box and wedged the xe driver on kernel 7.0.0-29 (corrupt 157 MB debug
  `xe.ko`); fixed by restoring the stock module, purging NVIDIA, reinstalling
  the 29 kernel stack. Always run the §8.1 gates before loading.

## vLLM status (2026-08-13 — experimental, not the public path)

llama.cpp GGUF + DFlash n2 remains the **recommended public recipe**.

What changed after the 2026-08-10 “vLLM blocked” note:

- **Architecture:** [vllm#51655](https://github.com/vllm-project/vllm/pull/51655)
  merged 2026-08-14. No local overlay on current vLLM. The 2026-08-10 smoke
  used a Python overlay on the older Pi digest; that overlay is historical.
  Chat still leaked reasoning/channel text into `content` on that smoke.
- **INT4 dispatch:** the `cyankiwi` compressed-tensors W4A16 artifact selected
  `XPUwNa16LinearKernel`. A leftover `hf_quant_config.json` / NVFP4-style
  metadata still fails closed — same class of trap as Nemotron DFlash.
- **Paged-decode tuple:** `16,128,64,false,true,false` is in kernels `main`
  via [vllm-xpu-kernels#526](https://github.com/vllm-project/vllm-xpu-kernels/pull/526)
  (merged 2026-08-14). Do not apply `0002-muse-paged-decode-tuple.py` on a
  current kernels checkout. Missing that tuple aborted Muse text decode on
  the 2026-08-13 experimental overlay.
- **Speed:** a text-only n=3 C1 screen at 150 W measured **21.34 / 24.60**
  client post-first at p512/p8192 g128. That is **slower** than llama.cpp
  DFlash **26.8 / 22.9** at 128K, and it is n=3 — not a headline cell.

Still true: FP8-block 34.4 GB does not fit; XPU has no FP8 linear kernel.
Do not apply Nemotron `grouped_topk` / SSU patches to Muse. Public recipe
remains llama.cpp. A current XPU nightly has the architecture and the decode
tuple; it is not a measured cookbook generation.

## LocalMaxxing submission (2026-08-10)

Submitted via `lmx speed-test submit` (v0.1.31) — record:
`submissions/llamacpp-muse-glimmer-30b.json`. Payload: `tokSOut` 26.8 (DFlash
n2, p512/g128, 128K ctx, 230 W) · `tokSPrefill` 1,301 (llama-bench pp4096) ·
`contextLength` 131,072 · engine `llama.cpp d2f83055d SYCL` · Q4_K_XL.
Engine id `cmsnly2su00goo001wn6c98ly`, benchmark run `cmsnly2sy00gqo001ui1k5l67`.
