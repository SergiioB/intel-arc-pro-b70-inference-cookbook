# quantpath: EXL3 decode probe, turboderp EXL3 3.00/3.50 bpw on exl3xpu, and the no-engine-code 3-bit test

Agent `quantpath`, wave 3, 2026-11-19. Follows `research/2026/b70-breakthrough-20261118/QUANT-AND-EXL3-PATH.md`.
Status: **draft, updated 08:25 CET. Step 1 measured (job 2190); steps 2 and 3 queued.** Sections are updated as jobs land.

## 1. Result

**Step 1 (measured, job 2190, card 1, 08:17 CET).** TensorFold's `exl3_gemv` probe kernel runs at **8–18 GB/s**.
TensorFold's own int4 kernel runs at **473–488 GB/s** on the same shapes in the same run. Per weight, the probe
is **28.7x slower** at 1 row and **62x slower** at 8 rows. Its time does not change with the bit width (gate|up,
1 row: 5.04 / 5.10 / 5.11 ms at 4 / 3 / 2 bits): it is bound by decode work, not bytes, so fewer bits buy nothing.
Priced as is, EXL3 inside TensorFold with this kernel would decode at about 1.3 tok/s serial at any width.
The probe is a scalar speed sketch (per-row FMAs, four sub-group shuffles per weight, no DPAS). It does not bound
a production kernel: RL-183's DPAS mul1 kernel reached 465–512 GB/s at 4 bits on this card. So the 4.0 / 3.5 / 3.0
bpw prices stay at the estimates in `QUANT-AND-EXL3-PATH.md` §3.2 (31–33 / 34–36 / 38–40 serial). Reaching them
means writing a new kernel, about 29x faster than upstream's probe, not tuning this one. Side result: the int4
kernel streams 532–549 GB/s once its g64 scales and biases are counted (87.5–90.3 % of the 608 GB/s pin). That
confirms the 88 % figure.
2. Step 2: turboderp `3.00bpw` and `3.50bpw` downloaded to `$DATA/samsung980/models/`, sha256 of every LFS
   file checked. The 3.50 bpw artifact is per-tensor mixed K=3/4, which the mixed-K exl3xpu image supports (K=2..7).
   Jobs 2481–2484 are queued.
3. Step 3: the base, the tool, the calibration data and the target container are all on disk. The AutoRound →
   MLX export → 4-bit repack pipeline passed a CPU dry run on a tiny random Qwen3.5 model, after five fixes
   (section 5). Repack semantics were checked against AutoRound's own packer (2/3/4/5/6/8 bits exact) and against
   the real TensorFold template (dequantised template vs bf16 base: 9.2 % relative error, the normal 4-bit RTN level).
   Jobs 2490–2496 are queued.

**Risk to the schedule:** at 07:45, 21 jobs with 16.2 h of timeout budget sit ahead of 2480 in the queue. The
scheduler does not backfill past a blocked big job. Without a renumber, probably none of this runs before 14:00.

## 2. What you need

| Item | Value |
|---|---|
| Cards | card 1 (`0000:0f:00.0`, 230 W, 2,800 MHz) for every timed job; AutoRound takes `any` card |
| TensorFold | `$TF/src-tp3586` @ `2286abc0c`; dev runner `zig-out/bin/xpu-qwen27` sha256 `50da998b…6005`; server `zig-out/xpu/bin/tensorfold-native` sha256 `e6e59bfd…5383` (the same binary as the wave-2 routecmp cells) |
| exl3xpu | image `ghcr.io/sergiiob/exl3xpu-mixedk@sha256:933eda54…79d3a`, arguments as `scripts/tmp/routecmp_20261118/c_exl3.sh` |
| EXL3 artifacts | `turboderp/Qwen3.8-27B-exl3` branch `3.00bpw` @ `6fe61ad620abfe97c5b49f9722c2bceeea4ccc28` (13.84 GB) → `$DATA/samsung980/models/qwen38-27b-exl3-3.00bpw`; branch `3.50bpw` @ `8351c54ef11a63e38bca978655086d7998b8a7f9` (15.36 GB) → `…-3.50bpw`. 3.50 = 137 tensors at K=3, 262 at K=4, `o_proj` of one layer at 5, head at 6 |
| bf16 base | `$DATA/sata1/models/Qwen3.8-27B-bf16` (59 GB, 18 shards, `Qwen3_5ForConditionalGeneration`) |
| AutoRound | source `$DATA/samsung980/cache/auto-round-src` @ `a5ea56e6`, venv `$DATA/samsung980/cache/arvenv` (torch 2.12.1+xpu, transformers 5.12.1), the tree from the Bolt export (`hatchetfish/results/2026/10/nemotron35-bolt-autoround-w4a16-20261016/PROCESS.md`) |
| Calibration | NeelNanda/pile-10k from the local datasets cache (offline), 128 samples × 2,048 tokens, seed 20261119. Not the battery and not our docs corpus (which could contain battery text) |
| Template | `$TF/models/Qwen3.8-27B-MLX-4bit` (MLX affine 4-bit g64, BF16 scales and biases) |
| Step 3 budget (estimates) | RAM: AutoRound under a 30 GiB cgroup cap, job refuses below 50 GiB MemAvailable (20 GiB reserve); repack under 12 GiB. VRAM about 9 GB on one card (RL-117: 8.65 GB at W4). Disk about 30 GB per arm on `$DATA/samsung980` (334 GB free). Time: AutoRound about 1.7 h per arm (RL-117: 70 s per block at 128×1,536; ×1.33 for 2,048 tokens; 64 blocks), repack about 20 min CPU, TensorFold eval about 1 h per model. Both arms plus control: about 7.3 h of queue time |

## 3. Timeline (CET, 2026-11-19)

| Clock | Event |
|---|---|
| 07:29 | Brief read; 14:00 stop noted |
| 07:30 | Probe source read; job 2480 queued (binary sha checked, stale-source check in the script) |
| 07:31–07:41 | HF refs and trees read without a token; both branches downloaded with sha256 per LFS file |
| 07:33 | exl3xpu jobs 2481–2484 queued (refuse to start until `REVISION.json` exists) |
| 07:35–07:41 | AutoRound dry run on a tiny random Qwen3.5 (CPU only, GPUs hidden): five fixes, section 5 |
| 07:42 | Step 3 jobs 2490–2496 queued |
| 07:44 | Lead: probe renumbered 2190, MTP3 jobs 24651/24653, guards to 23:30; health canary already `HEALTH PASS` (07:41) |
| 07:47 | Mixed jobs split: depth/canary exclusive on card 1, batteries `any` and non-exclusive (24652, 24654, 2492b, 2493b, 2496b); combined eval files moved to `quantpath/withdrawn-jobs/` |
| 08:17:00–08:17:23 | Job 2190 ran (23 s), rc 0, card 1 52 → 56 °C, 230 W and 2,800 MHz unchanged. Ran after the health canary passed, so no rerun |

## 4. Steps with exact commands

### 4.1 Step 1, what `XPU_EXL3_PROBE` measures

`qwen27_main.zig:446-488`, after a normal prompt and decode. For layer 0's `gu` (gate|up, K = 5,120) and `down`
(K = 17,408), at T = 1 and 8 rows:

- TensorFold's own int4 GEMM (`q.gemm`, the production kernel family). Best of 189 timed calls, each launch + `d.sync()`.
  Bytes counted = K × N / 2 (codes only, no scales).
- `exl3_gemv` (`decode.cl:209`) on random trellis bits at 4, 3 and 2 bits, mul1 codebook. Best of 94 calls.
  Bytes = K × N × bits / 8. K is split into slices so that about 2,048 work-groups run.

Limits: each time includes a launch and a host sync (about 8.5 µs measured by tpdiag job 104). `exl3_gemv` uses
scalar fp32 FMAs per row, not DPAS. Its 8-row number is not a verify-kernel number, and its Hadamard
transforms are not included.

```bash
# queue job 2480 (cards 1, exclusive): scripts/tmp/quantpath_20261119/exl3_probe.sh OUT
ZE_AFFINITY_MASK=1 XPU_ATTN_PREF=256 XPU_CTX=8192 XPU_EXL3_PROBE=1 \
  $TF/src-tp3586/zig-out/bin/xpu-qwen27 $TF/models/Qwen3.8-27B-MLX-4bit .../ids/code-0.ids 3
```

### 4.2 Step 2

```bash
# download (no token; every LFS file sha256-checked against the HF tree API)
scripts/tmp/quantpath_20261119/dl_exl3.sh 3.00bpw 6fe61ad6… $DATA/samsung980/models/qwen38-27b-exl3-3.00bpw tree-3.00bpw.json
scripts/tmp/quantpath_20261119/dl_exl3.sh 3.50bpw 8351c54e… $DATA/samsung980/models/qwen38-27b-exl3-3.50bpw tree-3.50bpw.json
# queue jobs (c_exl3.sh arguments, model path swapped, label b70.owner=quantpath)
2481 q_exl3.sh wave3/quantpath/exl3-3.00-mtp3   mtp3   …-3.00bpw "depth battery3"   exclusive
2482 q_exl3.sh wave3/quantpath/exl3-3.00-serial serial …-3.00bpw "depth"            exclusive
2483 / 2484: the same for 3.50bpw
```

`depth` = `depth_bench.py` at 0/8K/32K/128K, code and prose, 1 rep, the wave-2 prompts. `battery3` = the 329-item
battery three times (`routecmp_20261118/battery.py`), each graded with `breakthrough_20261118/grade.sh`.

### 4.3 Step 3

```bash
2490 ar_job.sh B          # AutoRound, turboderp-3.50 allocation (MLP 3/4 by layer, mixers 4), asym g64, MLX export
2491 repack_job.sh B      # CPU: codes into MLX 4-bit g64 storage -> $DATA/samsung980/models/qwen38-27b-ar-w3g64-armB-in-mlx4
2492 tf_eval.sh <armB> wave3/quantpath/tf-armB "canary depth battery"     # card 1, exclusive
2493 tf_eval.sh <stock MLX 4-bit> wave3/quantpath/tf-ctrl "canary depth battery"
2494-2496: arm A (uniform 3-bit)
```

`tf_eval.sh` = the unchanged binary, `--tp 1 --parallel 1 --context 139264 --prompt-cache-gib 0 --no-thinking`,
one warm-up, then:

- `canary.py`: 6 exact answers, including `sum(i*i for i in range(4))` = 14.
- `depth_bench.py` drafted and serial (`"draft": false`) at 0/8K/32K, to compare sha256.
- The 329 battery, drafted, then graded.
- `rounds=` / `accepted=` per request copied from `server.log` into `rounds.txt`.

## 5. Fix table (dry run, CPU only, tiny random Qwen3.5: 4 layers, hidden 256)

| # | Literal symptom | Cause | Fix | How verified |
|---|---|---|---|---|
| 1 | `OSError: Can't load image processor for '.../tiny'` | the tiny model lacked `preprocessor_config.json` | copied the base model's processor files | next run reached AutoRound |
| 2 | `Qwen2VLImageProcessor requires the PIL library ... Torchvision` | AutoRound treats a dir with `preprocessor_config.json` as multimodal and loads the image processor; arvenv has no PIL/torchvision | mark `qwen3_5` LLM-only (`_LLM_ONLY_MODEL_TYPES.add`) in `ar_q38.py`; the text model is quantized, vision is not | run reached block 0 |
| 3 | `RuntimeError: Expected attn_mask dtype to be bool or float ... got long int` at `model.layers.3` | AutoRound caches block 0's inputs (a GDN layer, 2-D padding mask) and replays them into every block; the first full-attention block gets a long mask | drop the mask in `Qwen3_5DecoderLayer.forward` during calibration: batch 1, no padding, so SDPA's `is_causal` path is the same causal mask | all three arms (R4/A/B) export, rc 0 |
| 4 | arm B layer names `model.language_model.layers.N…` never matched (text path names them `model.layers.N…`) | the LLM path loads the text model | names rewritten; widths looked up with the original names | export `config.json` lists per-layer bits |
| 5 | `RuntimeError: expected m1 and m2 to have the same dtype ... float != BFloat16` with disk streaming on CPU | CPU runs without bf16 autocast; streamed weights stay bf16 | not fixed on CPU (it does not occur in the path that will run if XPU autocast holds; NOT VERIFIED). The job tries disk streaming first and falls back to the mmap path, which passed the dry run; both logs kept | dry run of the mmap path rc 0 |
| 6 | `repack_job.sh` syntax: nested double quotes in a `python3 -c` | my typo | single quotes | `bash -n` |

Repack checks (CPU): AutoRound `_pack_weight_mlx` → my `unpack` is exact at 2/3/4/5/6/8 bits on random codes.
`pack4` → `unpack` is exact. The real template (layer 3 `down_proj`, `q_proj`; layer 0 `in_proj_qkv`) dequantises to
9.17–9.24 % relative error against the bf16 base, so the nibble order and the orientation match. In the tiny-model
repack the 3-bit codes gave 0.195 relative error against the 4-bit RTN template's 0.091.

## 6. Verification

| Job | Raw files | Check |
|---|---|---|
| 2190 probe | `exl3-probe/probe.log` (16 timing lines), `env.txt` (binary sha256 `50da998b…`, git `2286abc0c`), `telemetry.jsonl` (cap, temp, clock, energy before/after), `queue/logs/quantpath-exl3-probe.log` | rc 0; the dev runner decoded its 2 tokens (32.9 tok/s) before the probe; card cap unchanged |

Other jobs: to fill as they land.

## 7. Results and claim boundary

### 7.1 Step 1: `XPU_EXL3_PROBE` (measured, card 1, 230 W, 2,800 MHz, 52–56 °C; best of 189 int4 / 94 EXL3 calls, launch + sync included)

| Shape (layer 0) | Rows | int4 (TF) ms / GB/s | EXL3 4-bit ms / GB/s | 3-bit | 2-bit |
|---|---|---|---|---|---|
| gate\|up K 5,120 × N 34,816 | 1 | 0.183 / 488 | 5.037 / 18 | 5.095 / 13 | 5.105 / 9 |
| gate\|up | 8 | 0.185 / 482 | 11.817 / 8 | 11.823 / 6 | 11.783 / 4 |
| down K 17,408 × N 5,120 | 1 | 0.092 / 483 | 2.858 / 16 | 2.905 / 12 | 2.886 / 8 |
| down | 8 | 0.094 / 473 | 5.454 / 8 | 5.453 / 6 | 6.561 / 3 |

GB/s use the probe's own byte counts: int4 = K×N/2 (codes only), EXL3 = K×N×bits/8.

Derived (arithmetic on the table):
- Per-weight time ratio, EXL3 over int4, weighted 2/3 gate|up and 1/3 down: 28.7 / 29.1 / 29.1 at 1 row (4/3/2 bits);
  61.9 / 61.9 / 65.7 at 8 rows. Weight rate: 35 G weights/s (EXL3 probe) against 974 G/s (int4).
- Priced with this kernel: serial `1 / (3.8 ms + 26.9 ms × 28.7)` = **1.3 tok/s** at any width; drafted, using the 8-row
  ratio as a floor for the 16-row verify, `5.39 / (11.1 ms + 30.9 ms × 61.9)` = **2.8 tok/s**.
- int4 with g64 metadata counted (× 4.5/4): 532–549 GB/s = 87.5–90.3 % of pin.

Claim boundary: this measures upstream's probe kernel, not the EXL3 format. The probe has no DPAS, decodes each
weight with four `sub_group_shuffle`s, a 32-bit multiply and a byte sum, and leaves out the Hadamard transforms.
One run per cell (best of N calls), random trellis bits, layer 0 shapes only. The format's achievable rate on Xe2
is still RL-183's isolated DPAS kernel, 465–512 GB/s at 4 bits (82 % of pin). No EXL3 kernel exists that has
run inside TensorFold at speed.

Not tested yet: steps 2 and 3 (queued).

## 8. If you do this again

Negotiate queue numbers before preparing. With a strict first-in-first-out queue and no backfill, numbers decide
whether anything runs.

## 9. Adapting it

`repack_mlx4.py` takes any AutoRound MLX export (any mix of 2/3/4-bit, group 64) and any MLX 4-bit g64 template
with the same layer names.
