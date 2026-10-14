---
title: "Current Inference Architecture — Intel Arc Pro B70"
type: doc
date: 2026-07-16
status: final
---

# Current Inference Architecture — Intel Arc Pro B70

**Evidence cut:** 2026-07-18 20:34 CEST  
**Host:** `b70-host` (`<lab-host>`, Tailscale `<tailnet-ip>`)  
**Scope:** current production inference, its control plane and consumers, historical performance evidence, alternative runtimes, recovery, and optimization backlog  
**Method:** documentation and artifact review plus read-only live inspection. No process, profile, power, model, service, or network state was changed.

## Executive summary

The production system is a single-host, single-GPU OpenAI-compatible inference service. A pinned llama.cpp SYCL/Level Zero binary, build 9853 (`7af4279f4`), runs one fully GPU-offloaded Qwen3.6 27B Q5 model with MTP speculative decoding and a vision projector. It exposes one 131,072-token slot on `0.0.0.0:8765`. The active configuration is healthy, has run since 07:31 CEST without restart, and uses about 29.52 GB decimal (27.49 GiB) of the GPU's 32,656 MiB visible memory, leaving 4,506 MiB visible free. Its 180 W cap, 50 °C idle temperature, and sampled 46 W idle board power are normal. [E1, E2]

The primary local client is Grok Build through `127.0.0.1:8765/v1`. The Rock 5B Hermes Gateway can select the workstation endpoint, but the main Hermes profile defaults to a cloud model, not the B70. The old Pi Telegram bridge is stopped. A separate Hermes `work` profile intends to use the workstation but still points to the obsolete `<lab-host>:8080` endpoint and should be considered broken until tested or corrected. [E9, E10]

The newest comparable batch-1 evidence shows three distinct performance classes: Ornith/Qwen MoE decode at roughly 61–69 token/s with 4K prefill around 1.69–1.73K token/s; dense Qwen 27B MTP around 25 token/s at 165 W with about 613 token/s prefill; and Gemma 31B MTP around 24–27 token/s with 361–385 token/s prefill. These are not interchangeable with old `lmx` figures, wall-clock agent runs, or parallel-32 aggregate throughput. [E3]

The most immediate risks are operational rather than kernel-level:

1. `/mnt/<models-disk>` is 98% full with only 6.3 GiB available; `/` is 93% full with 15 GiB available.
2. The installed production unit has `Restart=no`; inference will remain down after a process crash. The OOM watchdog protects host availability by rebooting under extreme conditions, but it is not an inference-service watchdog.
3. Configuration and catalog drift is material: live profiles use K=`q8_0`, V=`q4_1`, while several labels and older recommendations claim K=`q5_0`, V=`q4_1`; profile IDs also say 256K where the effective context is 128K or 200K.
4. The running immutable binary is build 9853, while the current llama.cpp source checkout is newer (`0dc74e332`, described as `b10046-7-g0dc74e332`). Rebuilding from that checkout would not reproduce production.
5. The API binds all interfaces and relies on a static bearer key. The key is intentionally redacted here; rotation and firewall scope were not validated.

No optimization should begin until storage headroom, service recovery policy, endpoint/catalog drift, and a current reproducible baseline for the exact live KV pair are addressed.

## Evidence taxonomy and confidence

This report uses four labels deliberately:

- **Fact:** directly observed live or read from an authoritative active file.
- **Measurement:** produced by a recorded experiment under stated conditions.
- **Projection:** derived from component benchmarks or a performance model, not demonstrated end to end.
- **Assumption/open question:** plausible but not verified in this evidence cut.

Confidence is **high** when live state or raw artifacts agree, **medium** when a documented result has adequate methodology but was not reproduced in this audit, and **low** when evidence is historical, internally inconsistent, or lacks comparable conditions.

## System inventory

| Layer | Current inventory | Status / implication |
|---|---|---|
| Host | Ubuntu 26.04, kernel `7.0.0-27-generic`, ASUS TUF B450-PLUS GAMING | Live fact |
| CPU | Ryzen 7 5700X3D, 8 cores / 16 threads; performance governor | Server uses `-t 8` |
| RAM | 30 GiB; 25 GiB available at evidence cut; 15 GiB swap, effectively unused | `--no-mmap` makes overlapping model starts dangerous |
| GPU | Intel Arc Pro B70, Battlemage G31, `xe` driver, 32,656 MiB visible VRAM | Single accelerator; full offload |
| GPU state | 29,516,914,688 B allocated; 4,506 MiB visible free; 50 °C | Healthy idle, limited fit headroom |
| Power | 180 W cap; 2.01 s idle sample: 45.97 W card, 25.75 W package | Cap is not current draw |
| Root disk | 214 GiB total, 188 GiB used, 15 GiB free (93%) | High-risk capacity |
| Model disk | 229 GiB total, 211 GiB used, 6.3 GiB free (98%) | Critical capacity |
| Network | LAN `<lab-host>/24`; Tailscale `<tailnet-ip>`; server on `0.0.0.0:8765` | Old Tailscale address in `~/AGENTS.md` is stale |
| API | llama.cpp OpenAI-compatible `/v1`; `/health`, `/slots`, `/metrics`; bearer auth | Healthy at evidence cut |
| Scheduler | One slot, `n_ctx=131072`, speculative decoding enabled | Serial production workload |

There were no `xe` error/fault/reset or OOM-kill messages matching the audit filters in the current boot log. This is a point-in-time check, not a claim of lifetime error-free operation.

## Architecture and ownership

```mermaid
flowchart LR
    subgraph Clients[Consumers]
        GB[Grok Build\nlocal agent]
        HG[Hermes Gateway\nRock 5B]
        API[Direct OpenAI clients]
        LMX[lmx benchmark CLI\nRock 5B]
    end

    subgraph Control[Control plane on B70 host]
        SP[switch-profile / switch-any]
        AP[active-profile]
        PJ[profiles.json]
        US[systemd --user\nllama-profile.service]
        CAT[generated catalog]
        POW[xe hwmon\npower cap]
    end

    subgraph Data[Inference data plane]
        LS[llama-server b9853\nSYCL + Level Zero\n0.0.0.0:8765]
        SLOT[1 slot / 128K\nMTP-4 + vision]
        GPU[Arc Pro B70 32 GB\nfull model + KV offload]
        GGUF[/mnt/<models-disk>/gguf\nweights + mmproj + drafts]
    end

    subgraph Safety[Safety and fallback]
        OOM[oom-watchdog\nemergency host reboot]
        LEG[llama-hermes.service\ndisabled fallback :8080]
        PIN[pinned b9853 binary]
    end

    GB -->|127.0.0.1:8765/v1| LS
    HG -.->|selectable workstation alias\n<lab-host>:8765/v1| LS
    API -->|LAN / Tailscale| LS
    LMX -->|remote benchmark| LS
    PJ --> SP
    SP --> AP
    AP --> US
    US --> LS
    SP --> POW
    CAT -.-> HG
    LS --> SLOT --> GPU
    GGUF --> LS
    PIN --> US
    OOM -.->|host protection, not service restart| US
    SP -.->|mutual exclusion| LEG
```

### Request flow

1. A client sends an authenticated OpenAI-compatible request to port 8765.
2. llama-server's only slot tokenizes and evaluates the prompt. A concurrent request must wait because `--parallel 1`.
3. The SYCL backend dispatches through Unified Runtime/Level Zero to `SYCL0`; model layers, experts, KV cache, and vision projector execution are GPU-resident.
4. For the active dense model, its embedded MTP draft head proposes up to four tokens (`p_min=0.75`) and the main model verifies them.
5. Streaming tokens and llama timing metadata return to the client. Agent clients may then run tools and submit additional refills; those pauses are outside engine decode throughput.

### Configuration ownership and drift boundaries

| Object | Authoritative current source | Consumers | Known drift |
|---|---|---|---|
| Active profile ID | `inference/active-profile` | `run-profile`, status tools | ID says 256K; actual 128K |
| Launch definition | `inference/profiles.json` + `run-profile` | installed user service | Labels/docs retain old KV claims |
| Installed unit | `~/.config/systemd/user/llama-profile.service` | user systemd | Repository copy differs; installed unit has `Restart=no` |
| Workstation catalog | `inference/exports/workstation-catalog.json` | remote consumers | Generated 2026-07-16; K-cache descriptions stale |
| Main Hermes config | Rock 5B `~/.hermes/config.yaml` | active `hermes-gateway.service` | Local route selectable, cloud route default |
| Hermes work profile | Rock 5B `~/.hermes/profiles/work/config.yaml` | work-profile invocations | Obsolete `<lab-host>:8080/v1` |
| Grok Build config | `~/.grok/config.toml` | local Grok CLI | Defaults to local `qwen35-q5`; all aliases resolve to served model `active` |
| llama.cpp source | `~/llama.cpp` | development/builds | Source HEAD newer than production binary |

## Exact active production configuration

**Fact, observed live at 2026-07-18 20:34 CEST [E1, E2]:**

```text
profile: qwen27-mtp-q5-256k-vision-180w
service: llama-profile.service (user unit), active/running, PID 3025
started: 2026-07-18 07:31:12 CEST
restarts: 0
restart policy: no
model: /mnt/<models-disk>/gguf/Qwen3.6-27B-MTP-Q5_K_M.gguf
projector: /mnt/<models-disk>/gguf/mmproj-Qwen3.6-27B-F16.gguf
served alias: active
API: 0.0.0.0:8765, bearer-authenticated
health: {"status":"ok"}
slot: one idle 131072-token speculative slot (verified earlier in the same audit)
```

Reconstructed command, with the secret replaced:

```bash
~/llama.cpp/build-sycl-latest-l0/bin/llama-server \
  --model /mnt/<models-disk>/gguf/Qwen3.6-27B-MTP-Q5_K_M.gguf \
  --host 0.0.0.0 --port 8765 --api-key '[REDACTED]' -a active \
  -dev SYCL0 --no-mmap --metrics --slots --jinja \
  --parallel 1 --n-gpu-layers 99 -ncmoe 0 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q4_1 \
  --ctx-size 131072 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75 \
  -b 8192 -ub 4096 -t 8 \
  --mmproj /mnt/<models-disk>/gguf/mmproj-Qwen3.6-27B-F16.gguf \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
  --presence-penalty 1.5 --repeat-penalty 1 \
  --reasoning auto --reasoning-budget -1
```

API metadata reports 27,320,697,856 parameters, 19,823,058,944 model bytes, training context 262,144, and serving context 131,072. All 99 layers and all MoE experts (where applicable to other profiles) are assigned to the GPU; the active dense model does not use CPU expert offload.

### Runtime stack

| Component | Current fact |
|---|---|
| llama.cpp binary | build 9853, commit `7af4279f4`; IntelLLVM 2026.0.0 |
| Source checkout | `0dc74e332edee2616e4d8d9ab3b68dfc340fc14a`, `b10046-7-g0dc74e332` |
| Compiler / toolkit | oneAPI 2026.0.0, build 20260331 |
| Device runtime (last recorded stack snapshot, 2026-07-15) | Level Zero UR 20.2.0 `[1.14.37020]`; OpenCL NEO 26.05.037020; Level Zero CLI 1.2.42 / API 1.28.2 |
| Backend | llama.cpp SYCL, Level Zero device 0 |
| Build features recorded | SYCL, oneDNN, F16, graph, host-memory fallback, Level Zero API enabled |
| Environment | `ONEAPI_DEVICE_SELECTOR=level_zero:0`, `SYCL_DEVICE_FILTER=level_zero`, `SYCL_CACHE_PERSISTENT=0`, immediate command lists disabled, composite hierarchy, affinity mask 0, Sysman enabled |
| Explicitly absent | `GGML_SYCL_DISABLE_OPT` and `GGML_SYCL_DISABLE_DNN` are not set live |

The binary/source mismatch is intentional in effect even if not formally documented: production points at a stable build directory, not the active source tree. Reproducibility requires preserving the binary hash or its exact source/build inputs before any rebuild.

## Profile and model inventory

### Configured profiles

| Profile | Effective context | Power | Features | State |
|---|---:|---:|---|---|
| Qwen 35B A3B Q5 | 256K | 150 W | MoE, text | configured default, not active |
| Qwen 35B A3B Q4 XL | 256K | 150 W | MoE, text | configured |
| Qwen 27B Q5 MTP | 200K | 165 W | dense, MTP-4 | configured; misleading `256k` ID |
| Qwen 27B Q5 MTP | 128K | 200 W | dense, MTP-4 | configured; misleading `256k-230w` ID |
| Qwen 27B Q5 MTP vision | 128K | 180 W | dense, MTP-4, vision | **active** |
| Ornith 35B Q5 | 256K | 150 W | MoE, vision projector configured | vision smoke remains insufficiently evidenced |
| Gemma 4 31B Q4 | 128K | 165 W | dense/hybrid, external MTP | configured |
| Gemma 4 31B Q4 vision | 128K | 165 W | external MTP, vision | configured |

Every non-idle profile currently specifies K=`q8_0`, V=`q4_1`. This is a live configuration fact, not proof that this asymmetric pair has the best quality/speed/VRAM trade-off. Older validation studied K=`q5_0`, V=`q4_1` and cannot be silently transferred to K=`q8_0`.

### GGUF objects on the model volume

| File | Bytes | Role / fit note |
|---|---:|---|
| `Qwen3.6-27B-MTP-Q5_K_M.gguf` | 19,834,053,760 | Active dense weights; model metadata reports 19,823,058,944 payload bytes |
| `Qwen3.6-35B-A3B-UD-Q5_K_M.gguf` | 26,456,194,016 | Highest-quality configured MoE; tight fit at 256K |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | 22,360,456,160 | P4.3A control; SHA-256 `707a55…f4450` |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL-gateup.gguf` | 22,360,453,376 | Merged gate/up candidate; SHA-256 `2c50e2…8a08df` |
| `ornith-1.0-35b-Q5_K_M.gguf` | 24,729,130,848 | Fastest clean-suite decode result |
| `gemma-4-31B-it-Q4_K_M.gguf` | 18,323,731,456 | Gemma 31B MTP profiles |
| `dflash-draft-3.6-q8_0.gguf` | 1,849,481,856 | Separate draft, not active |
| `mmproj-F16.gguf` | 1,198,957,024 | Vision projector |
| `mmproj-Qwen3.6-27B-F16.gguf` | 927,607,360 | Active vision projector |
| `mmproj-deepreinforce-ai_Ornith-1.0-35B-f16.gguf` | 899,283,296 | Ornith vision projector |
| `mtp-gemma-4-31B-it.gguf` | 514,687,104 | Gemma external MTP draft |

Only already-recorded hashes were used; hashing all approximately 160 GB of GGUFs would have added avoidable I/O while the volume is critically full. The P4.3A experimental server binary hash recorded alongside those models is `4930ac…91af1a`.

### Memory model

The active allocation is authoritative: 27.49 GiB GPU allocation and 4.40 GiB visible free. File size alone is not VRAM use; context KV, compute buffers, projector state, allocator fragmentation, and runtime overhead must be included. Historical profile fit claims for 200K/256K are credible enough to retain as configured choices but are not current live measurements for K=`q8_0`, V=`q4_1`.

Because the server uses `--no-mmap`, model loading also consumes host memory transiently or as staging/cache. Two 20–26 GB models cannot coexist safely in 30 GiB RAM. `switch-any` therefore stops the legacy server before starting the profile service and waits for the old port/process to clear.

## Consumers and external control plane

### Grok Build on the B70 host

`~/.grok/config.toml` defaults to `qwen35-q5`, but that name is a client alias: all local entries send model `active` to `http://127.0.0.1:8765/v1`. Consequently the actual model is whatever `active-profile` has loaded; the alias name does not switch weights and can misdescribe the serving model. At this evidence cut, selecting `qwen35-q5` would still call the active Qwen 27B vision model unless the operator separately ran `switch-profile`. [E2, E10]

### Hermes and Telegram on Rock 5B

- `hermes-gateway.service` is the active messaging gateway and has an always-restart policy.
- Its main configuration defaults to cloud `openai-codex/gpt-5.6-terra`; `workstation-loaded` at `<lab-host>:8765/v1` is selectable through workstation aliases.
- The old `pi-telegram-bridge.service` is inactive; documents that describe it as the live Telegram path are historical.
- The Hermes work profile defaults to a workstation provider but uses stale `<lab-host>:8080/v1`, so it is not a verified route to current production.
- Gateway logs showed a cloud-provider 403/credit failure during inspection. That does not indicate a B70 inference failure.

The Rock 5B also hosts the ARM64 `lmx` CLI for remote benchmark submission. It is a benchmarking consumer, not a serving component.

## Runtime capability matrix

| Runtime | Local status | Proven strengths | Current blockers / limits | Production verdict |
|---|---|---|---|---|
| llama.cpp SYCL | Installed and active; b9853 | GGUF, MTP, vision/mmproj, quantized KV, 128–256K profiles, best proven single-stream local results | One slot; manual recovery; stable binary lags source; SYCL-specific maintenance | **Production** |
| vLLM / Intel LLM Scaler | Previously run in Docker (`0.14.0-b8.3.2` evidence) | Continuous batching and concurrency research; Qwen27 AutoRound loads with FP16; FP8 KV | BF16 GDN failure; about 18.3 output token/s single stream in the valid Qwen27 sweep; image/container not running now; several Jul-16 reproduction artifacts accidentally served Qwen3-0.6B | Research lane, not replacement |
| OpenVINO Model Server | Previously evaluated (`2026.2.1`) | Official Intel path; models load within about 2.5–3 min; around 19–20 GB VRAM for tested INT4 exports | Roughly 26–34 token/s depending model/test; chat template and reasoning extraction problems; text-only path failure in one Ornith test | Compatibility/research lane |
| SGLang | No installed/running evidence found | Potential serving/concurrency architecture | No B70 measurement; XPU speculative support not established in reviewed evidence | Unvalidated candidate |

The Jul-16 `reddit-1ub1lse` vLLM JSON/log set must not be cited as Qwen 35B performance: at least one nominal Qwen35 comparison served `Qwen/Qwen3-0.6B`. This is a workload-identity failure, not a minor measurement caveat.

## Reconciled performance evidence

### Most comparable current batch-1 suite

The 2026-07-16 clean TG/PP suite used build 9853 and the present K=`q8_0`, V=`q4_1` family. TG is engine generation throughput; PP4K is sustained 4K prompt processing. Results are measurements, not projections. [E3]

| Model / condition | TG token/s | PP4K token/s | Interpretation |
|---|---:|---:|---|
| Ornith 35B Q5, 150 W | 69.27 | 1,726 | Fastest tested single-stream MoE |
| Qwen 35B Q4, 150 W | 62.76 | 1,682 | Near Q5 decode; quant-quality comparison not part of speed test |
| Qwen 35B Q5, 150 W | 61.54 | 1,690 | Quality-oriented configured default |
| Qwen 27B Q5 MTP, 165 W | 25.07 | 613 | Dense; MTP active |
| Gemma 31B Q4 MTP, 165 W | 24.08–24.84 | 361–363 | DNN toggle produced no useful win |
| Gemma 31B Q4 MTP, 180 W | 26.56 | 385 | Power helps dense/hybrid path |
| Gemma 31B Q4 base, no MTP | 16.38 | — | MTP gain about 51.6% in this comparison |

Additional controlled observations:

- Gemma K/V=`q8_0/q8_0` reached 26.08 token/s versus 24.84 for `q8_0/q4_1`; this is a speed/VRAM observation, not a quality result.
- Reducing microbatch to 256 hurt prefill.
- Disabling oneDNN did not improve Gemma materially (24.41 versus 24.84 token/s in the recorded comparison), contradicting the older claim that `GGML_SYCL_DISABLE_DNN=1` was mandatory.
- Grok agent wall times of about 42–89 seconds include tool execution and repeated context refills and cannot be compared directly with engine TG.

### Dense Qwen MTP power evidence

The corrected diverse-prompt sweep, after rejecting the repeated-prompt inflation, measured [E4]:

| Cap | Engine token/s | Wall token/s | Meaning |
|---:|---:|---:|---|
| 165 W | 26.19 | 24.48 | Efficiency point |
| 180 W | 27.94 | 26.01 | Sustained production compromise |
| 230 W | 31.33 | 28.95 | Short-burst maximum, hotter and less efficient |

The current vision profile's label cites approximately 29 engine token/s at 180 W. That figure comes from a prior vision benchmark, not a request run during this read-only audit.

### Concurrency evidence

Production has one slot, so there is no production concurrency gain. A separate parallel-32 Ornith profile recorded roughly 143–165 aggregate token/s in different runs. Aggregate throughput at 32-way batching is not single-request decode speed and normally trades latency and per-slot context capacity. It is useful capacity evidence only if the target workload has enough simultaneous independent requests.

### Kernel research: measured versus projected

| Stage | Result type | Outcome | Decision |
|---|---|---|---|
| P0 FP8 | Component measurement | Kernel passed microbenchmark criteria | No engine integration |
| P1/P2 Q8 paths | Component/direct A/B | Repeated candidates below 1.10×; shared launch 1.0576× | Failed promotion |
| P3 transfer audit | Trace/measurement | Fused MoE path already active; per-token D2M mainly 993,280-byte logits | Original transfer premise disproved |
| P5 D2M ceiling | Projection from measurement | Perfect removal ceiling below 82 token/s (75.74/78.78 estimates) | Stop D2M lane |
| P4.0 GDN FP32 | Component measurement + projection | 1.30× component; only ~0.63% end-to-end projected | Insufficient alone |
| P4.1 GDN attention | Component measurement + projection | 1.24× component; ~5.3% end-to-end projected | Insufficient alone |
| P4.2 MoE block | Component measurement + projection | 1.596× component; combined 85.56–89.46 token/s projected | Required full integration test |
| P4.3A merged gate/up | **Server A/B measurement** | Control 68.68, candidate 69.53: +1.24% decode; prompt +24.09%; TTFT worsened 14.74% | **NO-GO** for decode objective |

The P4.3A result is the decisive evidence. The attractive 85–89 token/s number was a projection assembled from component coverage, not achieved server throughput. The merged model should remain preserved as an experimental artifact, but broad fusion work should not be resumed without a new bottleneck hypothesis and an end-to-end promotion gate. [E5, E6]

## Contradictions and invalidated claims

| Claim / artifact | Reconciliation | Confidence |
|---|---|---|
| Root `AGENTS.md` says production b9851 and old disk free space | Stale snapshot; live is b9853, root 93%, models 98% | High |
| Root docs recommend K=`q5_0`, V=`q4_1` | Relevant historical quality evidence, but live profiles use `q8_0/q4_1` | High |
| Catalog descriptions imply old KV pair | Generated data and descriptions are out of sync with profile args | High |
| `BASELINE-RESULTS.md` reports Qwen 10–12 token/s and a 67 token/s ceiling | Superseded by clean Jul-16 methodology; likely warmup/timeout/state contamination | Medium-high |
| Older analysis calls DNN disable mandatory | Clean controlled toggle found no win; live leaves it enabled | High |
| 143–165 token/s proves faster user decode | False comparison: it is parallel-32 aggregate throughput | High |
| All Jul-16 vLLM Reddit reproduction files represent Qwen35 | False; a logged endpoint served Qwen3-0.6B | High |
| P4 component projections predict 85–89 token/s server decode | Projection invalidated for the implemented P4.3A integration, which gained 1.24% | High |
| Old Telegram bridge is the live consumer | Stale; Hermes Gateway is active and primarily cloud-default | High |
| Grok alias `qwen35-q5` guarantees Qwen35 is loaded | False; the alias sends served model `active` and does not switch the profile | High |

Early thermal/power sweeps that used invalid flags, the wrong frequency sysfs path, JIT-contaminated first runs, or repeated prompts are retained only as research history. Corrected v4/diverse-prompt or clean-suite results take precedence.

## Bottleneck analysis by workload

### Short interactive TTFT

Short requests are dominated by fixed tokenization, scheduling, template, network/client, and kernel-launch overhead; short-prompt token/s is therefore unstable and should not be extrapolated to long prefill. One slot removes internal batch contention but creates queueing as soon as a second request arrives. Cold load is approximately 48 seconds before health, plus first-request JIT/warmup behavior. For agent workloads, tool calls and refills dominate end-to-end time even when engine TG is healthy.

### Long prefill

At 4K, current MoE profiles sustain about 1.68–1.73K token/s, Qwen27 about 613, and Gemma31 about 361–385. This separates architecture/backend costs much more clearly than tiny prompts. No current raw test in the reviewed clean suite proves the same rate at 128K or 256K; long-context attention scaling, memory pressure, and KV writes remain open measurements. The claim of “zero speed penalty through 512K” is historical and should not be treated as validated for every current model/KV pair.

### Decode

Dense Qwen/Gemma paths are bandwidth- and power-sensitive; MTP is the largest proven optimization, roughly 35% for earlier Qwen comparisons and 51.6% for the clean Gemma comparison. MoE decode sits around 61–69 token/s and gains little above its lower power envelope. Kernel micro-optimizations have not translated proportionally to the full server: the best integrated P4.3A decode gain was only 1.24%.

### Concurrency

`--parallel 1` is latency-oriented and prevents throughput scaling. The parallel-32 Ornith experiment proves batching can more than double aggregate output, but it does not establish acceptable per-request latency, long-context capacity, or agent reliability. vLLM remains attractive specifically for concurrency, but its valid B70 single-stream and model-compatibility evidence is weaker than llama.cpp.

### Model fit and KV cache

The active profile has only 4.4 GiB visible GPU headroom. Qwen35 Q5 plus 256K KV is a tight configuration, and historical fit data used different KV settings in some cases. K=`q8_0`, V=`q4_1` is unusual relative to the quality study; it may preserve key precision while saving value-cache space, but its end-to-end quality and long-context stability have not been documented with the same rigor as K=`q5_0`, V=`q4_1`.

### Host RAM and SSD

Host RAM is sufficient for one no-mmap server but not overlapping full model loads. Swap is an emergency buffer, not a safe inference capacity plan. Storage is the immediate hard constraint: new model downloads, container layers, OpenVINO exports, logs, or temporary repacks may fail or leave too little filesystem working space. The two 22.36 GB P4.3A control/candidate files alone account for substantial duplication, but deletion is outside this audit and requires an explicit retention decision.

## Failure handling, safety, and rollback

### Current safeguards

- `switch-profile` stops the current service, waits for port clearance, selects the profile, applies its cap, starts it, and polls model/health readiness.
- `switch-any` stops the legacy system `llama-hermes.service` before the user profile service, preventing two no-mmap model loads.
- `oom-watchdog.service` checks every five seconds and force-reboots after five OOM observations within 60 seconds or loss of `sshd`/`systemd-logind`.
- `systemd-oomd` and the kernel watchdog are enabled.
- The tested b9853 binary remains separate from experimental builds and the moving source checkout.

### Gaps

- `llama-profile.service` has `Restart=no`; there is no health-based inference restart.
- Emergency reboot is coarse recovery and can interrupt unrelated workloads. It does not guarantee the user service starts after reboot unless the user manager/linger path is healthy.
- The watchdog counts the same recent kernel kills repeatedly on each loop, so one OOM line can increase `oom_count` multiple times before the 60-second window resets. This may intentionally accelerate reboot but is not literally five distinct OOM events.
- Repository and installed unit files differ, making configuration-as-code unreliable.
- API exposure and static-key lifecycle were not audited end to end.

### Safe rollback procedure (documented, not executed)

1. Confirm the desired fallback model file and enough RAM/VRAM; record `inference-status`, current profile, service state, and power cap.
2. Use `~/inference/switch-profile qwen35-q5-256k` (or another known profile), which serializes stop/start and performs health polling.
3. Verify `/health`, `/v1/models`, `/slots`, a small authenticated completion, GPU allocation, temperature, and journal errors.
4. If the profile framework itself fails, stop `llama-profile.service` and ensure port 8765 and all llama-server PIDs are clear before considering the disabled `llama-hermes.service` fallback on port 8080.
5. Never run legacy and profile servers together. If using the legacy fallback, update clients explicitly because its endpoint, context, KV, reasoning, and restart policy differ.
6. Restore by stopping the fallback, confirming all old PIDs/ports clear, selecting the recorded prior profile, and re-running the same health checks.

The legacy unit is a last-resort compatibility fallback, not an equivalent rollback: it uses a Qwen35 Q5 model on port 8080, 32K context, flash attention off, different KV behavior, reasoning off, and `Restart=on-failure`.

## Evidence register

| ID | Source | Conditions / supports | Confidence | Limitation |
|---|---|---|---|---|
| E1 | Live `systemctl`, process command line, API, debugfs, hwmon, `free`, `df`, `dmesg`; 2026-07-18 20:34 CEST | Exact production state, health, resources, errors | High | Point in time; no inference request generated |
| E2 | `inference/profiles.json`, `run-profile`, `switch-profile`, `switch-any`, installed user unit | Configuration/control ownership and rollback | High | Installed and repository units drift |
| E3 | `~/inference/BENCHMARKS-TG-PP-2026-07-16.md` | Clean b9853 TG/PP comparisons with current KV family | High for speed | Quality and extreme context not tested |
| E4 | `~/B70-DOCS/docs/mtp-power-sweep-raw-results.md` and `~/B70-DOCS/b70-bench-results/engine-sweeps/` | Qwen27 MTP/power scaling | Medium-high | Historical config; not rerun today |
| E5 | `~/b70-lab/b70-engine/docs/EXPERIMENT_PROTOCOL.md`, `HANDOFF-2026-07-15.md`, and experiment reports | Kernel research gates and negative results | High for recorded experiments | Component results need engine integration |
| E6 | b70-lab-era experiment `2026-07-16-p4-3a-server-go-no-go` (predecessor repo; evidence not carried) | P4.3A full server A/B | High | One integrated candidate, not every proposed fusion |
| E7 | `B70-DOCS` vLLM/OVMS sweep docs and raw logs | Alternative-runtime compatibility/performance | Medium | Runtime versions and model exports can change |
| E8 | Jul-16 Reddit-reproduction raw logs | Demonstrates workload identity error | High | Invalid for intended Qwen35 comparison |
| E9 | Rock 5B service/config inspection | Live Hermes/Telegram routing | High | Secret values redacted; no message sent |
| E10 | `~/.grok/config.toml` | Local agent endpoint aliases | High | Client request was not generated |
| E11 | `~/AGENTS.md`, `B70-DOCS/docs/AGENTS.md` and research docs | Historical architecture and procedures | Medium | Several live facts are stale |
| E12 | `~/b70-lab/state/stack-2026-07-15-182708.txt` | Driver/runtime/build snapshot | Medium-high | Three days older than live inspection |
| E13 | P4.3A `sha256.txt` / `SHA256SUMS` | Exact experimental artifact identity | High | Full inventory hashes not computed |

## Facts, measurements, assumptions, projections, and open questions

### Facts

- llama.cpp b9853 is the only active inference runtime.
- Qwen3.6 27B Q5 MTP vision runs one 128K slot at a 180 W cap with full GPU offload.
- The service and API are healthy; no matching GPU/OOM error appeared in the current boot audit.
- The active KV pair is K=`q8_0`, V=`q4_1`.
- Hermes Gateway, not the old Pi bridge, is the active Telegram gateway; its primary default is cloud.
- Both storage filesystems have dangerously low headroom.

### Measurements

- Live allocation, temperature, sampled power, service memory, and disk values in the inventory.
- Clean Jul-16 TG/PP table and corrected MTP power table.
- P4.3A end-to-end decode improvement of 1.24%, with prompt improvement and TTFT regression.

### Assumptions requiring validation

- The active profile remains vision-correct after the server's warning that grounding tasks require at least 1,024 image tokens.
- The user service will be started automatically and become reachable after an OOM-triggered host reboot.
- K=`q8_0`, V=`q4_1` preserves acceptable long-context/tool quality across all profiles.
- 256K profiles still fit reliably under the current build, projector choices, and disk/RAM pressure.

### Projections, not achieved performance

- P4.2/P4.3 combined 85.56–89.46 token/s.
- Perfect D2M-removal ceilings and component coverage estimates.
- Any claim that SGLang or vLLM will beat llama.cpp at the actual target concurrency before a valid same-model A/B.

### Open questions

1. Why and when did profiles change from K=`q5_0` to `q8_0` while descriptions remained unchanged?
2. Is the production recovery intent manual (`Restart=no`) or automatic, and is user lingering/start-on-boot verified?
3. Which model/export artifacts can be archived off the 98%-full volume without losing rollback or reproducibility?
4. Should Hermes main/work profiles use local inference by default, and which endpoint is canonical?
5. What are measured TTFT and PP at 32K, 64K, 128K, and 256K for the exact current profiles?
6. Does the active vision profile pass grounding and general-image smoke tests with the new 1,024-token minimum recommendation?
7. What quality/tool-call regression, if any, results from V=`q4_1` relative to `q8_0` under long agent traces?
8. Is the static API credential scoped by host firewall/Tailscale ACL, and should it be rotated after appearing in process/unit metadata?

## Ranked backlog

| Rank | Action | Expected impact | Risk | Cost | Validation gate |
|---:|---|---|---|---|---|
| 1 | Establish storage retention policy and move/archive dispensable experiments, caches, duplicate exports | Prevent failed loads/downloads and filesystem incidents | Medium: deleting wrong artifact harms reproducibility | Low–medium | ≥15% free on `/mnt/<models-disk>`, ≥10% on `/`; hashes/manifest retained; rollback model preserved |
| 2 | Decide and test service recovery policy; reconcile installed/repository unit | Reduce inference downtime and config drift | Medium: restart loops can amplify OOM | Low | Forced non-OOM process exit recovers once, health passes, no duplicate server, backoff works; reboot test verifies user service |
| 3 | Make profile IDs, labels, catalog, Grok aliases, and Hermes endpoints derive from one source | Eliminate wrong-model and wrong-endpoint operation | Low | Low–medium | Automated lint asserts ID/context/power/KV/endpoint agreement; remote catalog refresh verified |
| 4 | Freeze a reproducibility manifest for production b9853 | Protect known-good rollback before development | Low | Low | Binary SHA-256, commit, CMake cache, oneAPI/driver snapshot, unit/profile hashes stored and restore-smoked |
| 5 | Run a current exact-config baseline for active Qwen27 vision | Establish trustworthy TTFT/PP/TG/quality baseline | Low–medium: load and thermal disturbance | Medium | Warmup discarded; diverse prompts; engine and wall metrics; vision smoke; 128K context sample; state restored |
| 6 | A/B K/V policies (`q8/q4`, `q5/q4`, `q8/q8`) on speed, fit, long-context retrieval, tool JSON, and vision | Resolve undocumented KV drift | Medium: context/OOM risk | Medium | Same binary/model/prompts/power; VRAM and quality gates; no OOM/xe errors |
| 7 | Benchmark concurrency at realistic 1/2/4/8 clients in llama.cpp, then valid same-model vLLM if demand exists | Size latency/throughput trade-off | Medium | Medium–high | p50/p95 TTFT, per-user TG, aggregate TG, success/quality, VRAM; exact model identity asserted |
| 8 | Validate and repair Hermes workstation routing only if local-first behavior is desired | Restore intended remote agent path | Low–medium: affects messaging | Low | Work profile calls `<lab-host>:8765/v1`, reports loaded model, fallback behavior tested |
| 9 | Address vision image-token warning and smoke all configured projectors | Avoid silent grounding/image failures | Low | Low | Standard image set, grounding test, minimum-token flag A/B, content and reasoning verified |
| 10 | Resume kernel work only from new end-to-end profiling evidence | Avoid repeating low-coverage optimization dead ends | High engineering opportunity cost | High | New profile shows ≥10% addressable server time and candidate clears ≥3% decode without TTFT/quality regression |

## Recommended next decision

Treat the existing b9853 Qwen27 vision service as a stable production baseline and do not tune it yet. First recover disk headroom, choose the intended restart behavior, and eliminate control-plane naming/endpoint drift. Then capture a same-binary, exact-KV, diverse-prompt baseline. Only that internally consistent baseline should seed performance work.

## Primary local sources reviewed

- `~/inference/` — all Markdown documents, profile/control scripts, unit copy, active profile, catalog, benchmark summaries and JSON baselines.
- `~/b70-lab/b70-engine/` — README, experiment protocol, handoff, P0–P5/P4.3 reports, raw summaries, telemetry and artifact hashes.
- `~/B70-DOCS/` — document map, host/runbook material, recommended configurations, KV/MTP/thermal research, vLLM/OVMS reports and raw sweep evidence.
- `~/llama.cpp/AGENTS.md`, current git metadata, active build metadata, and binary version.
- `~/AGENTS.md` — historical host guide, explicitly reconciled against live state.
- Rock 5B live service/config inspection for Hermes Gateway and the retired Telegram bridge.

Generated only after the source inventory, historical reconciliation, and live read-only inspection were complete.

## State-change addendum — 2026-07-18 22:52 CEST

The operator subsequently removed the two Gemma 31B production profiles and their catalog entries, then deleted `gemma-4-31B-it-Q4_K_M.gguf`, `mtp-gemma-4-31B-it.gguf`, and `mmproj-F16.gguf`. This was verified without modifying production.

- The active profile remains `qwen27-mtp-q5-256k-vision-180w`; the service retained PID 3025, zero restarts, healthy API state, 180 W cap, and 50 °C temperature.
- Six non-idle profiles remain: two Qwen35, three Qwen27, and one Ornith. No active profile or catalog record refers to the deleted Gemma files.
- `/mnt/<models-disk>` improved from 98% to 89% used, with 26.70 GB decimal (about 25 GiB) available.
- `/mnt/<other-disk>` is a separate writable ext4 volume with 49.40 GB decimal (about 47 GiB) available. No placement decision was made.
- `/mnt/<other-disk>` has 350.04 GB decimal (about 327 GiB) available but is mounted NTFS3 read-only and remains out of scope for model storage.
- Historical Gemma benchmark evidence remains valid as historical evidence; the operator preserved pre-removal control-plane files in `inference/backups/gemma-removal-20260718T204526Z/`.

Backlog item 1 (immediate model-volume pressure) is materially mitigated but not fully closed because `/` remains 93% used and `/mnt/<models-disk>` remains 89% used. Backlog item 4 is now complete: the exact active b9853 binary bundle, model, projector, build cache, profiles, control scripts, units, watchdog, and catalog have SHA-256 identities in `~/inference/docs/reproducibility/production-b9853-20260718T224645+0200.md` and its companion `.sha256` file. No benchmark, restart, profile switch, download, power change, or mount change was performed.

## Measurement addendum — 2026-07-18 23:28 CEST

The first exact-live API baseline is complete under `/mnt/<other-disk>/inference-models/baselines/qwen27-live-exact-20260718T231705+0200/`; its research report is `~/inference/docs/research/live-exact-baseline-20260718.md`.

- Four diverse 384-token text requests measured 26.11 ± 0.45 engine token/s, 724.8 ± 2.9 ms TTFT, and 88.66% mean MTP acceptance at 178.76 W measured card power.
- Prompt processing measured 620.10 token/s at 4,112 server tokens and 305.27 token/s at 32,768, confirming a 50.77% long-context rate loss. A repeated 32K request measured 307.26 token/s and retrieved `M00909` / `cobalt-909` exactly.
- Automatic unlimited reasoning exhausted 384-, 768-, and 1,024-token caps for short text and vision, leaving final `content` empty. The historical “512+” guidance is therefore invalid for the tested prompts.
- Vision recognized the logo semantics but repeatedly transcribed `INFERENCE` as `INFERENC`, never reached final content, and emitted non-consecutive-token-position warnings without a kernel fault.
- The extension stopped when available host RAM reached 19.86 GiB, below the 20 GiB research floor. This was a guard stop, not an OOM; production remained healthy on PID 3025 with zero restarts and unchanged configuration.

This changes the top research priority: request-level reasoning control is now ahead of additional decode optimization. Long-context prefill remains the principal engine bottleneck. The approved experimental storage policy is production weights in `/mnt/<models-disk>`, bounded raw/candidate work in `/mnt/<other-disk>/inference-models`, and no Windows/NTFS writes.
