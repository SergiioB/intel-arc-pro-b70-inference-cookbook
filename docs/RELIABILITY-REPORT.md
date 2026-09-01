# Intel Arc Pro B70 Inference Reliability Report

> **What works, what breaks, and which layer owns each bug — as of 2026-08-31.**
> Compiled from the cookbook's own dual-B70 measurements plus open upstream issues
> (intel/compute-runtime, vllm-project/vllm, intel/llm-scaler, LKML). Every claim
> links to its evidence. This is a reliability map, not a benchmark catalog:
> numbers live in [BENCHMARK-CATALOG.md](BENCHMARK-CATALOG.md).

**Status legend:** ✅ works out of the box · ⚠️ works with documented workaround · ❌ broken / no upstream fix

---

## TL;DR for Intel ISV teams

Single-card inference on the Arc Pro B70 is production-viable today (vLLM XPU
GPTQ-INT4 + MTP, llama.cpp SYCL). Multi-card serving — the configuration that
makes the B70 cost-competitive at 32 GB × N — is **not** production-viable:
every known dual/quad-B70 path requires workarounds that disable Intel's newer
runtime code paths (L0 V2, oneCCL SYCL kernels), and sustained multi-GPU load
produces unrecoverable driver-level wedges. The failure modes cluster in three
layers: the **xe KMD/GuC firmware** (wedges, ASPM brick), the **Level Zero /
Unified Runtime** (multi-device context creation, peer memcpy corruption), and
**oneCCL** (SYCL-kernel collectives on non-P2P topologies). None of these have
upstream fixes as of this writing.

| Area | Status | Owning layer | Key evidence |
|---|---|---|---|
| Single-card vLLM XPU serving | ✅ | — | [cookbook recipes](../README.md) |
| Single-card llama.cpp SYCL | ✅ | — | [cookbook recipes](../README.md) |
| Dual-B70 TP2 (vLLM, patched stack) | ⚠️ | oneCCL + L0 | [DUAL-B70-TP2.md](DUAL-B70-TP2.md), [#948](https://github.com/intel/compute-runtime/issues/948) |
| Dual-B70 TP2 (stock oneAPI 2025.3) | ❌ | L0 V2 / UR | [llm-scaler#463](https://github.com/intel/llm-scaler/issues/463), [torchlib-xpu#78](https://github.com/intel/torchlib-xpu/issues/78) |
| Sustained multi-GPU load (2–6 h soak) | ❌ | xe KMD / GuC | [#948](https://github.com/intel/compute-runtime/issues/948), [vllm#41663](https://github.com/vllm-project/vllm/issues/41663) |
| 4-GPU peer memcpy | ❌ | Level Zero / runtime | [#942](https://github.com/intel/compute-runtime/issues/942) |
| Linux driver bring-up (fresh install) | ⚠️ | xe KMD + firmware | [Debian 13 writeup](https://blog.anantshri.info/experiments-with-intel-arc-pro-b70-on-debian-13/) |
| ASPM power management (AMD platforms) | ❌ | PCI/ASPM + xe | [LKML report, May 2026](https://lkml.iu.edu/2605.0/10905.html) |
| Fan control (Pro B70) | ⚠️ | xe KMD (hidden PCODE mailbox) | exzile/intel-arc-pro-fan-control |

---

## 1. Single-card: the working baseline

Everything in this section is proven and repeatable; it anchors the rest of the
report so "broken" has a reference point.

- **vLLM XPU, GPTQ-INT4 + native MTP**: MoE (Qwen3.6-35B-A3B) at 170.9 t/s
  client post-first p512/g128 n=5; dense Qwen3.8-27B C1 106.7 t/s with MTP4.
  INT4 is the XMX-native format — this is the hardware's fast path, not a
  compromise (see [quantization-format-strategy.md](../research/quantization-format-strategy.md)).
- **llama.cpp SYCL**: production single-user engine; dense models via GGUF
  Q4_K_M/Q5_K_M; 256K context with K=q8_0/V=q4_1 at ~70 t/s.
- **Quant decision rule** (measured, not theoretical): AutoRound INT4 > GPTQ for
  conversational MoE — GPTQ stacks expert errors on MoE experts. XMX INT4 DPAS
  beats NVFP4 on this silicon.

- **Correctness gap — hybrid MTP + prefix caching (⚠️, single card).** With
  `--enable-prefix-caching` (which forces `mamba_cache_mode="align"` on
  Qwen3_5-class hybrids), MTP speculative decode and async scheduling on the V1
  runner, three upstream defects combine into **silent** output corruption
  (umbrella vllm#53912 / #43559; the reported signatures — short duplicated
  keys, fragments of another context, `content` empty with `finish_reason=stop`
  — match what we saw in production OCR traffic on the B70):
  1. the accepted-token D2H copy lands in step-N row order while step N+1's
     `_update_states`/`condense()` permutes the same pinned buffer before the
     event is awaited, and `prev_positions` permutes it a second time — a
     request decodes with **another request's** accepted count, and the wrong
     state is written back into the cache (vllm#53919, open; our V1-side audit
     confirms both trees of both pinned images carry the ordering bug);
  2. `MambaManager.find_longest_cache_hit` accepts `drop_eagle_block` and never
     acts on it, so a snapshot taken over draft positions that verification
     rejected stays reachable through the prefix cache (vllm#48375/#43650,
     open — #48375 also misses the fine-grained branch entirely);
  3. the align boundary state copy has no guard against a **backward**
     (`dest < src`) copy, which stomps the state `preprocess` just migrated
     (vllm#53505; includes the `precopy_mamba_align_fused_kernel` path that
     V2 executes).
  Text-only fail-closed ports with env gates: `patches/patch_fix_accepted_sync.py`,
  `patch_fix_eagle_drop.py`, `patch_fix_backward_copy.py`; GPU-free dry-run of the
  whole apply-list: `scripts/verify-mtp-apc-fixes.sh`. Mitigations without them:
  `--no-enable-prefix-caching` (drops `align`; upstream A/B 0/288 vs 16/288 with
  async also shows `--no-async-scheduling` suffices). The V2 runner keeps its
  counters GPU-resident, so (1) does not apply there — the patch detects this and
  says so — but (2)/(3) are runner-independent. On images ≥ 0.28.1, note the
  separate **performance** defect: when the drafter's KV groups cannot be
  identified, every group is treated as draft and mamba prefix reuse is silently
  disabled (vllm#52047; measured 0 % hits at 32k on nightly vs 91 % at C1 on
  these pinned builds).

**Reliability note:** single-card runs in the cookbook have never produced an
unrecoverable wedge; restarts were always clean. The failures below are
specifically multi-card and/or sustained-load phenomena.

---

## 2. Multi-GPU: what breaks, layer by layer

### 2.1 oneAPI 2025.3 / L0 V2: multi-device context creation fails (❌)

On dual B70 with torch ≥ 2.10 + intel-sycl-rt 2025.3.x:

- `ONEAPI_DEVICE_SELECTOR=level_zero:0,1` → single-device contexts OK, but any
  **explicit multi-device SYCL context** throws `UR_RESULT_ERROR_UNKNOWN` (2147483646).
- PyTorch XPU device enumeration returns 0 devices in fresh Docker builds
  (ABI mismatch between torch XPU and UR v2) — [torchlib-xpu#78](https://github.com/intel/torchlib-xpu/issues/78).
- Disabling L0 V2 (`SYCL_UR_USE_LEVEL_ZERO_V2=0`) restores enumeration, but then
  a **2 MiB allocation OOMs with 30.3 GiB free** unless `SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0` is also set — the legacy L0 adapter reports false USM residency pressure.
- oneCCL SYCL-kernel all-reduce then fails with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`; only `CCL_ENABLE_SYCL_KERNELS=0 CCL_ALLREDUCE=direct` passes.

Full chain: [intel/llm-scaler#463](https://github.com/intel/llm-scaler/issues/463).
The workaround stack disables the newer code paths Intel is actively investing
in; it is a stopgap, not a fix. torch 2.9.1 + sycl-rt 2025.2.1 still works —
this is a **regression introduced in the 2025.3 runtime stack**.

**Owner:** Level Zero V2 adapter / Unified Runtime (intel/compute-runtime).

### 2.2 oneCCL on non-P2P desktop topologies: zeMemOpenIpcHandle (⚠️ → ❌ under load)

Desktop dual-B70 boards have no GPU P2P. Above a small message threshold,
oneCCL's large SYCL all-reduce exchanges peer device-memory handles and the xe
driver rejects them: `ze error at zeMemOpenIpcHandle, code: ZE_RESULT_ERROR_INVALID_ARGUMENT`.

The cookbook's documented fix (validated on three dual-B70 hosts):
spawn-time per-worker `ZE_AFFINITY_MASK=<rank>` injection plus raising the four
oneCCL simple-algorithm thresholds to Intel's documented workaround value
(4294967296). Details and repro: [DUAL-B70-TP2.md](DUAL-B70-TP2.md), confirmed
via cookbook [issue #8](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/issues/8)
on two additional hosts — one of which crashed without the env contract despite
matching drivers.

**Owner:** oneCCL algorithm selection + xe IPC path. Intel has documented this
workaround for dual B60 ([llm-scaler#594](https://github.com/intel/llm-scaler/issues/594))
but not shipped a topology-aware fix.

### 2.3 Sustained load: permanent GPU wedge (❌, unrecoverable without restart)

Under sustained multi-request inference (vLLM TP=2, 8–12 concurrent requests,
mixed 4k–32k contexts), **once every 2–6 hours** one card takes a GPU page fault
and the xe driver resets its compute/copy engines:

```
xe 0000:c7:00.0: [drm] Tile0: GT0: Engine reset: engine_class=ccs, logical_mask: 0x1, state=0x289
xe 0000:c7:00.0: [drm] Tile0: GT0: Engine reset: engine_class=bcs, ...
xe 0000:c7:00.0: [drm] Fault response: Unsuccessful -ENOENT / -EINVAL
```

After the reset the Level-Zero context is **permanently wedged**: in-flight SYCL
kernels never complete, the vLLM worker hangs (`TimeoutError: RPC call to
sample_tokens timed out` → `EngineDeadError`), and only a full process/container
restart recovers. Both cards are affected across multiple rigs; the dying
scheduler step can be tiny (9 decode tokens, KV 14%) — this is not a pressure
bug. The same ccs engine-reset signature appears **without vLLM or oneCCL at all**
(darktable on B580 via OpenCL), pointing at driver/firmware level.

Triage datapoint from an independent dual-B70 rig ([vllm#41663](https://github.com/vllm-project/vllm/issues/41663)):
on kernel 6.17 + GuC 70.44 the ccs resets are **recoverable** (2-hour soaks, zero
engine resets with `CCL_ENABLE_SYCL_KERNELS=0`); on kernel 7.0 + GuC 70.58 +
compute-runtime 26.05 they come **with** `Fault response: Unsuccessful` and wedge
permanently. Same GPU model, same workload class → the permanent-wedge mode
correlates with the newer KMD/GuC/compute-runtime combination.

**Owner:** xe KMD + GuC firmware (kernel 7.x era). Bisect candidates offered by
the reporter: GuC 70.58→70.44, kernel xe, compute-runtime 26.05→26.22 — no
upstream response yet as of this writing.

### 2.4 Quad-B70: peer memcpy silently corrupts data (❌)

On 4× Arc Pro B70, the runtime reports peer access available via
`ext_oneapi_can_access_peer(...)`, but actual `queue.memcpy` between devices
**returns incorrect bytes with no error**; larger copies escalate to
`UR_RESULT_ERROR_DEVICE_LOST`. Model output becomes gibberish; only host-staged
copies are correct. The corruption reproduces even after verifying
`ACSCtl=0` on every switch port — the runtime claims a capability the path does
not deliver, and the copy API completes "successfully."

**Owner:** Level Zero peer-access detection + xe P2P path. Evidence:
[intel/compute-runtime#942](https://github.com/intel/compute-runtime/issues/942).

---

## 3. Platform / driver bring-up: the Linux experience (⚠️)

Fresh-install pain that keeps recurring across community reports:

1. **No PCI alias in stock kernels.** Debian 13's kernel 6.12 `xe` driver has
   zero aliases for the B70 (`8086:e223`); `force_probe` taints the kernel and
   still fails to bind. Requires kernel ≥ 7.x (trixie-backports).
2. **GuC firmware version coupling.** The B70 needs GuC ≥ 70.54; Debian's
   `firmware-linux-nonfree` ships 70.40.2 → card fails with
   `Failed to initialize uC (-ENXIO)`. Fix is a manual single-file firmware
   replacement from the upstream linux-firmware tarball. The version coupling is
   undocumented anywhere user-facing.
3. **Board-specific fan-control blob missing.** `fan_control_8086_e223_*.bin`
   fails to load; compute still works, but stock xe exposes **no usable PWM fan
   controls** for the Pro B70. Community fix (exzile/intel-arc-pro-fan-control)
   patches the xe module to expose the PCODE mailbox as hwmon — requires secure
   boot disabled or per-kernel shim signing, and breaks on every kernel upgrade
   without DKMS-style automation.
4. **ASPM_L1.1 bricks the card at boot (AMD platforms).** With
   `pcie_aspm.policy=powersupersave`, the B70 cannot wake from D3cold:
   `Unable to change power state from D3cold to D0, device inaccessible`, then xe
   misclassifies the dead PF as an SR-IOV VF and probe fails with -EPROTO. The
   card is permanently inaccessible until reboot with a different policy. A
   quirk patch (disable L1SS on the BMG upstream port `8086:e2ff`) was submitted
   to LKML in May 2026 and works, but has **not merged**; users lose ~25 W of
   idle savings until it does. Evidence: [LKML](https://lkml.iu.edu/2605.0/10905.html).

**Owner:** xe KMD + linux-firmware packaging + PCI/ASPM core. Individually each is
a one-file fix; collectively they make "install Linux, plug in B70" a multi-day
exercise for non-experts.

---

## 4. What this means (and what would close the gaps)

Ranked by impact on real serving deployments:

1. **Fix L0 V2 multi-device context creation** (llm-scaler#463). This is the
   single highest-leverage item: it blocks *all* multi-GPU PyTorch/vLLM paths on
   current oneAPI, and the current workaround disables the new runtime Intel is
   building toward. A regression test for "two B-Series GPUs visible → explicit
   multi-device context" would have caught this before 2025.3 shipped.
2. **Make the xe KMD recover from engine resets** (compute-runtime#948). Even if
   the root cause of the ccs/bcs resets is a GuC firmware bug, a permanently
   wedged userspace context after a reset is a driver contract violation: the
   runtime should fail fast with a catchable error, not hang in-flight kernels
   forever. This is what converts "occasional restart" into "unreliable service."
3. **Topology-aware oneCCL** (DUAL-B70-TP2.md). Detect non-P2P desktop layouts
   and default to simple/tmp-buffer algorithms instead of requiring operators to
   know four magic env vars. The workaround values are Intel's own documented
   numbers — productizing them is packaging, not new engineering.
4. **Honest peer-access reporting** (compute-runtime#942). `can_access_peer`
   returning true on a path that silently corrupts data is worse than returning
   false: it makes the fast path the *wrong* path.
5. **Ship the B70 bring-up as a supported path**: PCI aliases + matching GuC in
   distro kernels/firmware packages, documented firmware coupling, and either
   stock fan controls or an officially blessed DKMS module.

**The product opportunity:** items 2–4 are exactly the class of problems a
reliability tool could own — a pre-flight validator that checks the KMD/GuC/
compute-runtime/oneCCL combination against known-good matrices before deployment,
plus a soak harness that detects wedge signatures in dmesg and auto-recovers.
The cookbook's dual-B70 rig already reproduces items 1–3 deterministically; the
gap is packaging that knowledge as something Intel can test against.

---

## Appendix: evidence index

| # | Issue / source | Layer | State (2026-08-31) |
|---|---|---|---|
| 1 | [intel/llm-scaler#463](https://github.com/intel/llm-scaler/issues/463) | L0 V2 / UR | open, workaround only |
| 2 | [intel/compute-runtime#948](https://github.com/intel/compute-runtime/issues/948) | xe KMD / GuC | open, bisect offered |
| 3 | [intel/compute-runtime#942](https://github.com/intel/compute-runtime/issues/942) | Level Zero P2P | open |
| 4 | [vllm-project/vllm#41663](https://github.com/vllm-project/vllm/issues/41663) | TP2 dual-B70 | cross-referenced, triage datapoints |
| 5 | [intel/torchlib-xpu#78](https://github.com/intel/torchlib-xpu/issues/78) | torch XPU / UR ABI | open |
| 6 | [LKML: PCI/ASPM BMG L1.1 brick](https://lkml.iu.edu/2605.0/10905.html) | PCI/ASPM + xe | patch submitted, unmerged |
| 7 | [cookbook DUAL-B70-TP2.md](DUAL-B70-TP2.md) | oneCCL topology | documented workaround, 3 hosts validated |
| 8 | [cookbook issue #8](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/issues/8) | community repro | closed with fix confirmed |
| 9 | [Debian 13 B70 writeup](https://blog.anantshri.info/experiments-with-intel-arc-pro-b70-on-debian-13/) | bring-up | community documentation |

*Compiled by the cookbook author from first-hand dual-B70 measurements and open
upstream reports. All issue links verified reachable on 2026-08-31.*
