# Host stability and hardware notes — third-party findings

Status: **reported, largely not reproduced here.** This page records host-level findings from
[steveseguin/b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab) (Unlicense,
public domain) so that a future host freeze, `-EINVAL` at load, or dead peer-polling experiment
starts from a cause instead of a fresh guess. Exactly one item has been checked on this host; the
rest are his measurements, on his hosts, quoted with their caveats.

## How to read this page

| Field | Meaning |
|---|---|
| Claim | What his record says, with his numbers only |
| Source | Path inside `steveseguin/b70-optimization-lab` |
| Our status | `new` (not in this cookbook), `have` (already recorded here), `contradicts` (this repo said otherwise), `verified here` (read-only check on this host) |
| Verify | A read-only command you can run before believing the claim on your own board |
| Risk | What acting on the item can cost |

**His own limits, kept.** Two hosts only — a 2-card Threadripper Gen4 machine and a 4-card
Threadripper PRO machine. No Intel desktop CPU, no AM5 board, no Gen3-x8, x1 riser or Thunderbolt
data. Several items are correlation, not causation: a change and a symptom in the same boot is not
an experiment, and he says so in the source. Nothing here is a rate to quote — this page publishes
causes, not numbers, per [EVIDENCE-LEDGERS.md](EVIDENCE-LEDGERS.md) rule 8.

| Item | One-line claim | Our status |
|---|---|---|
| [H1](#h1-ecc-is-a-capacity-trade-not-a-bandwidth-knob) | ECC costs ~4 GB of visible VRAM and buys nothing in bandwidth | new |
| [H2](#h2-the-b70-endpoint-always-reports-25-gts-x1) | The GPU function's link registers never describe the host link | **verified here**, contradicts our old reading |
| [H6](#h6-a-multi-card-process-keeps-a-host-ram-shadow-of-every-vram-buffer) | Every multi-card buffer also holds host pages of its own size | new |
| [H7](#h7-load-time-einval-is-a-temporary-external_host_ptr-mapping) | H2D copies ≥512 MiB make a temporary mapping the copy engine then faults on | new |
| [H8](#h8-a-cgroup-at-its-limit-swaps-regardless-of-swappiness) | A memory cap without zero swap turns into memory pressure | new |
| [H9](#h9-hardlockup_panic1-without-a-catcher-turns-a-stall-into-a-freeze) | Panic-on-lockup without capture is a silent freeze | new (panic already 0 here) |
| [H10](#h10-three-fault-classes-only-one-blames-the-card) | `bcs` at teardown is not card death; `ccs` from our own kernel is | have, partial |
| [H11](#h11-guc-firmware-version-drift) | GuC drift is a stability fault, not a speed fault | have, partial |
| [H12](#h12-amd-deep-idle-soft-lockup-storms) | AMD C6-class idle storms freeze a host with nothing running | new |
| [H14](#h14-evidence-vanishes-by-design-and-a-runtime-aspm-write-is-a-wedge) | journald and ext4 lose the last minutes; a runtime ASPM write wedges cards | new, plus our boot-time rule |
| [H16](#h16-peer-access-without-atomics-is-why-cross-device-polling-fails) | Peer `ACCESS` without `ACCESS\|ATOMICS` makes polling timeout-limited | have the conclusion, new the reason |
| [H17](#h17-a-bounded-spin-waits-faster-than-a-1-ms-sleep-poll) | Replacing a 1 ms sleep poll with a bounded spin gained 8.6% with identical arithmetic | new |
| [H19](#h19-stack-identity-is-first-class-and-the-jit-cache-can-be-poisoned) | Record the whole stack identity; a poisoned SYCL cache survives a reboot | have the switch, new the consequence |

## H1 ECC is a capacity trade, not a bandwidth knob

**Claim.** ECC on leaves roughly 28 GB of the card visible, ECC off the full 32 GB; his cards
report `32656 MiB` with `Current` and `Pending` both `disabled`. Intel specifies 608 GB/s either
way and publishes no matched ECC-on/off bandwidth result, so he refuses an ECC-off bandwidth claim.
Toggle is `xpu-smi config --device N --memoryecc 0` plus a reboot, and both fields must read
`disabled` before a benchmark calls itself an ECC-off run.

**Source.** [`docs/b70-ecc-and-vram.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/docs/b70-ecc-and-vram.md),
[`data/2026-08-30-b70-ecc-state.json`](https://github.com/steveseguin/b70-optimization-lab/blob/main/data/2026-08-30-b70-ecc-state.json).

**Our status.** `new` — no cookbook page discusses ECC. Read-only check on this host: both cards
report `visible_size: 32656MiB`, the number his record gives for ECC off, while `xpu-smi discovery`
returns `No device discovered` on kernel 7.0.0-28, so the ECC mode fields themselves are
**unverified here**.

**Verify.**
```bash
xpu-smi discovery && xpu-smi config --device 0 -j | grep -i ecc
sudo grep -E 'visible_size|visible_avail' /sys/kernel/debug/dri/0000:*/tile0/vram_mm
```

**Risk.** Turning ECC off removes error correction from the memory that holds every weight; reclaim
it only when the extra capacity changes what fits, and record the mode next to the run.

## H2 The B70 endpoint always reports 2.5 GT/s x1

**Claim.** The GPU function sits behind the card's own PCIe switch, and that internal hop reports
`2.5 GT/s x1` as both live and maximum state. The real host link is readable only at the switch
upstream port and the root port.

**Source.** [`docs/pcie-topology-and-llm-inference.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/docs/pcie-topology-and-llm-inference.md),
[`scripts/report-pcie-topology.sh`](https://github.com/steveseguin/b70-optimization-lab/blob/main/scripts/report-pcie-topology.sh),
[`notes/2026-06-10-qwen36-b70-host-link-audit.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/notes/2026-06-10-qwen36-b70-host-link-audit.md).

**Our status.** **verified here, and it contradicted this repository.** `scripts/doctor.py` read the
endpoint and could only ever print `2.5 GT/s x1` or nothing. Fixed to walk the ancestry and print
live and maximum state per hop. On this host (2026-10-29): GPU `0000:0c:00.0` and `0000:10:00.0`
report `2.5 GT/s x1` live and maximum at the function and at their card-internal bridge
(`0000:0b:01.0`, `0000:0f:01.0`), while root ports `0000:00:03.1` / `0000:00:03.2` and the switch
upstream ports `0000:0a:00.0` / `0000:0e:00.0` train at **16.0 GT/s x8** — the expected real link.
The BDFs recorded in two runnable examples were also stale (`0b`/`0f` now belong to bridges).

**Verify.**
```bash
python3 scripts/doctor.py                 # host link, then every hop with live/max state
sudo lspci -vv -s 0000:0a:00.0 | grep -E 'LnkSta|LnkCap'   # the upstream port, not the GPU
```

**Risk.** None for reading. The trap is reporting: a contributor who quotes the endpoint's `2.5 GT/s
x1` claims a Gen1-x1 host and invalidates their own reproduction coordinates.

## H6 A multi-card process keeps a host-RAM shadow of every VRAM buffer

**Claim.** In one multi-card process on his 4-card host (86–100 GB of VRAM in use), `GPUActive`
read 92.6 GiB of 115.6 GiB, `dma_buf` held 4,107 shared buffers totalling 98.9 GB, imports were
74.1 / 78.4 / 70.3 / 77.7 GB, page cache under 1 GiB, RSS 1–9 GiB, and the out-of-memory killer
ended the server. `NEOReadDebugKeys=1 EnableDeferBacking=0` dropped host RAM for a 12 GiB probe to
0.3 GiB with sharing and 603 GB/s unchanged and identical results; afterwards `GPUActive` was
3.6 GiB and 115/115 clips were byte-identical. Single-card processes do not pay this.

**Source.** [`docs/host-stability-and-fault-diagnosis.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/docs/host-stability-and-fault-diagnosis.md)
("Host RAM that vanishes"), [`experiments/ltx25-b70/notes/2026-10-04-host-ram-shadow-of-vram.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/ltx25-b70/notes/2026-10-04-host-ram-shadow-of-vram.md).

**Our status.** `new`. Both cookbook serving routes are one process per card, so we avoid it by
deployment shape, not by design — a future multi-card process on this 30 GB host inherits it.
`GPUActive` does not exist in `/proc/meminfo` on this host's kernel 7.0.0-28, and `dma_buf/bufinfo`
read `6 objects, 50 MB` at idle: the symptom only exists while the job runs.

**Verify.**
```bash
grep -E 'GPUActive|MemAvailable|SwapFree' /proc/meminfo     # absent on 7.0.0-28: use the two below
sudo tail -1 /sys/kernel/debug/dma_buf/bufinfo
sudo grep usage /sys/kernel/debug/dri/0000:*/tile0/gtt_mm    # imports, per card
```

**Risk.** `EnableDeferBacking=0` changes buffer placement, not arithmetic, but it is a runtime debug
key: pin the exact string and the compute-runtime version in the run identity, or a later bump
silently changes placement again.

## H7 Load-time `-EINVAL` is a temporary `EXTERNAL_HOST_PTR` mapping

**Claim.** Five saved load-time faults all show the copy engine reading the same 160 KiB at GPU
address `0x800400200000`; the runtime's allocation log names that address as an `EXTERNAL_HOST_PTR`
mapping that compute-runtime 26.27 makes for a host-to-card copy. A copy of 256 MiB or less makes no
such mapping, 512 MiB or more does, and `-EINVAL` means the mapping was gone when the copy engine
read it. Uploading in 128 MiB pieces makes no mapping, gives bit-identical weights and exact
answers, and costs no load time. Container swap aggravated the timing (5 faults in 4 days → 1 in 59
starts) and was not the cause: the 59th start faulted with swap frozen.

**Source.** [`docs/host-stability-and-fault-diagnosis.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/docs/host-stability-and-fault-diagnosis.md)
("copy-engine `-EINVAL` at load"), [`experiments/qwen38-27b-b70/overlays/b70-chunked-upload/`](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/overlays/b70-chunked-upload/b70_chunked_upload.py).

**Our status.** `new`. Our llama.cpp and vLLM routes upload 13–25 GB of weights with no chunking
guard and no allocation logging.

**Verify.**
```bash
journalctl -k -o cat | grep -B2 -A6 'Fault response: Unsuccessful -EINVAL'
# the kernel records this fault as one multi-line message; a line filter keeps only its empty first line
```

**Risk.** Chunked upload is a loader change: prove token identity against the unchunked load before
trusting it. Keeping the pieces at 128 MiB rather than 256 MiB is his margin, not a measured cliff.

## H8 A cgroup at its limit swaps regardless of swappiness

**Claim.** With `--memory 12g --memory-swap 16g`, a 27–29 GB load filled the container's page cache,
hit its own ceiling about two thousand times, and pushed 4.4 GB of working memory into the 4 GB swap
allowance with `vm.swappiness` at 1 and 7 GB of host memory free. Setting `--memory-swap` equal to
`--memory` (so `memory.swap.max` is 0) went from 5 faults in 4 days to 1 in 59 starts. A 4 GiB
`MemoryMax` "tripwire" around a 27 GB load made `systemd-oomd` kill the user session: a cap below
what a job really touches is not a safety net.

**Source.** [`docs/host-stability-and-fault-diagnosis.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/docs/host-stability-and-fault-diagnosis.md).

**Our status.** `new` — this cookbook has no cgroup or swap rule; the family recipes only warn about
OOM after it happens.

**Verify.**
```bash
cat /sys/fs/cgroup/system.slice/docker-<id>.scope/memory.swap.peak   # want 0
cat /sys/fs/cgroup/system.slice/docker-<id>.scope/memory.events      # want oom_kill 0
```

**Risk.** No swap at all removes slack for a host that also runs a desktop; the rule is a cap that
matches reality plus zero swap, not zero swap on an under-provisioned cap.

## H9 `hardlockup_panic=1` without a catcher turns a stall into a freeze

**Claim.** Five hard lockups, always the CPU servicing one card, in
`memcpy_fromio ← g2h_read ← xe_guc_ct_fast_path ← xe_guc_irq_handler`; on kernel 7.0.0-38 the frame
sits inside `g2h_read` with `xe_guc_pagefault_handler` on the stack, i.e. firmware page-fault
messages drained in interrupt context. With panic on, the machine froze silently; with it off the
host survived 14 s and 24 s stalls (`clocksource: Long readout interval ... 24242715672`) and the
workload's outputs stayed byte-exact. A multi-second gap in a run is this fault until the journal
says otherwise.

**Source.** [`docs/host-stability-and-fault-diagnosis.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/docs/host-stability-and-fault-diagnosis.md),
[`data/2026-10-03-xe-guc-hard-lockup/`](https://github.com/steveseguin/b70-optimization-lab/tree/main/data/2026-10-03-xe-guc-hard-lockup).

**Our status.** `new`. We ship [the wedge watchdog](../watchdog/README.md) for wedges
and nothing for lockups. Read-only check here: `/proc/sys/kernel/hardlockup_panic` is already `0`
and `watchdog_thresh` is `10`.

**Verify.**
```bash
cat /proc/sys/kernel/hardlockup_panic
journalctl -k -b 0 | grep -c 'hard LOCKUP'
```

**Risk.** Leaving it at 0 trades a silent freeze for a shorter freeze plus a lost trace when
`hardlockup_panic=1` would have panicked. Enable panic only once netconsole, pstore or kdump
actually catches panics on your board.

## H10 Three fault classes, only one blames the card

**Claim.** `-ENOENT` plus `Engine memory CAT error [18]: class=bcs` plus a coredump at process
teardown is a driver/card teardown signature, not card death: it is not followed closely by freezes
(22–100 minutes later, and two such boots ended cleanly). The same signature is open upstream in
[`intel/compute-runtime` #948](https://github.com/intel/compute-runtime/issues/948) across kernels,
drivers and firmware, and other dual-B70 owners report it. Hard-killing a busy job logs the same
lines by itself, and a direct card-to-card copy faulted both cards (stage through host memory
instead).

**Source.** [`docs/host-stability-and-fault-diagnosis.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/docs/host-stability-and-fault-diagnosis.md).

**Our status.** `have`, partial. [RELIABILITY-REPORT.md](RELIABILITY-REPORT.md) records our own
`class=ccs` CAT from an out-of-bounds gather and the `-ENOENT` fault strings, and points at the same
upstream issue; it does not separate `bcs`-at-teardown from `ccs`-from-our-kernel, so an agent can
read a teardown fault as a dead card.

**Verify.**
```bash
journalctl -k -o cat -b 0 | grep -E 'class=(bcs|ccs)|Fault response'
```

**Risk.** Reading a `bcs` teardown fault as a hardware failure retires a healthy card for a boot.
Reading a `ccs` CAT from our own kernel as the driver hides a real kernel bug.

## H11 GuC firmware version drift

**Claim.** On his 4-card host the Ubuntu package shipped GuC 70.44.1 while the driver requested
70.54.0 and 70.72.1 was available; four-card runs stalled at the oneCCL topology warning and the
kernel looped **360 GuC resets per minute across three cards while idle**, with `dmesg` naming the
recommended version. Installing the upstream `xe/bmg_guc_70.bin` and unbinding/reloading `xe`
cleared it. Speed was unchanged: the probe read 5.15 µs / 31.2 µs / 132 µs on 70.44.1 versus
5.18 / 30.2 / 134 µs on 70.72.1.

**Source.** [`learn/host-tuning.html`](https://github.com/steveseguin/b70-optimization-lab/blob/main/learn/host-tuning.html),
[`experiments/qwen38-flash-next-fp8-b70/tools/install-upstream-xe-firmware.sh`](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-flash-next-fp8-b70/tools/install-upstream-xe-firmware.sh).

**Our status.** `have`, partial. [RELIABILITY-REPORT.md](RELIABILITY-REPORT.md) already records the
B70's GuC ≥ 70.54 requirement and a `-ENXIO` bring-up failure on older firmware; it does not record
the idle-reset-count signature, and [BENCHMARK-FORMAT.md](BENCHMARK-FORMAT.md) asks for kernel and
driver in a run identity but not GuC. Read-only check here: installed and running **70.58.0**,
wanted 70.54.0 — no drift on this host.

**Verify.**
```bash
sudo cat /sys/kernel/debug/dri/0000:*/gt0/uc/guc_info | grep -E 'wanted|found release'
sudo dmesg | grep -i 'guc firmware'
```

**Risk.** Replacing a distribution firmware file is an out-of-band change that a kernel or
linux-firmware upgrade can undo silently; unrecorded, it makes two runs look identical when they are
not.

## H12 AMD deep-idle soft-lockup storms

**Claim.** Two boots died in `soft lockup` storms inside `smp_call_function_many_cond` (6,603 log
lines in one), one of them 2.5 minutes after boot **with nothing running**. Disabling the deepest
per-CPU idle state ended that signature and an idle boot then ran 5.6 days; the cleaner fix is BIOS
**Power Supply Idle Control = Typical Current Idle**.

**Source.** [`docs/host-stability-and-fault-diagnosis.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/docs/host-stability-and-fault-diagnosis.md)
("The idle-state freezes"), [`learn/host-stability.html`](https://github.com/steveseguin/b70-optimization-lab/blob/main/learn/host-stability.html).

**Our status.** `new`, and relevant: this repository's host is a **Ryzen 7 5700X3D (Zen 3)**, an AMD
part, so an idle freeze would be blamed on the GPU by default. Read-only check here: the kernel
exposes only `POLL`, `C1` and `C2` (state2 = `C2`, not disabled), so the exact "disable state2"
mitigation does not map one-to-one; the *signature* is what transfers.

**Verify.**
```bash
for f in /sys/devices/system/cpu/cpu0/cpuidle/state*/{name,disable}; do printf '%s=' "$f"; cat "$f"; done
journalctl -k -b -1 | grep -c 'soft lockup'
```

**Risk.** Disabling idle states costs power and heat at idle for a fault that may not exist on your
board. Change it on a signature, not pre-emptively.

## H14 Evidence vanishes by design, and a runtime ASPM write is a wedge

**Claim.** journald syncs every five minutes by default (`SyncIntervalSec=5m`) and ext4 writes data
back about 30 seconds late, so after a freeze the journal stops minutes early and recent files are
zero-length or NUL-filled **on a healthy disk** — his multi-day "storage fails first" false lead,
and a way to zero Git objects. Capture ranked: `netconsole` to a second host > `journalctl -kf` with
a `sync` per line > pstore/kdump (his was empty) > BMC (logged nothing). Separately, writing
`performance` to `/sys/module/pcie_aspm/parameters/policy` at runtime made three cards report
`runtime_status=error` on the 4-card host and made both Xe devices vanish with a panic on the
2-card host; no decode change was ever measured.

**Source.** [`docs/host-stability-and-fault-diagnosis.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/docs/host-stability-and-fault-diagnosis.md)
("Why evidence disappears"), [`learn/host-tuning.html`](https://github.com/steveseguin/b70-optimization-lab/blob/main/learn/host-tuning.html).

**Our status.** `new` for the capture rules; `have` the power-setting prohibition — our
[RELIABILITY-REPORT.md](RELIABILITY-REPORT.md) already records that `pcie_aspm.policy=powersupersave`
bricks a B70 at boot, which is the boot-time face of the same failure. Read-only check here: the
ASPM policy is `[default]`, and `/etc/systemd/journald.conf` leaves `SyncIntervalSec` at its
commented default, so both of his traps are still armed.

**Verify.**
```bash
grep -E '^#?SyncIntervalSec' /etc/systemd/journald.conf
cat /sys/module/pcie_aspm/parameters/policy
```

**Risk.** A netconsole receiver is another always-on host; without one, accept that the last minutes
before a freeze are gone and treat a short journal as normal, not as a storage failure.

## H16 Peer `ACCESS` without `ATOMICS` is why cross-device polling fails

**Claim.** All four cards are visible, every source→destination pair reports `ACCESS`, and **only
self-pairs report `ACCESS|ATOMICS`**. Remote fills and forked-IPC opens verified 16/16. A
cross-device mailbox without a CPU barrier read `[[5.0,50.0]]` instead of the expected
`[[6.5,65.0]]`; the "validated" single-kernel sequence polling ran 416.86 ms/iter against XCCL's
0.0618 ms/iter, and the atomic-counter variant failed too. Volatile loads plus system fences behave
as timeout-limited, and `sycl::atomic_ref` is not a correct cross-device sequence flag.

**Source.** [`notes/2026-05-20-minimax-qk-ipc-peer-polling-negative.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/notes/2026-05-20-minimax-qk-ipc-peer-polling-negative.md),
[`experiments/xpu_level_zero_peer_probe/README.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/xpu_level_zero_peer_probe/README.md).

**Our status.** `have` the conclusion, `new` the mechanism. Our lab ledger says "copy-only, no peer
atomics"; this cookbook never said why, which is exactly what lets the next agent retry
in-kernel polling. It also does not contradict [DNR-004](DO-NOT-REPEAT.md) — copy-path exchanges
still work, only atomic synchronisation across devices does not.

**Verify.**
```bash
sudo lspci -vvv -s 0000:0a:00.0 | grep -i 'ACSCtl'    # ACS redirect is the other half of "no P2P"
# per-pair ACCESS / ACCESS|ATOMICS needs a Level Zero query (zeDeviceGetP2PProperties);
# the probe lives in his repo and has no port here yet.
```

**Risk.** The tempting workaround — volatile loads plus fences — passes a smoke test and then
timeouts under load. Do not build a transport on it.

## H17 A bounded spin waits faster than a 1 ms sleep poll

**Claim.** With identical arithmetic, kernels and selection, changing only the native `wait()`
policy from a 1 ms `sleep_for` poll to a per-call spin budget (periodic yields, deadline kept, async
errors checked each iteration) moved his one-card engine from 31.966820 to 34.577 / 34.568 tok/s,
+8.58–8.71% against a +0.5% gate, 60/60 output-exact. At that point the engine issued **2,307
launches per token at 5.4–11.7 µs each** with the GPU waiting on the host.

**Source.** [`notes/2026-10-19-own-runtime-packet58as-result.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/notes/2026-10-19-own-runtime-packet58as-result.md),
[`experiments/own-xpu-runtime/SCOREBOARD.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/own-xpu-runtime/SCOREBOARD.md).

**Our status.** `new`. Every cookbook route is a stock llama.cpp or vLLM server; none of them exposes
a wait policy, so this transfers to a custom engine or a host-driver loop, not to a flag.

**Verify.**
```bash
pidstat -w -p <server-pid> 1 3      # voluntary context switches per second per waiting thread
```
A wait that sleeps a full scheduler quantum on an incomplete event shows up as roughly a thousand
voluntary switches per second per waiting thread.

**Risk.** A spin burns CPU the desktop also needs, and an unbounded spin can outlive a deadline:
keep the deadline and the error check, and gate the arm on exact output, not on the rate.

## H19 Stack identity is first-class and the JIT cache can be poisoned

**Claim.** His recorded identities include compute-runtime 26.18.38308.1 with L0 loader 1.28.2 on
kernel 7.0.0-39 (4-card) and compute-runtime 26.27 (2-card). Kernel 7.0.0-34 was security-only for
`xe`/DRM; -38 carried an exec-queue teardown deadlock fix and two page-table bind fixes; -39 stopped
handing out compression-owned VRAM. `SYCL_CACHE_PERSISTENT=1` "poisons cache, next boot SEGV on
B70". `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` is needed above 4 GB of KV.
`GGML_SYCL_DISABLE_OPT=1` is mandatory for MoE (SEGV), not for dense. Two SYCL servers on one card
measured a 10× slowdown.

**Source.** [`notes/2026-07-12-b70-qwen27-prior-art-research.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/notes/2026-07-12-b70-qwen27-prior-art-research.md),
[`AGENTS.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/AGENTS.md).

**Our status.** `have` the switch, `new` the consequence. [FEATURE-FLAGS.md](FEATURE-FLAGS.md)
lists `SYCL_CACHE_PERSISTENT=0` as "disables the persistent SYCL kernel cache" for one family and
never says why it is not optional; nothing here records the MoE-versus-dense split for
`GGML_SYCL_DISABLE_OPT` (our host notes already say not to set it by default, for a different
reason: a measured decode regression). This host's identity, read-only: kernel 7.0.0-28, `xe`
driver, GuC 70.58.0.

**Verify.**
```bash
env | grep -E 'SYCL_CACHE|NEO_CACHE|XDG_CACHE|UR_L0' ; uname -r ; modinfo xe | head -2
```

**Risk.** A poisoned cache survives a reboot and looks like a driver regression; the cheap defence
is a per-run cache directory, which also makes two runs comparable.

## Not tested anywhere

Nothing on this page has been reproduced on a host of ours except H2. H6, H7, H9, H12 and H17 are
single-lab findings on two hosts; treat their causes as hypotheses with a named owner, and their
numbers as his, not ours. No item here changes a rate in [data/benchmarks.v1.json](../data/benchmarks.v1.json).

## Process ideas to evaluate

Governance, not recipes — nothing here needs a GPU to decide, and nothing here is adopted yet. Each
idea would change how this repository publishes, so each needs its own arm.

| Ref | Idea | What it would change here | Effort |
|---|---|---|---|
| P2 | **Class-balanced headline metric** — median within each input class, then median across class medians, over a fixed 12-prompt suite with at least five declared classes | Our headline is a median over all valid samples, so a class with more prompts can dominate it. His measured effect: Gemma 4 26B legacy `124.977141` / all-prompt `123.727369` / class-balanced `122.160357`; Qwen 27B Q4_K_M `27.824790` → `27.825726`. Source: [`docs/benchmark-integrity-audit-20260827.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/docs/benchmark-integrity-audit-20260827.md) | M |
| P3 | **N events span N−1 intervals** — time decode over the 99 inter-token intervals between timestamps 1 and 100, and record event count, interval count, numerator and endpoints | His helper divided 100 events by a 99-interval span, so an approved row at `102.971435596` is `101.941721240` under conventional accounting. Ratios survive; absolutes do not. Our server-side `timings` observer is not exposed today, but any client-interval metric needs both rules. Source: [`docs/research-workflow-playbook.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/docs/research-workflow-playbook.md) "Metric Accounting Gate" | S |
| P5 | **Certification ladder with an honest empty label** — `starter-guide` / `candidate-portable-repro` / `lab-replay` / `record-capsule` / `research-status` / `archived`, only a clean-host replay earning "Install guide", with a mandatory dependency-closure table and **no guide currently certified** | Our [START.md](../START.md) calls every path a quick start, so a reader cannot tell a replayed route from a research one. Source: [`docs/reproduction-guide-certification.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/docs/reproduction-guide-certification.md) | S/M |
| P6 | **Promotion attestation** — one JSON that hash-binds the exact performance file to model, runtime and optimization identity plus six quality/determinism gates, and fails closed on unbound booleans; no oracle means `featured_metric: null` | Our results are evidence directories with no cross-file binding, so a stale number and a live quality file can drift apart silently. Source: [`docs/promotion-attestation.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/docs/promotion-attestation.md) | M |
| P10 | **Claims registry with a lifecycle** — one JSON per claim, `submitted → queued → reproducing → confirmed / confirmed-adjusted / refuted / stale`, an append-only `history[]`, a validator that requires the evidence path to exist, and confirmation only on identical model, quant, runtime, patch set, topology, metric and gate | We keep submission payloads and ledger rows, but a claim has no state, so "adjusted" is invisible and refuted claims are not retained as claims. Source: [`claims/README.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/claims/README.md) | M |
| A7 | **Incoming-diff guard before any rebase or merge** — read `git diff --stat` for unexpected deletions, never `git add -A`; one of his commits from another host zeroed 302 files | Our trees are shared between two machines with no such check, and the failure mode is silent until a file is needed. Source: [`AGENTS.md`](https://github.com/steveseguin/b70-optimization-lab/blob/main/AGENTS.md) | S |
