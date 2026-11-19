# Draft: intel/compute-runtime issue (NOT filed; the operator decides)

**Recommendation: do not file.** The hang reproduces on the installed 26.05.37020.3 and does **not** reproduce on the
latest release 26.40.40196.8 (side-by-side, same reproducer, same host, 2026-11-19). The project's Linux issue template
requires "I am using the latest GPU driver version" in its pre-submission checklist, so this would not be a valid report.
The useful action is local: move the host to 26.40 (operator decision; nothing was installed) or keep the workaround key.

Policy check (read 2026-11-19 08:20 CET on `master`): `CONTRIBUTING.md`, `README.md`,
`.github/ISSUE_TEMPLATE/1-linux-gpu-driver-issue.yml`, `.github/ISSUE_TEMPLATE/config.yml` and
`documentation/ISSUE_SUBMISSION_GUIDE.md` contain no statement about AI-assisted or machine-generated reports. Relevant quotes:
- Issue template checklist: "I am using the latest GPU driver version ([releases](https://github.com/intel/compute-runtime/releases))".
- Submission guide: "Use the latest driver version before reporting a bug" and "Blank issues are not accepted."
- `README.md`: "Debug parameters, environmental variables, and internal data structures are considered as internal
  implementation detail and may be changed or removed at any time." (applies to the workaround key below)

Copies: `$WORK/tpdiag/src-ref/cr-policy/`.

---

## Title (if it were filed)

Hang on two GPUs: reused, host-reset timestamp events on in-order immediate lists with cross-device event waits never complete (implicit counter-based conversion)

## Versions and hardware

| Item | Value |
|---|---|
| GPU | 2x Intel Arc Pro B70 32 GB (BMG-G31), PCI 0000:0b:00.0 and 0000:0f:00.0, PCIe 4.0 x8 each, one context |
| Driver (fails) | intel-opencl-icd / libze-intel-gpu1 26.05.37020.3-1 (`driverVersion` 17010844), Level Zero loader 1.28.2 |
| Driver (passes) | 26.40.40196.8 (`driverVersion` 17014020) + IGC 2.42.6, loaded with `LD_LIBRARY_PATH` |
| OS | Ubuntu 26.04, kernel 7.0.0-28-generic, xe KMD; `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1` |
| CPU | AMD Ryzen 7 5700X3D |

## Minimal steps

Reproducer: `tp/tpdiag/repro/repro.cpp` + `repro.cl` (about 170 lines, Level Zero only), run script `repro/run.sh`.

1. One context, two devices; one in-order immediate compute list per device (`ZE_COMMAND_QUEUE_FLAG_IN_ORDER`, async).
2. Per device an event pool with `HOST_VISIBLE | KERNEL_TIMESTAMP` (that device only); every kernel signals one of its events.
3. A shared host-visible pool over both devices for "ready" events (signal/wait scope 1).
4. Batch of 128 steps: on each device a kernel, then `zeCommandListAppendSignalEvent(ready[s][d])`; then on each device a
   kernel that waits on both `ready[s][*]`. Batch end: device 1 signals, device 0 waits on it, then a blocking
   `zeCommandListAppendMemoryCopy` into host stack memory, `zeCommandListHostSynchronize` on both lists, read every
   timestamp, `zeEventHostReset` on every event.
5. Repeat the batch with the same events.

## Expected vs actual

Expected: every batch completes. Actual (26.05.37020.3, two devices): batch 0 completes, batch 1 never completes
(3 of 3 runs; the watchdog reports no progress for 30 s at step 127, i.e. in the blocking copy at the batch end).

| Variant | 26.05.37020.3 | 26.40.40196.8 |
|---|---|---|
| two devices, default | **hang** (3/3) | pass |
| two devices, `NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0` | pass | pass |
| two devices, `EnableDirectSubmission=0` | **hang** | not run |
| one device (either card) | pass | not run |

Raw: `$WORK/tpdiag/raw/repro/*.log`.

## Stack (application: TensorFold dev runner, same pattern)

Main thread spins in `libze_intel_gpu.so.1` under the blocking host copy at the end of the forward
(`forward.zig:583`, `zeCommandListAppendMemoryCopy` to stack memory), frames `sched_yield` ← 6 driver frames ← app;
one driver thread waits in `pthread_cond_clockwait`. Kernel stacks: `futex_do_wait`. Full dump:
`$WORK/tpdiag/raw/prof-hang/asbuilt.gdb`.

## Workaround

`NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0` (no measurable speed change in TensorFold:
151.3/151.4 vs 151.5/151.7 tok/s, 94.5 vs 94.6 at 32K), or runtime 26.40.40196.8.
