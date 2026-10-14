---
title: "Incident and Cleanup Procedure"
type: operation
date: 2026-08-27
status: living
---

---
doc_id: DOC-OPS-INCIDENT-CLEANUP
doc_type: operational_method
title: Incident and Cleanup Procedure
lifecycle: active
created: 2026-08-19
updated: 2026-08-19
last_verified: 2026-08-19
canonical_for:
  - inference-incident-response
  - campaign-cleanup
---

# Incident and Cleanup Procedure

## Stop conditions

Abort a load or campaign when any of these occurs:

- a second engine or unrelated GPU workload appears;
- post-load free VRAM falls below the declared gate;
- the server reports a device assert, OOM, corrupted timing, or fatal engine-core error;
- progress stops beyond the predeclared per-cell timeout;
- exact prompt/output tokens, cache counters, or timing sources cannot be verified;
- temperature or telemetry coverage is lost for a comparison that requires it;
- storage headroom crosses the conversion/download reserve;
- an output path resolves through a source symlink.

## Controlled stop

1. Stop the exact campaign harness.
2. Stop the exact server/container without broad deletion of unrelated containers.
3. Restore each card's pre-campaign observed `power1_cap` through the dynamically discovered `xe` hwmon node (the 150 W default is retired 2026-09-14).
4. Re-run the process, container, VRAM, and mount checks.
5. Preserve logs and partial artifacts; mark them failed, aborted, or incomplete rather than deleting them.
6. Record whether cleanup completed and whether production service restoration was authorized.

For user systemd units, initialize the login-session bus when needed:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=${XDG_RUNTIME_DIR}/bus
```

## Evidence preservation

An incident record should retain:

- UTC and monotonic timestamps;
- triggering condition;
- process/container identity;
- server and harness logs;
- VRAM and storage state;
- configured cap and verified hwmon path;
- cleanup commands/outcome;
- affected observations;
- whether raw measurements remain usable;
- replacement campaign, if any.

Do not erase failed or contaminated observations. They are negative evidence and may explain later decisions.

## Common incident classes

### Silent prefix-cache contamination

Preserve the cell, mark it contaminated, record cache-hit deltas, and rerun with entropy-first unique prompts. Do not fold contaminated samples into a replacement median.

### VRAM overflow or corrupted timings

Treat sub-gate reserve, garbage throughput, stalled device workers, or system instability as an invalid capacity/performance state. A server process that remains alive does not make the result valid.

### Service respawn

Stopping a container is insufficient when a user unit has restart enabled. Stop or temporarily constrain the unit first, verify no replacement process appears, and record the isolation state.

### Failed container startup

Capture `docker logs` before removing a failed container. Cleanup traps should not destroy the only startup evidence.

### Partial storage mount

Verify `findmnt`, LVM state, and the known partial-VG trap before declaring artifacts deleted.

## Completion statement

A campaign is not complete until:

- the exact engine/container is stopped unless continued service was explicitly requested;
- each card's power cap is restored to its pre-campaign observed value;
- process and container isolation is rechecked;
- cleanup state is retained under the campaign root;
- failures and exclusions are reflected in status metadata.
