---
doc_id: DOC-COMPAT-LVM-AND-STORAGE
old_doc_id: DOC-LEGACY-LVM-AND-STORAGE-CURRENT-GUIDANCE
doc_type: compatibility_view
title: Storage Documentation Has Moved
lifecycle: superseded
updated: 2026-08-19
superseded_by:
  - DOC-OPS-STORAGE-VRAM-SAFETY
  - DOC-OPS-CURRENT-HOST-STATE
---

# Storage Documentation Has Moved

This path is retained for compatibility. It previously mixed stable storage safety, incident history, and volatile disk/model inventory, which caused stale operational guidance.

Use:

- Storage and VRAM Safety (`operations/storage-and-vram-safety.md`) for stable operating rules.
- Current B70 Host State (`operations/current-host-state.md`) for live, dated mount and artifact observations.
- Archive copy (`../archive/docs/pre-migration-20260819/lvm-and-storage.md`) for the complete pre-migration historical document.

Do not use historical free-space or model-presence values from the archive to authorize a download, conversion, or model load.
