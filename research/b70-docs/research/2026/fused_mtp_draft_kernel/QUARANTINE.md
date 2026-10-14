---
title: "Invalid MTP prototype quarantine"
type: research
date: 2026-09-05
status: final
---

# Invalid MTP prototype quarantine

These files are not an MTP implementation. The vLLM C++ draft writes token zero, omits all advertised math, and the Python intercept omits embeddings/hidden-state fusion, attention with positions/KV state, decoder MLP/residuals, and the actual sampler contract. Norm + GEMV + argmax cannot replace native MTP. No speed or correctness claim is valid.

Original bytes are retained under `preserved-originals/` with `.disabled` suffix and `original-sha256.json`. Python raises before importing torch; C++ refuses compilation; shell exits 78; patch files are deliberately non-applicable. Never bypass these guards. A replacement must start from actual pinned vLLM source and measured profiling, not these sketches.

`python3 test_quarantine.py` validates the guards. The authoritative copy is on b70-host in B70-DOCS; the Rock copy is receive-only and was explicitly quarantined because that is where the erroneous files existed.
