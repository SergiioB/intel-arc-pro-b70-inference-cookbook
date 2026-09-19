# Confirm a Recipe (Lightweight Contributor Guide)

> **Did you run one of the cookbook recipes on your Intel Arc Pro B60/B70?**  
> We want your evidence! You don't need to write a new family slice, patch, or catalog renderer to confirm an existing recipe.

---

## The 3-Step Confirmation Process

### Step 1: Capture your environment with `doctor.py`
Run the environment doctor and save the JSON diagnostic:
```bash
python3 scripts/doctor.py --json > doctor.json
```
This logs your Linux kernel, driver version, GPU count, and PCIe link width/speed.

### Step 2: Run the model and execute the canary probe
Once your chosen recipe server is healthy on port 8000:
```bash
python3 scripts/canary.py --port 8000 --model <served-model-name> --json > canary.json
```
This runs 4 golden canaries (code syntax, JSON schema, arithmetic determinism, and prose) and logs your tokens/second and TTFT.

### Step 3: Open a "Recipe Confirmation" GitHub Issue
Open an issue on GitHub using the **Recipe Confirmation** template, or paste:
1. **Recipe name / ID:** (e.g. `Qwen3.8-27B GPTQ-INT4 single-card` or `Muse-Glimmer llama.cpp`)
2. **Hardware config:** (e.g. `1× Intel Arc Pro B70 32 GB, PCIe 4.0 x16, Power cap 230 W`)
3. **`doctor.json`** snippet
4. **`canary.json`** snippet
5. **Decoded tok/s observed:** (and sample prompt length)

---

## What happens next?

Once verified:
1. Your report will be linked directly as community reproduction evidence in `docs/BENCHMARK-CATALOG.md`.
2. The benchmark entry's trust label will advance from `official-lab` to `community-reviewed`.
3. If you ran on a different PCIe generation (e.g. PCIe 4.0 vs PCIe 5.0) or different power cap (150 W vs 230 W), your result will be tracked as an official comparison coordinate.

Thank you for helping keep Intel Arc Pro B60/B70 inference numbers transparent, portable, and reproducible!
