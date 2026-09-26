# Quick Start Guide: Intel Arc Pro B60/B70 Inference

> **30 minutes from zero to serving your first model on Intel Arc Pro B60/B70.**  
> For the automated agent control plane and architectural layers, see [`AGENTS.md`](AGENTS.md) and [`data/system-map.v1.json`](data/system-map.v1.json).

---

## 1. Choose by task, not by a peak number

| Setup | Start here | Limitation |
|---|---|---|
| One B70, GPTQ serving and agents | [Qwen3.8-27B](docs/qwen38-27b/QWEN38-VLLM-XPU.md) | The command below uses the BF16 drafter, not the separate [draft-INT4 speed experiment](docs/qwen38-27b/DRAFT-INT4-S-M1.md). Verify answer quality on your own tasks. |
| One B70, GGUF or vision | [Muse-Glimmer llama.cpp](docs/muse-glimmer/MUSE-GLIMMER-B70.md) | Its engine decode metric cannot be ranked against vLLM client timing without a matched test. |
| MoE or long context | [Qwen3.6-35B-A3B](docs/qwen36-35a3/QWEN36-MOE-VLLM-XPU.md) | The historical MTP4 peak did not reproduce on a [later stack](docs/qwen36-35a3/MOE-INT4-THREE-ENGINE-CEILING-20260917.md). |
| Two B70s | [Independent servers](docs/DUAL-B70-TP2.md) | TP2 is research, not the default single-request speed route. |

See [what worked, what lost, and the next test](docs/WHAT-WORKED.md) before tuning. Exact image, patch, model, cache, and timing definitions matter.

---

## 2. Single-B70 fast path: Qwen3.8-27B (Linux)

This pinned BF16-drafter route serves an OpenAI-compatible API with native MTP. Its measured rates depend on prompt, sampling, cache, and workload; it does **not** run the optional draft-INT4 overlay or guarantee a headline speed. Tool calling needs the parser flags in the command below. A passing four-canary smoke is not an exact-answer quality assessment.

### Prerequisites Check (30 seconds)
Verify that your card is detected and your user belongs to the `render` group:
```bash
python3 scripts/doctor.py
```
If you don't have `doctor.py` dependencies yet, just check:
```bash
ls -l /dev/dri/render*
groups | grep -E 'render|video'
```

---

### Step 1: Pull the pinned vLLM XPU container
```bash
export IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
docker pull "$IMAGE"
```

### Step 2: Download the pinned model weights
```bash
export MODEL_DIR="$HOME/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
huggingface-cli download SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  --revision 9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e \
  --local-dir "$MODEL_DIR"
```

### Step 3: Launch the inference server
```bash
RENDER_GID="$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')"
docker rm -f b70-qwen38 >/dev/null 2>&1 || true

docker run -d --name b70-qwen38 -p 8000:8000 \
  --device /dev/dri --group-add "$RENDER_GID" \
  -v /dev/dri:/dev/dri:ro \
  -v "$MODEL_DIR:/model:ro" \
  -v "$PWD/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
  -v "$PWD/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" -lc \
  'set -e; python /patch_mtp.py; python /patch_boundary.py; exec vllm serve /model \
   --quantization gptq --dtype float16 --max-model-len 100000 \
   --gpu-memory-utilization 0.88 --kv-cache-dtype fp8 --port 8000 \
   --max-num-seqs 1 --max-num-batched-tokens 8192 \
   --no-enable-prefix-caching --served-model-name qwen38 \
   --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_xml \
   --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}"'
```

---

## 3. Verify Health and Run First Completion

### Check server health
```bash
curl -f http://127.0.0.1:8000/health
```

### Run a sample completion with token timing
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38",
    "messages": [{"role": "user", "content": "Explain why Intel Arc Pro B70 is great for inference in 2 bullet points."}],
    "max_tokens": 128,
    "temperature": 0.0
  }' | python3 -m json.tool
```

### Run the four basic canaries (arithmetic, JSON, code syntax, prose)
```bash
python3 scripts/canary.py --port 8000 --model qwen38
```

---

## 4. Immediate Triage & Failure Cards

| Symptom | Cause | Immediate Fix |
|---|---|---|
| `curl /health` hangs or never responds | **Xe driver ring wedge** | The Intel Xe driver wedged. Check `dmesg \| grep -iE 'xe\|wedged'`. Recover with `python3 scripts/doctor.py --recover` or reboot the host. See [RELIABILITY-REPORT.md](docs/RELIABILITY-REPORT.md). |
| `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` | **VRAM KV overflow** | Lower `--max-model-len` from `100000` to `65536` or reduce `--gpu-memory-utilization` to `0.85`. |
| `UR_RESULT_ERROR_OUT_OF_RESOURCES` during compile | **Parallel inductor compile threads** | Level Zero context exhausted during JIT. Add `-e TORCHINDUCTOR_COMPILE_THREADS=1` to the `docker run` line. |
| `zeMemOpenIpcHandle` failure on dual-card | **Missing oneCCL thresholds** | Host does not support P2P direct access. Export the 4 simple-threshold variables in [DUAL-B70-TP2.md](docs/DUAL-B70-TP2.md). |
| Gibberish output or repeating tokens | **APC + MTP conflict** | Prefix caching (`--enable-prefix-caching`) combined with MTP produces silent corruption on this kernel build. Keep `--no-enable-prefix-caching`. |

---

## 5. Next Steps

- **Connect your coding agents:** See [CONNECTING-CLIENTS.md](docs/CONNECTING-CLIENTS.md) for Pi, Hermes, and Open WebUI configs.
- **Tune power sweet spots:** See [POWER-SWEET-SPOTS.md](docs/POWER-SWEET-SPOTS.md) for 150 W vs 230 W trade-offs.
- **Full hardware setup:** See [FULL-SETUP-COMMANDS.md](docs/FULL-SETUP-COMMANDS.md) for driver, GuC firmware, and kernel setup.
- **Report or reproduce numbers:** See [CONFIRM-A-RECIPE.md](docs/CONFIRM-A-RECIPE.md) to confirm numbers with a single command.
