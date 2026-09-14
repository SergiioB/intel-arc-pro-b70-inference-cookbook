# Suggested title

WSLC Intel XPU inference is 2.4-2.8x slower than Docker Desktop using the same vLLM image and Arc Pro B70

# Description

GPU inference inside a WSLC container is substantially slower and less stable than inside a Docker Desktop Linux container on the same Windows host, using the same image, model, Intel GPU, vLLM/PyTorch versions, model mount, API benchmark, and XPU settings.

The Docker container also accesses the GPU through `/dev/dxg` and the Windows WSL driver libraries. This makes Docker Desktop a useful control: the model and Intel XPU software stack can achieve approximately 70 tokens/second on this machine, while WSLC falls to approximately 26 tokens/second.

# Environment

- Windows: Windows 11 Pro, version 10.0.26200, build 26200
- WSL: 2.9.4.0
- WSLC: 2.9.4.0
- WSL kernel: 6.18.35.2-microsoft-standard-WSL2
- WSLg: 1.0.79
- Intel GPU: Intel Arc Pro B70, PCI device reported by Level Zero as `0xe223`, 32 GiB VRAM
- Intel graphics driver: 32.0.101.8805, dated 2026-07-07
- Docker client/engine control: 29.7.2
- Image base: pinned `vllm/vllm-openai-xpu`
- vLLM: 0.27.2rc1.dev77+gac7509e2b
- vLLM XPU kernels: 0.1.12.3
- PyTorch: 2.13.0+xpu
- Model: Qwen3.8-27B GPTQ INT4, text-only

# Common inference configuration

- One Intel XPU / one active sequence
- MTP speculative decoding with 4 draft tokens
- MTP acceptance: 100% in all cited measurements
- XPU graphs enabled
- Maximum model length: 100,000 tokens
- FP8 KV cache
- Explicit KV cache allocation: 4.25 GiB
- Reported cache capacity: 102,631 tokens (1.03x concurrency at 100K)
- `max_num_batched_tokens=8192`
- Identical OpenAI-compatible completion requests and exact-token benchmark script

# Results

| Scenario | WSLC | Docker Desktop control | Difference |
|---|---:|---:|---:|
| 512-token prompt, decode after initial run/idle | 25.76 tok/s | 69.76 tok/s after 60 s idle | Docker 2.71x faster |
| 512-token prompt, best observed initial WSLC run | 48.18 tok/s | 71.84 tok/s median | Docker still 1.49x faster |
| 8,192-token prompt, decode | 26.91 tok/s | 64.26 tok/s median | Docker 2.39x faster |
| 8,192-token prompt, time to first token | 348.82 s | 4.42-4.51 s | Docker approximately 78x faster |

Docker consistency measurements on the restored recommended configuration:

- Short batch 1, five runs: median 71.84 tok/s; range 71.63-72.94; TTFT 0.356-0.359 s.
- After 60 seconds idle, five runs: median 69.76 tok/s; range 69.53-71.01; TTFT 0.361-0.365 s.
- Two sustained 1,024-output-token runs: 68.45 and 67.08 tok/s.
- Three 8K-prompt runs: 64.42, 64.26 and 64.08 tok/s; TTFT 4.42-4.51 s.
- MTP acceptance remained 100% throughout.

The Docker short-prompt median changed by only 2.9% after the idle interval. WSLC previously dropped from 48.18 to 25.76 tok/s under the comparable repeat/cooldown scenario.

# Reproduction outline

1. Build the supplied Dockerfile twice from the same directory: once as a WSLC image and once in Docker Desktop. Keep the pinned base-image digest and all copied patches identical.
2. Bind-mount the same Windows model directory at `/model`.
3. Expose the Intel B70 to both containers. Docker Desktop requires `/dev/dxg` plus read-only mounts of `/usr/lib/wsl/lib` and `/usr/lib/wsl/drivers`; the restricted (non-privileged) Docker XPU allocation/compute probe succeeds.
4. Start vLLM with the common configuration above and served model name `qwen38`.
5. Wait for `GET /v1/models` to return successfully.
6. Send deterministic, streaming `POST /v1/completions` requests with `temperature=0`, `ignore_eos=true`, and exact 512- or 8,192-token prompts.
7. Measure TTFT and post-first-token decode time for a fixed token count. Read vLLM metrics to confirm MTP draft/accepted-token counters.
8. Repeat after a 60-second idle interval without restarting or reconfiguring the server.

The local benchmark used for these figures is `Test-CookbookDecode.ps1` and can be attached with the container Dockerfile/start script.

# Expected behavior

WSLC and Docker Desktop do not need to be exactly identical, but GPU-bound decode throughput and long-prompt TTFT should be broadly comparable when the same image, Windows driver, `/dev/dxg` device, model, runtime versions, and inference arguments are used.

# Actual behavior

WSLC is 2.4-2.8x slower in steady-state decode, shows a large post-repeat/idle degradation, and takes approximately 349 seconds to first token for the tested 8K prompt. Docker Desktop produces 64-72 tok/s consistently and handles the same 8K prompt in approximately 4.4 seconds TTFT.

# Questions for investigation

- Does WSLC configure `/dev/dxg`, Level Zero, shared/dedicated GPU memory, command queues, or GPU scheduling differently from Docker Desktop's WSL2 backend?
- Is WSLC causing GPU allocations to spill into shared system memory or use a slower memory-placement path despite sufficient B70 VRAM?
- Are XPU command buffers, graph replay, pinned host memory, 9P reads, or synchronization handled differently by the WSLC runtime?
- Can Microsoft provide a GPU scheduling/ETW trace collection procedure suitable for comparing WSLC with Docker Desktop?
- Are there WSLC diagnostics that expose GPU memory residency, page migration, queue utilization, or `/dev/dxg` scheduling decisions?

# Additional observation

Allowing vLLM to automatically allocate cache at `gpu_memory_utilization=0.80` created a 22.07 GiB KV cache (capacity 536,842 tokens) and reduced Docker decode to 25.73 tok/s. Returning to the explicit 4.25 GiB cache restored approximately 70 tok/s. The WSLC/Docker comparison above uses the explicit cache configuration to avoid this independent memory-pressure effect.

# Suggested attachments

- `Test-CookbookDecode.ps1`
- Dockerfile and container startup script
- Full WSLC and Docker vLLM startup logs
- `wsl --version`, `wslc --version`, Windows build and Intel driver output
- WSLC diagnostic logs collected using the method requested by Microsoft maintainers
