# Launcher-equivalent launch for recipe `qwen38-27b-sc5` — derivation

Sources: `b70launcher/builders/exl3.py` (EXL3 docker tokens),
`builders/common.py` (device/label helpers), `b70launcher/exl3_stack.py` (daemon only),
`recipes.json` model `qwen38-27b-sc5` (line ~458).

## What the launcher builds (builder exl3.py:56-80)

```
docker -H unix:///run/b70-exl3-docker.sock run --rm --name <cname> \
  --label b70.launcher.rid=<rid> --label b70.launcher.port=<port>      # common.py:148-152
  --device <render> --device <card>                                   # common.py:83-85 (selected cards only)
  -v <filtered-by-path>:/dev/dri/by-path:ro                           # common.py:56-73,87
  --group-add <gid per node>                                          # common.py:88-96
  -v <model_dir>:/mnt/exl3model:ro                                    # exl3.py:58
  -p localhost:<host_port>:8100                                       # exl3.py:59
  -e UR_L0_USE_COPY_ENGINE=0 -e SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0  # exl3.py:64-65
  ghcr.io/sergiiob/exl3xpu-mixedk:20261014@sha256:933eda54...         # recipes.json image pin
  models/qwen3.8-27b-exl3-4.00bpw                                     # recipes.json serve_config
  --gpu 0 --model-path /mnt/exl3model                                 # exl3.py:68-69
  --set vllm.gpu_memory_utilization=0.94                              # ctx 196608 > ctx_safe -> gmu "0.94" (exl3.py:49-52)
  --set vllm.max_model_len=196608                                     # recipes.json ctx
  --set vllm.enable_prefix_caching=true                               # exl3.py:72
  --set served_model_name="Qwen3.8-27B SC5"                           # exl3.py:74 (json.dumps of model name)
  --set vllm.reasoning_parser=qwen3                                   # exl3.py:77
  --set vllm.enable_auto_tool_choice=true                             # exl3.py:78
  --set vllm.tool_call_parser=qwen3_coder                             # exl3.py:79
```

- No `--memory` cap in the launcher (added here per campaign rule: `--memory=14g --memory-swap=14g`).
- No `EXL3_INT8_PREFILL` env/--set anywhere in the launcher — the value comes from the image's
  `models/qwen3.8-27b-exl3-4.00bpw/model.yaml` (`EXL3_INT8_PREFILL: "1"`). **Arm A (INT8=1) therefore
  equals the as-served default; arm B (INT8=0) is the priced alternative.**
- No `EXL3_DRAFT_VOCAB` override in the launcher → model.yaml default
  `EXL3_DRAFT_VOCAB={model_dir}/draft_vocab.json` stands (the file is absent in the SC5 artifact
  dir; the launcher tolerates this since it ships the same default).
- MTP3 stays ON (`model.yaml` `speculative_config: {method: qwen3_5_mtp, num_speculative_tokens: 3}`),
  `max_num_batched_tokens=4096`, `kv_cache_dtype=fp8`, `mamba_cache_mode=align`, graphs
  FULL_DECODE_ONLY — all from model.yaml, untouched.
- Image entrypoint is `python3 scripts/serve.py` (`docker inspect`), so recipe tokens start with the
  serve-config positional directly.

## Deviations applied for this run (plumbing/diagnostic only)

- Card: GPU1 (`0000:0f:00.0`) — `--device /dev/dri/card1 --device /dev/dri/renderD129`,
  `--group-add 44 --group-add 990`, by-path dir filtered to that card's links
  (`dri-by-path/card1-renderD129` under the work dir; mirrors `common.by_path_dir`).
  Inside the container the card is device 0 → `--gpu 0` (same as launcher semantics).
- Port: `--network=host` + serve.py `--port 8031` instead of bridge `-p localhost:P:8100`.
- `--set vllm.enable_prefix_caching=false` — diagnostic deviation per PLAN (every request cold).
- `--set env.EXL3_INT8_PREFILL=<1|0>` — the arm variable.
- Container name `askl-ttft-<arm>-<block>`, label `b70.owner=AsServedKL` (contract label instead of
  the launcher's `b70.launcher.*` labels — those exist for launcher adoption, unused here).
- `--rm` dropped (container kept until log capture, then `rm -f` by the runner).

Derived run command (arm A shown; identical for B except `EXL3_INT8_PREFILL=0`):

```
sudo -n docker -H unix:///run/b70-exl3-docker.sock run -d --name askl-ttft-A-b1 \
  --label b70.owner=AsServedKL \
  --device /dev/dri/card1 --device /dev/dri/renderD129 --group-add 44 --group-add 990 \
  -v <work-dir>/dri-by-path/card1-renderD129:/dev/dri/by-path:ro \
  -v <models-dir>/qwen38-27b-exl3-SC_5.00bpw_H6:/mnt/exl3model:ro \
  -e UR_L0_USE_COPY_ENGINE=0 -e SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0 \
  --network=host --memory=14g --memory-swap=14g \
  ghcr.io/sergiiob/exl3xpu-mixedk:20261014 \
  models/qwen3.8-27b-exl3-4.00bpw --gpu 0 --model-path /mnt/exl3model --port 8031 \
  --set vllm.gpu_memory_utilization=0.94 --set vllm.max_model_len=196608 \
  --set vllm.enable_prefix_caching=false \
  --set 'served_model_name="Qwen3.8-27B SC5"' \
  --set vllm.reasoning_parser=qwen3 --set vllm.enable_auto_tool_choice=true \
  --set vllm.tool_call_parser=qwen3_coder \
  --set env.EXL3_INT8_PREFILL=1
```
