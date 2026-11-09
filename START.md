# Start one model on one Intel Arc Pro B70

Use the [Qwen3.8-27B EXL3 route](docs/qwen38-27b/EXL3-XPU.md) for the first run.
This is the same primary route selected by the README. Start at 8,192 context
with speculative decoding and prefix reuse disabled. This reduces the number
of moving parts during setup; it does not establish general model quality.

## 1. Check the host

```bash
python3 scripts/doctor.py
```

Use Linux with an accessible B70 render node, Docker access, and room for both
the model and the pinned engine image. Read the doctor's failures before
continuing. Stop only your own servers; do not replace another GPU workload.
For driver installation, use [host setup](docs/FULL-SETUP-COMMANDS.md).
Other Intel cards and Windows need their own validated instructions.

## 2. Install the launcher

```bash
curl -fsSL https://xecores.com/downloads/get.sh | sh
```

The published installer verifies the archive checksum. The
[launcher README](https://github.com/SergiioB/b70-launcher#install-current-user-only)
has inspect-first instructions and prerequisites. The engine and model are
separate downloads. The EXL3 recipe selects an image digest and model revision;
inspect them in the command preview before launch.

## 3. Review and download the EXL3 artifact

```bash
b70 download qwen38-27b -e exl3 --dry-run
b70 download qwen38-27b -e exl3
```

The first command checks size and free disk space without downloading. The
second requests confirmation. Keep the model and the large Docker image on a
disk with enough free space. If the preview asks for an image pull or isolated
Docker setup, follow the [EXL3 setup](docs/qwen38-27b/EXL3-XPU.md#serving-recipe)
after checking that disk's budget.

## 4. Preview and start the baseline

```bash
b70 launch qwen38-27b -e exl3 --ctx 8192 --port 8100 --prefill fp16 --extra '--set vllm.speculative_config=null --set vllm.enable_prefix_caching=false' --dry-run
b70 launch qwen38-27b -e exl3 --ctx 8192 --port 8100 --prefill fp16 --extra '--set vllm.speculative_config=null --set vllm.enable_prefix_caching=false'
```

The preview must contain `vllm.speculative_config=null` and
`vllm.enable_prefix_caching=false`. The explicit `--extra` override works with
published launchers whose EXL3 builder ignores `--no-mtp`. The published 0.6.10 launcher also lacks `--cache`; pass both settings through
`--extra`. Once a release includes the switches, they can replace the override. First loading
may take several minutes for compilation and graph capture. Do not start a
second server while the first loads.

## 5. Check the answer and stop your server

```bash
b70 status
b70 test 8100
b70 stop 8100
```

The default test asks for `B70-OK`. Check the returned text. A health response
or one correct answer proves only that the endpoint runs. Before using a
route for your work, follow [recipe confirmation](docs/CONFIRM-A-RECIPE.md)
and retain representative task outputs. Connect a client through
[client setup](docs/CONNECTING-CLIENTS.md).

The older GPTQ guide is [historical](docs/qwen38-27b/QWEN38-VLLM-XPU.md).
Its commands are retained as evidence and are excluded from this first run.
MTP3 can change the greedy answer; cache-on claims need checks on the selected
engine, artifact, and cache-hit path. Full context, concurrent clients, and
speed require separate evidence in the [catalog](docs/BENCHMARK-CATALOG.md).

For contributors, [how the Intel reference work is organized](docs/INTEL-REFERENCE-PLAN.md)
separates qualified routes, published screens, and future investigations.
