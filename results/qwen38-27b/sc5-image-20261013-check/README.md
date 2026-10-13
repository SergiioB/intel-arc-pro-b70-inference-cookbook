# exl3xpu-mixedk 20261013 image check — 2026-10-13

Checks that the image built from public source behaves like the prior
`20261012c` image before it became the SC5 pin.

- New pin: `ghcr.io/sergiiob/exl3xpu-mixedk:20261013@sha256:5e2d1fab3a8ba559b4d3af7fb1fa5eeff324c4f4274847e3079ec2d57c698ba4`.
  Built from the public mixed-K patch source at commit `244f57e5c220a5ea6e438869fa996c5b7e96ca75`;
  patches and image source: [SergiioB/exl3xpu-mixedk](https://github.com/SergiioB/exl3xpu-mixedk).
  The checks below ran on that build; `20261013` adds OCI labels only, no
  file changes.
- Prior pin: `ghcr.io/sergiiob/exl3xpu-mixedk:20261012c@sha256:7bb0c5cfed8a35da8f4db0034338d8dbe5f53731253d96ea8f1db2b5bddd4a40`.
- Model: `turboderp/Qwen3.8-27B-exl3` branch `SC_5.00bpw_H6` @ `dfe2f4fd71bf9a1994051d4d47e6460fa96dd723`.
- Hardware: one Arc Pro B70, 230 W cap. Served through the b70-launcher
  `qwen38-27b-sc5` recipe with only the image overridden.

| Check | 20261013 build | 20261012c | File |
|---|---|---|---|
| Mixed-K kernel checks on real SC5 tensors | 46/46 pass | 46/46 pass | `kernels.log` |
| Exact-answer canaries (greedy) | 6/6 | 6/6 | `canary.txt` |
| Long prompts (8,161–8,162 tokens) | 3/3 | 3/3 | `canary.txt` |
| Greedy p512/g128 MTP3, post-first tok/s, median of 3 after 1 warmup | 70.32 | 70.29 | `cells.jsonl`, `baseline-20261012c-cells.jsonl` |

The greedy decode cell is a screen for regressions. It uses greedy sampling,
so it is not comparable with the recipe-sampling A/B in
[`sc5-mixedk-ab-20261012`](../sc5-mixedk-ab-20261012/README.md).
