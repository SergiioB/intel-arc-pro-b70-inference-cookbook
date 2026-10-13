# SC_5.00bpw_H6_V6 check — 2026-10-13

Does turboderp's `SC_5.00bpw_H6_V6` change text quality or speed versus the
served `SC_5.00bpw_H6`, and does it load on the SC5 image?

- Candidate: `turboderp/Qwen3.8-27B-exl3` branch `SC_5.00bpw_H6_V6` @
  `f33f26d929e2b20ef21361145d582f5239e3831f`, size- and sha256-verified
  against the Hugging Face API.
- Served: branch `SC_5.00bpw_H6` @ `dfe2f4fd71bf9a1994051d4d47e6460fa96dd723`.
- Image: `ghcr.io/sergiiob/exl3xpu-mixedk:20261013@sha256:5e2d1fab3a8ba559b4d3af7fb1fa5eeff324c4f4274847e3079ec2d57c698ba4`
  (patches and build source: [SergiioB/exl3xpu-mixedk](https://github.com/SergiioB/exl3xpu-mixedk)).

## Files

- `tensor-diff.json`: every safetensors tensor of both checkpoints hashed on
  CPU. Language model (incl. MTP head): 2,093 common tensors, 2,093
  identical, 18,979,263,076 bytes each. Vision tower: 250 common tensors,
  none identical; fp16 921,460,192 B in H6 vs EXL3 K=6 linears
  570,139,024 B in V6.
- `engine3.log`: one load attempt of V6 with the SC5 recipe command,
  changing only the model mount and `gpu_memory_utilization` 0.85 (the card
  also drives a desktop). Weight loading fails:
  `There is no module or parameter named 'blocks.0.attn.k_proj' in
  Qwen3_VisionTransformer`. The exl3xpu vLLM plugin keeps the vision tower
  unquantized and has no path for EXL3 vision linears.

`quantization_config.json` and the calibration trace are also identical
between the two branches.

## Verdict

V6 is a vision-only repack. Text output and text speed are identical to
`SC_5.00bpw_H6` by construction (byte-identical language-model tensors).
It saves about 350 MB of weights and does not load on this image. Keep
serving `SC_5.00bpw_H6`.
