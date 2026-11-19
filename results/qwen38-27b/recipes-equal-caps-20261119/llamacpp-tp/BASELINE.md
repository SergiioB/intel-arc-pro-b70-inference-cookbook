# Card 1 Qwen3.8-27B Q4_K_M baseline (llama.cpp SYCL, upstream)

One measured C1 request per depth after a separate same-shape warmup.
llama.cpp `6ac5702` (patch build; the one-card path does not reach the patched allreduce code), SYCL, one Arc Pro B70 at PCI `0f:00.0`, 230 W cap.
Full GPU offload, q8_0 K/V, 40,960-token context, 512/512 batch/microbatch, graph capture off.
Greedy chat, thinking off, no speculation, zero cached prompt tokens, 128 output tokens (EOS ignored).
Client decode = streamed rate after the first content token. Engine decode = server `predicted_per_second`.

Server command: `$WORK/tpllama/llama.cpp/build/bin/llama-server -m $DATA/sata2/gguf/qwen38-27b-Q4_K_M.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -b 512 -ub 512 -t 8 -lm none -dev SYCL0 -sm none --fit off -c 40960 -np 1 --no-cache-prompt --cache-ram 0 --reasoning off --alias qwen38 --host 127.0.0.1 --port 8051`

| Depth | Input tokens | Output tokens | Client decode tok/s | Engine decode tok/s | TTFT s | Wall s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 78 | 128 | 28.11 | 28.11 | 0.250 | 4.768 |
| 8k | 7985 | 128 | 26.92 | 26.92 | 25.376 | 30.095 |
| 32k | 32560 | 128 | 24.79 | 24.79 | 105.142 | 110.266 |

Model SHA256 `2554eb0dcdb99710b316efea0d43a271db7c3161fd1db44b8fcf33f9b97fbece`.
Raw cells `$WORK/tpllama/evidence/base-single/served.jsonl`.
Raw cells SHA256 `0be45af4da8fd2ff45bf48f4f999cf9f30bc0aff9ef3267393dc707bb732038b`.
llama-bench (same build, card 1): pp512 319.07 ± 1.16 tok/s (n=3), tg128 28.28 ± 0.02 tok/s (n=3).
Measured cells are in [one-card-depth.jsonl](one-card-depth.jsonl).
