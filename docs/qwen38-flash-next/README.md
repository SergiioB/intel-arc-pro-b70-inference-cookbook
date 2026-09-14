# Qwen3.8-Flash-Next on two Arc Pro B70s

llama.cpp SYCL on two cards, one request at a time. 8K / 16K / 128K are
**context windows**. The same recipe serves all three; 16K is the prefill
peak, not a maximum context.

| Route | Hardware | Start here |
|---|---|---|
| llama.cpp SYCL, AtomicChat M64 GGUF | Two B70s, C1 | [QWEN38-FLASH-NEXT-LLAMACPP.md](QWEN38-FLASH-NEXT-LLAMACPP.md) |

Qwen3.8-27B GPTQ and FP8 TP2 live under [`docs/qwen38-27/`](../qwen38-27/README.md).
