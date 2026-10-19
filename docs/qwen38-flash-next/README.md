# Qwen3.8-Flash-Next on two Arc Pro B70s

Two-card serving routes, one request at a time. Context capacity, prompt length
and thinking mode are separate settings; each result below links its measured protocol.

| Route | Hardware | Speed | Start here |
|---|---|---:|---|
| **XeStrata**, IQ3_S / IQ3_XXS | **Two B70s, C1** | **86.7 / 86.8 tok/s code decode**, thinking off, P34/G512, n=3 medians | [Matched quant comparison](QWEN38-FLASH-NEXT-XESTRATA.md) |
| llama.cpp SYCL, AtomicChat M64 GGUF | Two B70s, C1 | 33.3 tok/s | [QWEN38-FLASH-NEXT-LLAMACPP.md](QWEN38-FLASH-NEXT-LLAMACPP.md) |
Qwen3.8-27B GPTQ and FP8 TP2 live under [`docs/qwen38-27b/`](../qwen38-27b/README.md).
