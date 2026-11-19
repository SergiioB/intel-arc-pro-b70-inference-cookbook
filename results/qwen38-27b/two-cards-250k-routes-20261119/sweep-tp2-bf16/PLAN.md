# Qwen3.8-27B context sweep plan

Measure one streamed 256-token C1 code response at each requested prompt
length that fits. Keep temperature at 0, thinking and prompt caching off.
Use one full-shape warm-up and three measured samples per arm and length.
Within each TensorFold length, run serial and DFlash2 on the same server.
Alternate arm order between measured pairs. Match prompt bytes within each
pair and compare full token SHA-256. Nonces at the start of the prompt vary
between repetitions.

Each job starts below the queue's 60 °C entry limit. Run continuously within
the job and capture both card temperatures, power caps, and clock caps
before and after each request. Stop a measurement on an invalid response.
An early EOS gets one replacement attempt. Retain the early response.
Stop starting work at the night deadline. Keep each job under 90 minutes.

Report arithmetic mean and range of the streamed rate after first content,
mean time to first content, and mean prompt rate. TensorFold reports native
prefill time. For exl3xpu use cold input tokens divided by client TTFT unless
the stream supplies an isolated prefill timer. A listed point must have three
valid samples with exact usage counts and zero cache hits. A drafted point
must also have three matching pairs and actual draft work.

Queue order:

1. Attempt TP2 bf16 with the requested 270,336-token window. The pinned
   source's XPU limit is 262,144 tokens. Preserve the launcher response.
2. Measure TP2 bf16 at the 262,144-token supported window through 131,072
   prompt tokens. Budget 30 minutes. A 262,144-token prompt needs at least
   262,400 positions with its reply.
3. Measure TP1 bf16 at 139,264 positions through 131,072 prompt tokens.
   Budget 40 minutes.
4. Check the TP1 int8 HTTP admission result at 262,144 input plus 256
   output tokens with its requested 262,144-position window. Budget 5 minutes.
5. Measure EXL3 serial through 131,072. Budget 20 minutes. Add MTP3 only if
   the serial job completes with sufficient time before 06:30 CET.

This is an external-protocol characterization of existing builds. No
performance improvement, alternate runtime adoption, or hardware change
is proposed. No job is expected to use an hour of GPU time.
