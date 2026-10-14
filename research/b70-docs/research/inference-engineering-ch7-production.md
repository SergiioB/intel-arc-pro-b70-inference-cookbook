# Inference Engineering (Philip Kiely) - Chapter 7 Notes

## 7. Production Infrastructure

### Containerization & Dependencies
- Use **Docker**. Container images should be lightweight, starting from inference-engine official base images (like vLLM or TensorRT-LLM).
- **Pin all dependencies** (CUDA, PyTorch, xformers, triton, flash-attn) to exact versions to avoid breaking changes when spinning up new nodes. 
- Avoid blindly using NVIDIA NIMs (NVIDIA Inference Microservices) if you want full customization; they are highly opinionated pre-built containers.

### Autoscaling
- **Metrics:** Base scaling on **traffic** (queue depth, request volume) rather than just **utilization** (GPU compute), as LLM requests are bursty.
- **Batching & Concurrency:** Use **continuous batching** (in-flight token-level batching) to maximize throughput without cratering latency. Set concurrency targets corresponding to your batch sizes.
- **Scale to zero:** Possible, but limited by **Cold Starts**.
- **Cold Starts:** Time to spin up nodes is dictated by image loading, weight loading, and engine compilation. 
  - *Fix:* Cache compiled engines (TensorRT-LLM) directly on network storage inside the datacenter, load weights over high-bandwidth internal network (not S3 over public internet).

### Routing & Load Balancing
- **Cache-aware routing:** Send requests to the replica that already holds the relevant KV cache prefix (crucial for long conversations or repetitive codebase queries).
- **Global Load Balancing:** Time zones matter. 1 time zone = ~5ms network latency. Route traffic geo-locally to stay within tight latency SLAs.

### Multi-Cloud & Reliability
- Don't rely on a single cluster. "Hardware failure is expected" (e.g., Llama 3 training lost a GPU once every 50k hours).
- **Bin Packing / Global Orchestration:** Treat GPUs across AWS, GCP, Coreweave, and on-premise as a single fluid pool of compute (Multi-Cloud Capacity Management). 
- Use **Active-Active** or **Active-Passive** fallback clusters for true reliability. 

### Client Code & Protocols
- Network overhead from TLS handshakes can consume 10% of a 300ms SLA. Re-use client sessions.
- **Bi-directional streaming:** For real-time Audio/Video (like Voice Agents), use **WebSockets** (unstructured) or **gRPC** (structured) instead of standard HTTP chunking.
- **Asynchronous endpoints:** Use for high-throughput batching tasks (like transcribing an hour-long podcast). Return the result to a webhook instead of keeping the connection open.