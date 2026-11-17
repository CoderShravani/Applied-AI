# Serving 20 concurrent users on one 24 GB GPU — back-of-envelope design

## Assumptions (make them explicit in the report)

- One 24 GB GPU; reserve 2 GB for CUDA/runtime/fragmentation.
- A 1.5B instruction model in 4-bit weights; rough weight footprint 1.0–1.3 GB including quantization metadata/runtime representation.
- Average prompt 1,000 tokens, average generated answer 200 tokens.
- FP16 KV cache; actual bytes/token depends on the model's layer count, KV-head count, head dimension, and cache layout. Use the actual model config for a precise estimate.
- 20 active sessions, not necessarily 20 simultaneous decode tokens at every instant.

## Plan

1. Run one model server (vLLM with continuous batching if the model format is supported; otherwise llama.cpp/Ollama with a queue and bounded parallelism).
2. Set a maximum context length and output-token cap. Reject or summarize oversized requests before they reach the model.
3. Enable continuous batching and prefix caching where available. Reuse shared system prompts.
4. Limit simultaneous sequences based on measured KV-cache use, not just weight size. Start conservatively (e.g. 8 active sequences), measure peak memory and latency, then increase.
5. Put an API gateway in front: request queue, per-user concurrency limit, timeout, cancellation, and back-pressure.
6. Track p50/p95 time-to-first-token, inter-token latency, throughput, queue depth, GPU memory, and error rate.
7. If 20 users arrive at once, queue some requests rather than promise 20-way low-latency decoding. A small model can fit easily while KV cache and compute throughput remain the limiting resources.

## Illustrative memory budget

- Quantized weights + runtime: ~1.5 GB (verify with actual model).
- CUDA/runtime reserve: ~2.0 GB.
- KV cache + activations/workspace budget: remaining ~20.5 GB.
- With a small 1.5B model, memory is unlikely to be the only bottleneck; token-generation compute and queueing may dominate. For a 7B model, KV cache and weight footprint are materially larger.

Do not state that 20 concurrent users are supported until a load test verifies the target prompt lengths, output lengths, latency SLO, and error rate. The design target is 20 connected users with controlled concurrency and queueing, not necessarily 20 simultaneous full-speed decoders.
