# Task 3 — Local inference and quantisation benchmark

## Verified configurations

| Label | Ollama tag | Verified quantization | Model size reported by Ollama |
|---|---|---|---:|
| Q4 | `qwen2.5:1.5b` | Q4_K_M | 1.2 GB |
| Q8 | `qwen2.5:1.5b-instruct-q8_0` | Q8_0 | 1.8 GB |

Both quantization formats were verified with `ollama show`. The experiment ran on an NVIDIA Tesla T4 with 15,360 MiB GPU memory and 12 GiB system RAM.

## Measured benchmark

Ten prompts per configuration (20 responses total), one timed run per prompt, one warm-up per model, temperature 0.2, maximum output 256 tokens, and 4,096-token context as reported by `ollama ps`.

- Q4_K_M: **136.81 tokens/s** mean generation throughput; **1.691 s** mean wall time.
- Q8_0: **99.31 tokens/s** mean generation throughput; **2.078 s** mean wall time.
- Q4 throughput advantage in this run: **37.8%**.

## Quality review

The same ten prompts were assessed blind using four 1–5 criteria. Q4 means: correctness 3.40, completeness 3.10, instruction-following 3.60, clarity 3.80, overall 3.48. Q8 means: correctness 3.40, completeness 3.20, instruction-following 3.70, clarity 4.10, overall 3.60. These are prompt-set-specific rubric scores, not a universal model ranking.

## Memory observation

With both models loaded at once, `ollama ps` showed 100% GPU placement for both. The simultaneous `nvidia-smi` snapshot reported 3,113 MiB / 15,360 MiB used; process entries were 1,240 MiB and 1,870 MiB. System RAM showed 2.5 GiB used and 10 GiB available. This is a combined resident snapshot, not isolated per-model peak memory.

## Run

```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:1.5b-instruct-q8_0
ollama serve
python benchmark.py
python quality_review.py
```

The verified tags and run settings are recorded in `config.yaml`; measured summaries are in `results/experiment_summary.json`.

## KV cache and quantization

The KV cache stores attention keys and values for tokens already processed in the current context. It grows with context length, layer count, KV-head dimensions, cache precision, and concurrent sequences. Quantization reduces weight storage; runtime memory also includes KV cache, activations, temporary buffers, and framework overhead.

## 20-user design target on a 24 GB GPU

The accompanying `concurrency_plan.md` specifies a bounded queue, an initial eight active sequences, context/output caps, continuous batching where supported, prefix caching, and latency/memory monitoring. Twenty connected sessions are handled through controlled concurrency and queueing. The design does not claim that twenty simultaneous low-latency decoders were load-tested.
