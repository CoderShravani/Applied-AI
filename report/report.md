# SMALL MODELS, REAL SYSTEMS
## Fine-tuning • retrieval-augmented generation • local inference

**Engineering report | 17 November 2026 | Shravani Kulkarni**  

### Executive summary
This project builds and measures three practical components around compact language models: a QLoRA adapter that converts natural-language work requests into strict JSON, a retrieval system grounded in a 48-page NIST publication, and a local Ollama benchmark comparing two verified quantization formats. Across the work, the implementation produced an adapter, evaluated 100 untouched test examples, indexed 15 RAG questions, benchmarked 20 model responses, and captured GPU and system-memory observations.

### End-to-end engineering workflow
**Prepare data → adapt model → evaluate structured output → ingest source → retrieve evidence → benchmark serving → assess quality → design deployment.** Each task has executable code, configuration, and machine-readable result summaries in its own folder.

## 1 | QLoRA fine-tuning: natural language to validated JSON

**Model:** `Qwen/Qwen2.5-0.5B-Instruct` • **Method:** 4-bit NF4 QLoRA • **Hardware:** NVIDIA Tesla T4  
**Dataset:** 800 deterministic examples (600 train / 100 validation / 100 held-out test). The test split is separate from training. The output contract contains five fields: `action`, `object`, `priority`, `due_in_days`, and `assignee`.

**Training configuration:** rank 8, alpha 16, dropout 0.05; attention targets `q_proj`, `k_proj`, `v_proj`, `o_proj`; learning rate 2×10⁻⁴; 3 epochs; seed 42; micro-batch 4 × gradient accumulation 4 (effective batch 16); maximum sequence length 256; NF4 quantization with bfloat16 compute. The completed run produced a PEFT adapter and run metadata.

| Held-out metric (n = 100) | Base model | QLoRA model |
|---|---:|---:|
| JSON schema validity | 0% | **100%** |
| Exact five-field match | 0% | **75%** |
| Action accuracy | Not scoreable* | **99%** |
| Object accuracy | Not scoreable* | **96%** |
| Priority accuracy | Not scoreable* | **83%** |
| Deadline accuracy | Not scoreable* | **96%** |
| Assignee accuracy | Not scoreable* | **100%** |

*The evaluator computes field accuracy only for schema-valid predictions. The base model produced no schema-valid predictions, so its field-level scores are not meaningful and are shown as not scoreable rather than interpreting the zero-filled counters as field errors. The adapter achieved valid output on all 100 test inputs and exact matches on 75.*

**What the result means.** The adaptation materially improved adherence to the downstream JSON contract: schema validity rose from 0% to 100%, while three quarters of held-out requests matched all five target fields exactly. Assignee extraction reached 100%; priority was the most challenging field at 83%.

**Three concrete fine-tuned error cases and targeted refinements:**
1. “Show me support case tomorrow, priority as soon as possible.” The deadline was normalized to day 0 rather than day 1. Add contrastive today/tomorrow examples and deterministic relative-date normalization.
2. “Show me work item … priority not urgent …” The object was mapped to `ticket` rather than `task`. Expand synonym coverage with hard negatives for work item, task, ticket, and support case.
3. “... due there is no due date ...” The no-deadline phrase became day 0 rather than -1. Add explicit no-deadline paraphrases and a rule-based no-deadline normalization step.

## 2 | RAG from scratch: NIST AI RMF question answering

**Corpus:** NIST AI Risk Management Framework 1.0 (48 PDF pages; January 2023). **Evaluation:** 15 authored questions; top-5 retrieval; 60-word overlap; 350- and 700-word chunks. The pipeline extracts page-aware text, embeds with `sentence-transformers/all-MiniLM-L6-v2`, normalizes vectors, ranks with NumPy cosine similarity, and combines dense and BM25 scores (0.75 / 0.25). No LangChain or LlamaIndex is used.

| Chunk size | Questions | Recall@5 | Gold-term coverage@5 |
|---|---:|---:|---:|
| 350 words | 15 | 60.0% (9/15) | 100% (15/15) |
| 700 words | 15 | **66.7% (10/15)** | 100% (15/15) |

**Finding:** 700-word chunks led by one additional gold-page hit (6.7 percentage points) while both settings surfaced at least one expected gold term for every question. For this corpus and question set, 700 words is the stronger measured starting point; 350-word chunks remain useful when passage precision and compact context matter.

**Retrieval diagnostics:** Q10 (intended purpose of the AI RMF) and Q13 (response when risk cannot be adequately mitigated) were among the questions whose supporting pages did not enter the top five. Query expansion with source terminology, hybrid lexical weighting, and a cross-encoder re-ranker are targeted improvements. BM25 helps match exact terms and acronyms; a re-ranker can improve ordering of a smaller candidate set at additional latency.

## 3 | Local inference: verified quantization comparison

**Runtime:** Ollama / llama.cpp runner • **GPU:** NVIDIA Tesla T4, 15,360 MiB • **System RAM:** 12 GiB. Ten identical prompts were evaluated per model after warm-up, with temperature 0.2 and a 256-token generation cap (20 measured responses total).

| Configuration | Verified tag | Quantization | Mean generation speed | Mean wall time |
|---|---|---|---:|---:|
| Q4 | `qwen2.5:1.5b` | Q4_K_M | **136.81 tokens/s** | 1.691 s |
| Q8 | `qwen2.5:1.5b-instruct-q8_0` | Q8_0 | 99.31 tokens/s | 2.078 s |

Q4 generated at **37.8% higher mean token throughput** in this run. The blinded rubric review gave Q4 / Q8 mean scores of **3.48 / 3.60 out of 5** overall. Q8 scored higher in completeness (3.2 vs 3.1), instruction-following (3.7 vs 3.6), and clarity (4.1 vs 3.8); correctness tied at 3.4. This is a small, prompt-set-specific quality difference, not a universal model ranking.

**Memory observation:** with both models resident simultaneously, `ollama ps` reported 100% GPU placement for each; their resident sizes were 1.2 GB and 1.8 GB. The simultaneous `nvidia-smi` snapshot showed 3,113 MiB GPU memory used out of 15,360 MiB; process entries were 1,240 MiB and 1,870 MiB. System RAM showed 2.5 GiB used and 10 GiB available. This is a combined snapshot with both models loaded, not isolated per-model peak memory.

## 4 | Deployment design and 7B memory economics

**20 connected users on a 24 GB GPU:** reserve 2 GB for runtime and fragmentation; budget approximately 1.5 GB for a 1.5B 4-bit model and leave 20.5 GB as an illustrative KV-cache, activation, and workspace budget. Assume 1,000 prompt tokens, 200 generated tokens, FP16 KV cache, and 20 active sessions. Begin with eight concurrent sequences; use continuous batching where supported, a bounded queue, per-user limits, context/output caps, prefix caching, cancellation, and back-pressure. Track p50/p95 time-to-first-token, inter-token latency, throughput, queue depth, memory, and errors. This is a controlled-queueing design target—not a claim that 20 simultaneous low-latency decoders were load-tested.

**7B parameter-state estimate (decimal GB; approximate):**

| Method | Approximate parameter-state footprint | Main components |
|---|---:|---|
| Full mixed-precision AdamW | **112 GB** | 14 GB FP16 weights + 28 GB FP32 master weights + 14 GB FP16 gradients + 56 GB Adam moments |
| LoRA | **14.1 GB** | 14 GB frozen FP16 base + ~0.11 GB adapter weights/gradients/optimizer state (7M adapter parameters assumed) |
| QLoRA | **4.31 GB** | 3.5 GB 4-bit base + 0.7 GB metadata allowance + ~0.11 GB adapter state |

These are parameter-state estimates, not peak training memory; activations, temporary buffers, attention workspaces, context length, and fragmentation add to practical requirements. The central engineering outcome is a reproducible compact-model workflow with measured held-out quality, evidence retrieval, and local serving trade-offs—not an unsupported production-capacity claim.
