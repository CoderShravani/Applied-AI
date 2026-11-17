# Small Models, Real Systems
### A reproducible engineering study of fine-tuning, retrieval, and local inference

This repository contains three implemented experiments and their measured result summaries. The project follows one practical question: how can compact language models be adapted to a task, grounded in a long technical source, and served efficiently on constrained hardware?

## Results at a glance

| Workstream | Dataset / evaluation | Measured result |
|---|---|---|
| QLoRA fine-tuning | 800 examples; 100 held-out test cases | 100% JSON schema validity; 75% exact match; 99% action, 96% object, 83% priority, 96% deadline, 100% assignee accuracy |
| RAG | 48-page NIST AI RMF; 15 questions; top-5 retrieval | Recall@5: 60.0% with 350-word chunks; 66.7% with 700-word chunks; 100% gold-term coverage in both runs |
| Local inference | 10 prompts × 2 quantization settings | Q4_K_M: 136.81 tokens/s; Q8_0: 99.31 tokens/s; Q4 was 37.8% faster in this run |
| Quality review | 20 blinded model responses; four criteria | Overall rubric means: Q4 3.48/5; Q8 3.60/5 |

## Repository map

```text
main/
├── README.md
├── requirements.txt
├── LICENSE
├── report/
│   ├── report.md
│   ├── make_report.py
│   └── report.pdf
├── task1_finetuning/
│   ├── config.yaml
│   ├── make_dataset.py
│   ├── validate_dataset.py
│   ├── train_lora.py
│   ├── evaluate.py
│   ├── memory_estimator.py
│   ├── data/                  # 600 train / 100 validation / 100 test
│   ├── outputs/qwen25-05b-lora/  # trained adapter weights + tokenizer/config
│   └── results/               # measured metrics, run summary, failure examples
├── task2_rag/
│   ├── config.yaml
│   ├── ingest.py
│   ├── rag.py
│   ├── bm25.py
│   ├── evaluate.py
│   ├── questions.json
│   └── results/               # measured retrieval summary and diagnostics
└── task3_local_inference/
    ├── config.yaml
    ├── benchmark.py
    ├── prompts.json
    ├── quality_review.py
    ├── concurrency_plan.md
    └── results/               # measured benchmark and quality summaries
```

## Task 1 — QLoRA structured extraction

Base model: `Qwen/Qwen2.5-0.5B-Instruct`. Training used 4-bit NF4 QLoRA, rank 8, alpha 16, dropout 0.05, attention projections `q_proj`, `k_proj`, `v_proj`, `o_proj`, learning rate `2e-4`, three epochs, seed 42, micro-batch 4 and gradient accumulation 4 (effective batch 16). The held-out set contains 100 examples. The saved result summary is in `task1_finetuning/results/`.

Run on a compatible GPU environment:

```bash
cd task1_finetuning
python make_dataset.py
python validate_dataset.py
python train_lora.py
python evaluate.py --model-path outputs/qwen25-05b-lora
python memory_estimator.py
```

The trained adapter is included at `outputs/qwen25-05b-lora/`, including `adapter_model.safetensors`, adapter configuration, tokenizer files, chat template, and run metadata. Evaluation can be rerun directly against this directory.

## Task 2 — RAG without orchestration frameworks

Corpus: the official [NIST AI Risk Management Framework 1.0 PDF](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf), 48 pages. The implementation uses `pypdf`, overlapping page-aware chunks, `sentence-transformers/all-MiniLM-L6-v2`, normalized embeddings, NumPy cosine similarity, and a custom BM25 scorer. Dense/BM25 hybrid weights are 0.75/0.25. No LangChain or LlamaIndex is used.

```bash
cd task2_rag
python ingest.py --download
python evaluate.py --chunk-sizes 350 700 --top-k 5
python rag.py --question "What does the framework mean by trustworthy AI?"
```

Measured Recall@5 was 60.0% for 350-word chunks and 66.7% for 700-word chunks over the 15-question evaluation set. Results and retrieval diagnostics are in `task2_rag/results/evaluation_summary.json`.

## Task 3 — Ollama inference benchmark

Verified model tags and quantization:

- `qwen2.5:1.5b` — Q4_K_M
- `qwen2.5:1.5b-instruct-q8_0` — Q8_0

Both quantization labels were checked with `ollama show`. The run used ten prompts per model, warm-up, temperature 0.2, and a 256-token output cap.

```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:1.5b-instruct-q8_0
ollama serve
cd task3_local_inference
python benchmark.py
python quality_review.py
```

The completed run measured 136.81 tokens/s for Q4_K_M and 99.31 tokens/s for Q8_0. The result summary and quality rubric aggregates are stored under `task3_local_inference/results/`.

## Report

The four-page PDF is `report/report.pdf`. Rebuild it with:

```bash
python report/make_report.py
```

The report uses the measured run summaries and states the assumptions behind the 7B memory estimate and 20-user serving design. The 20-user plan is an architecture target with queueing and bounded concurrency; it is not represented as a completed load test.

## Reproducibility notes

- Training/evaluation splits are 600/100/100 and the held-out test set is not used for training.
- Retrieval reports page-based Recall@5 and gold-term coverage for 15 questions.
- Inference uses the same ten prompts, warm-up, temperature, and output-token cap for both model configurations.
- GPU memory was sampled with both Ollama models resident simultaneously; the 3,113 MiB observation is combined memory, not per-model peak.
- All benchmark values are hardware- and runtime-specific. Machine-readable summaries are included beside each task.
