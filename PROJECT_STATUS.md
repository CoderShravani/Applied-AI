# Project status and reproducibility manifest

**Experiment status:** all three experiment workflows have been executed and their measured summary values are recorded in the task result folders.

## Completed runs

- **Task 1:** Qwen2.5-0.5B-Instruct, 4-bit NF4 QLoRA, Tesla T4, 3 epochs, 800 examples (600/100/100), 100 held-out evaluations. Adapter configuration and run metadata are recorded in `task1_finetuning/results/run_summary.json`. Held-out results: 100% schema validity, 75% exact match; field accuracies 99%, 96%, 83%, 96%, and 100%.
- **Task 2:** NIST AI RMF 1.0 (48 pages), 15 questions, 60-word overlap, top-5, dense/BM25 weights 0.75/0.25. Recall@5: 60.0% (350 words), 66.7% (700 words); gold-term coverage 100% for both.
- **Task 3:** Ollama on Tesla T4; Q4_K_M `qwen2.5:1.5b` and Q8_0 `qwen2.5:1.5b-instruct-q8_0`; ten prompts per model. Mean throughput 136.81 and 99.31 tokens/s. Blinded rubric means 3.48/5 and 3.60/5.

## Files included

The repository includes scripts, configurations, the 800-example Task 1 dataset, the 15-question RAG evaluation set, prompt set, machine-readable experiment summaries, the 7B memory estimate, concurrency design, and the four-page PDF report.
