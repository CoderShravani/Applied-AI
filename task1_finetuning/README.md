# Task 1 — QLoRA fine-tuning for request-to-JSON parsing

## Task definition

Convert a plain-English work request into one JSON object with exactly five fields: `action`, `object`, `priority`, `due_in_days`, and `assignee`. Allowed values are defined in `config.yaml` and the evaluator. No prose should accompany the JSON.

## Completed experiment

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Method: 4-bit NF4 QLoRA; bfloat16 compute
- Dataset: 800 examples (600 train / 100 validation / 100 held-out test)
- Adapter: rank 8, alpha 16, dropout 0.05; `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Optimisation: learning rate `2e-4`, 3 epochs, seed 42, micro-batch 4 × accumulation 4 (effective batch 16)
- Hardware: NVIDIA Tesla T4

Held-out QLoRA results: **100% JSON schema validity**, **75% exact match**, action 99%, object 96%, priority 83%, deadline 96%, and assignee 100%. The baseline produced no schema-valid outputs on this evaluation. Field-level baseline accuracy is not scoreable because the evaluator only scores fields for schema-valid predictions.

The saved run summary and three representative fine-tuned error cases are in `results/`. The examples are used to guide focused improvements in relative-date normalization, object synonym coverage, and explicit no-deadline handling.

## Run

```bash
python make_dataset.py
python validate_dataset.py
python train_lora.py
python evaluate.py --model-path outputs/qwen25-05b-lora
python memory_estimator.py
```

The trainer uses the test split only for final evaluation, not for gradient updates. Dataset generation is deterministic with seed 42.

## Adapter artifact

The trained adapter is included in `outputs/qwen25-05b-lora/`. The directory contains `adapter_model.safetensors`, `adapter_config.json`, tokenizer files, the chat template, run metadata, and an artifact checksum note. It is configured for `Qwen/Qwen2.5-0.5B-Instruct` and can be loaded with PEFT.
