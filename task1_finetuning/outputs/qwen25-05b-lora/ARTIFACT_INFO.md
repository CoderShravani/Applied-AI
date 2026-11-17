# Trained QLoRA adapter

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Method: 4-bit NF4 QLoRA
- LoRA rank: 8; alpha: 16; dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Learning rate: `0.0002`; epochs: 3; seed: 42
- Recorded training GPU: NVIDIA Tesla T4
- Weights: `adapter_model.safetensors`
- SHA-256: `260ab0f83017b08c65af56a67f9e446faa693007e807bae2bebde2c8cef32b5a`

Load with PEFT using the base model named in `adapter_config.json`.
