"""Transparent back-of-envelope memory estimate for a 7B parameter model."""
import json
from pathlib import Path

# Decimal GB throughout. Actual peak memory varies with sequence length,
# micro-batch, optimizer, checkpointing, kernels and allocator fragmentation.
P = 7_000_000_000
BYTES = {"fp32":4, "fp16":2, "int8":1, "int4":0.5}
# Full AdamW training, mixed-precision weights: fp16 weights + fp32 master weights,
# fp32 gradients, and two fp32 Adam moments => approximately 16 bytes/parameter.
full = {
    "weights_fp16_GB": P*2/1e9,
    "master_weights_fp32_GB": P*4/1e9,
    "gradients_fp16_GB": P*2/1e9,
    "adam_moments_fp32_GB": P*8/1e9,
}
# LoRA: frozen fp16 base weights, trainable adapter params are assumed 0.1% of base.
adapter_params = int(P*0.001)
lora = {
    "frozen_base_fp16_GB": P*2/1e9,
    "adapter_params_assumed": adapter_params,
    "adapter_training_state_16_bytes_per_adapter_param_GB": adapter_params*16/1e9,
}
# QLoRA: 4-bit base + quantization metadata approx 0.1 byte/parameter overhead;
# adapter state same assumption. Double quant can lower overhead further.
qlora = {
    "4bit_base_GB": P*0.5/1e9,
    "quantization_metadata_assumed_GB": P*0.1/1e9,
    "adapter_training_state_GB": adapter_params*16/1e9,
}
result = {
    "assumptions": {
        "parameters": "7.0 billion",
        "units": "decimal GB (1 GB = 1e9 bytes)",
        "adapter_size": "0.1% of base parameters (illustrative; actual depends on target modules/rank)",
        "excluded": "activations, temporary buffers, CUDA context, fragmentation, attention workspace"
    },
    "full_finetuning": {**full, "parameter_state_total_GB": sum(full.values()),
                        "practical_peak_estimate_GB": "at least ~112 GB plus activations/workspace; often 120+ GB"},
    "lora": {**lora, "parameter_state_total_GB": lora["frozen_base_fp16_GB"] + lora["adapter_training_state_16_bytes_per_adapter_param_GB"],
             "practical_peak_estimate_GB": "roughly 20-30+ GB; base weights dominate"},
    "qlora": {**qlora, "parameter_state_total_GB": sum(qlora.values()),
              "practical_peak_estimate_GB": "roughly 8-12+ GB; sequence length and batch can raise this"}
}
print(json.dumps(result, indent=2))
Path(__file__).with_name("memory_estimate.json").write_text(json.dumps(result, indent=2))
