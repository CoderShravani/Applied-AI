"""Supervised LoRA/QLoRA fine-tuning. Run on a CUDA GPU for practical speed."""
from pathlib import Path
import json, yaml, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())

def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this training configuration. Use free Colab/Kaggle GPU.")
    torch.manual_seed(CFG["seed"])
    data_dir = ROOT / CFG["dataset_dir"]
    for split in ("train", "validation"):
        if not (data_dir / f"{split}.jsonl").exists():
            raise FileNotFoundError("Run make_dataset.py first.")
    tokenizer = AutoTokenizer.from_pretrained(CFG["base_model"], trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = bool(CFG.get("use_4bit", False))
    quant_cfg = None
    if use_4bit:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=CFG["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True
        )
    model = AutoModelForCausalLM.from_pretrained(
        CFG["base_model"], quantization_config=quant_cfg,
        torch_dtype=torch.float16 if not use_4bit else None,
        device_map="auto"
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    lc = CFG["lora"]
    peft_cfg = LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"], bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    system = ("You are a precise information extraction system. Convert the request into "
              "exactly one JSON object with keys action, object, priority, due_in_days, assignee. "
              "Use only allowed enum values. Use -1 for no deadline and null for no assignee. "
              "Return JSON only.")
    def encode(row):
        messages = [
            {"role":"system","content":system},
            {"role":"user","content":row["instruction"]},
            {"role":"assistant","content":row["output"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        enc = tokenizer(text, truncation=True, max_length=CFG["max_seq_length"])
        # Mask prompt tokens; optimize only the assistant response.
        prompt_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=CFG["max_seq_length"])["input_ids"]
        labels = enc["input_ids"].copy()
        labels[:len(prompt_ids)] = [-100] * min(len(prompt_ids), len(labels))
        enc["labels"] = labels
        return enc

    train = Dataset.from_list(read_jsonl(data_dir / "train.jsonl")).map(encode, remove_columns=["id","instruction","input","output"])
    val = Dataset.from_list(read_jsonl(data_dir / "validation.jsonl")).map(encode, remove_columns=["id","instruction","input","output"])
    args = TrainingArguments(
        output_dir=str(ROOT / CFG["output_dir"]), num_train_epochs=CFG["num_train_epochs"],
        learning_rate=CFG["learning_rate"], per_device_train_batch_size=CFG["per_device_train_batch_size"],
        per_device_eval_batch_size=CFG["per_device_train_batch_size"],
        gradient_accumulation_steps=CFG["gradient_accumulation_steps"],
        warmup_ratio=CFG["warmup_ratio"], weight_decay=CFG["weight_decay"],
        logging_steps=CFG["logging_steps"], evaluation_strategy=CFG["evaluation_strategy"],
        save_strategy=CFG["save_strategy"], load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        report_to="none", save_total_limit=2
    )
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                      data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100))
    trainer.train()
    out = ROOT / CFG["output_dir"]
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    metadata = {"base_model":CFG["base_model"], "mode":"QLoRA" if use_4bit else "LoRA",
                "lora":lc, "learning_rate":CFG["learning_rate"], "epochs":CFG["num_train_epochs"],
                "seed":CFG["seed"], "gpu":torch.cuda.get_device_name(0),
                "torch":torch.__version__}
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved adapter and tokenizer to {out}")

if __name__ == "__main__":
    main()
