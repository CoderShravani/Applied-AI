"""Compare base and adapter model on the untouched held-out test split."""
from pathlib import Path
import argparse, json, re, yaml, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
SYSTEM = ("You are a precise information extraction system. Convert the request into exactly one "
          "JSON object with keys action, object, priority, due_in_days, assignee. "
          "Allowed action: create/update/delete/search; object: task/meeting/document/ticket; "
          "priority: low/medium/high. Use -1 for no deadline and null for no assignee. Return JSON only.")

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match: return None
    try: return json.loads(match.group())
    except Exception: return None

def valid(x):
    return (isinstance(x, dict) and set(x) == {"action","object","priority","due_in_days","assignee"}
            and x.get("action") in {"create","update","delete","search"}
            and x.get("object") in {"task","meeting","document","ticket"}
            and x.get("priority") in {"low","medium","high"}
            and isinstance(x.get("due_in_days"), int)
            and (x.get("assignee") is None or isinstance(x.get("assignee"), str)))

def load_model(model_path=None):
    base = AutoModelForCausalLM.from_pretrained(CFG["base_model"], torch_dtype="auto", device_map="auto")
    if model_path:
        base = PeftModel.from_pretrained(base, model_path)
    tok = AutoTokenizer.from_pretrained(model_path or CFG["base_model"])
    return base.eval(), tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(ROOT / CFG["output_dir"]))
    ap.add_argument("--base-only", action="store_true")
    args = ap.parse_args()
    test = [json.loads(x) for x in (ROOT/CFG["dataset_dir"]/"test.jsonl").read_text().splitlines()]
    model, tok = load_model(None if args.base_only else args.model_path)
    rows, exact, valid_n, field_ok = [], 0, 0, {k:0 for k in ["action","object","priority","due_in_days","assignee"]}
    for ex in test:
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":ex["instruction"]}]
        prompt=tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs=tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids=model.generate(**inputs, max_new_tokens=100, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        generated=tok.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred=extract_json(generated)
        gold=json.loads(ex["output"])
        is_valid=valid(pred)
        valid_n += int(is_valid)
        is_exact=is_valid and pred==gold
        exact += int(is_exact)
        if is_valid:
            for k in field_ok: field_ok[k] += int(pred.get(k)==gold[k])
        rows.append({"id":ex["id"],"input":ex["instruction"],"gold":gold,"raw_output":generated,"prediction":pred,"valid":is_valid,"exact":bool(is_exact)})
    n=len(test)
    metrics={"n":n,"json_schema_validity":valid_n/n,"exact_match":exact/n,
             "field_accuracy":{k:v/n for k,v in field_ok.items()}}
    tag="base" if args.base_only else "finetuned"
    out=ROOT/"results"; out.mkdir(exist_ok=True)
    (out/f"{tag}_predictions.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    (out/f"{tag}_metrics.json").write_text(json.dumps(metrics,indent=2))
    failures=[r for r in rows if not r["exact"]]
    (out/f"{tag}_failures.json").write_text(json.dumps(failures,indent=2,ensure_ascii=False))
    print(json.dumps(metrics,indent=2))
    print(f"Saved predictions and {len(failures)} failure cases in {out}")

if __name__=="__main__": main()
