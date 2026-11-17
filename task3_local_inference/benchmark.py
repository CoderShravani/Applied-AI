"""Repeatable local Ollama benchmark; captures outputs and machine telemetry."""
from pathlib import Path
import argparse, csv, json, platform, subprocess, time, requests, yaml
try:
    import psutil
except ImportError:
    psutil=None

ROOT=Path(__file__).parent
CFG=yaml.safe_load((ROOT/"config.yaml").read_text())
PROMPTS=json.loads((ROOT/"prompts.json").read_text())

def gpu_memory():
    try:
        out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits"],text=True,stderr=subprocess.DEVNULL)
        return out.strip().splitlines()[0]
    except Exception: return "unavailable"

def ollama_generate(tag,prompt):
    start=time.perf_counter()
    r=requests.post("http://localhost:11434/api/generate",json={
        "model":tag,"prompt":prompt,"stream":False,
        "options":{"temperature":CFG["temperature"],"num_predict":CFG["max_tokens"]}
    },timeout=600)
    r.raise_for_status()
    elapsed=time.perf_counter()-start
    data=r.json()
    count=int(data.get("eval_count") or 0)
    eval_ns=int(data.get("eval_duration") or 0)
    tps=count/(eval_ns/1e9) if eval_ns else (count/elapsed if elapsed else 0)
    return data.get("response",""),elapsed,count,tps,data

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--allow-placeholder",action="store_true")
    a=ap.parse_args()
    models=CFG["models"]
    bad=[m for m in models if "REPLACE_" in m["ollama_tag"]]
    if bad and not a.allow_placeholder:
        raise SystemExit("Edit config.yaml: replace placeholder with a locally pulled, verified second model tag.")
    out=ROOT/CFG["results_dir"]; out.mkdir(exist_ok=True)
    rows=[]
    for m in models:
        tag=m["ollama_tag"]
        print(f"\nModel: {m['label']} ({tag})")
        if CFG["warmup"]:
            try: ollama_generate(tag,"Reply with the word ready.")
            except Exception as e: raise RuntimeError(f"Warm-up failed for {tag}: {e}")
        for i,prompt in enumerate(PROMPTS,1):
            text,elapsed,count,tps,raw=ollama_generate(tag,prompt)
            row={"label":m["label"],"tag":tag,"quantization_claim":m["quantization"],
                 "prompt_id":i,"prompt":prompt,"output":text,"wall_seconds":round(elapsed,4),
                 "generated_tokens":count,"tokens_per_second":round(tps,3),
                 "system_ram_used_percent":psutil.virtual_memory().percent if psutil else None,
                 "gpu_memory_used_total_MiB":gpu_memory(),
                 "load_duration_ns":raw.get("load_duration"),"prompt_eval_count":raw.get("prompt_eval_count"),
                 "eval_duration_ns":raw.get("eval_duration")}
            rows.append(row)
            print(f"  {i:02d}: {tps:.2f} tok/s | {elapsed:.2f}s")
    path=out/"benchmark.csv"
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    meta={"platform":platform.platform(),"python":platform.python_version(),"models":models,
          "prompt_count":len(PROMPTS),"warmup":CFG["warmup"],"temperature":CFG["temperature"],
          "max_tokens":CFG["max_tokens"],"note":"Quantization strings must be verified using `ollama show`."}
    (out/"run_metadata.json").write_text(json.dumps(meta,indent=2))
    print(f"\nSaved {len(rows)} rows to {path}")

if __name__=="__main__": main()
