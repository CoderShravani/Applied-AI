"""Create a human scoring sheet for paired model outputs; no synthetic quality claims."""
from pathlib import Path
import csv
ROOT=Path(__file__).parent
src=ROOT/"results"/"benchmark.csv"
if not src.exists(): raise SystemExit("Run benchmark.py first.")
rows=list(csv.DictReader(src.open(encoding="utf-8")))
out=ROOT/"results"/"quality_review.csv"
fields=["prompt_id","prompt","model_label","output","correctness_1_to_5",
        "completeness_1_to_5","instruction_following_1_to_5","clarity_1_to_5","reviewer_notes"]
with out.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in rows:
        w.writerow({k:r.get(k,"") for k in ["prompt_id","prompt","output"]} |
                   {"model_label":r["label"],"correctness_1_to_5":"","completeness_1_to_5":"",
                    "instruction_following_1_to_5":"","clarity_1_to_5":"","reviewer_notes":""})
print(f"Review sheet created: {out}")
print("Score both settings against the same rubric. Prefer blinded review and note factual errors separately.")
