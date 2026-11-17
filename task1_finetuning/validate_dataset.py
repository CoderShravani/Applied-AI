"""Validate generated split integrity, target schema, and absence of ID leakage."""
from pathlib import Path
import json
ROOT=Path(__file__).parent/"data"
required={"action","object","priority","due_in_days","assignee"}
allowed={"action":{"create","update","delete","search"},
         "object":{"task","meeting","document","ticket"},
         "priority":{"low","medium","high"}}
all_ids=set()
for split in ("train","validation","test"):
    path=ROOT/f"{split}.jsonl"
    if not path.exists(): raise SystemExit(f"Missing {path}; run make_dataset.py")
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    ids={r["id"] for r in rows}
    if len(ids)!=len(rows): raise AssertionError(f"Duplicate IDs in {split}")
    if all_ids & ids: raise AssertionError(f"Split leakage detected in {split}")
    all_ids |= ids
    for row in rows:
        target=json.loads(row["output"])
        assert set(target)==required, (split,row["id"],target)
        for key,values in allowed.items(): assert target[key] in values
        assert isinstance(target["due_in_days"],int)
        assert target["assignee"] is None or isinstance(target["assignee"],str)
    print(f"{split}: {len(rows)} examples — schema OK")
print(f"PASS: {len(all_ids)} unique examples across disjoint splits.")
