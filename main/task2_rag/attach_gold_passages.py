"""Attach the exact extracted source-page text to each authored retrieval question."""
from pathlib import Path
import json
ROOT=Path(__file__).parent
pages=json.loads((ROOT/"data"/"pages.json").read_text())
page_map={p["page"]:p["text"] for p in pages}
questions=json.loads((ROOT/"questions.json").read_text())
for q in questions:
    q["gold_source_passages"]=[
        {"page":n,"passage":page_map.get(n,"")}
        for n in q["gold_pages"] if page_map.get(n)
    ]
(ROOT/"questions_with_passages.json").write_text(json.dumps(questions,indent=2,ensure_ascii=False))
print(f"Attached extracted source text for {len(questions)} questions.")
print("Review page references against the PDF's printed page labels before submission.")
