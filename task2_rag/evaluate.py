"""Compare chunk sizes on 15 authored questions using evidence-page Recall@5."""
from pathlib import Path
import argparse, json, yaml, numpy as np
from sentence_transformers import SentenceTransformer
from bm25 import BM25

ROOT=Path(__file__).parent
CFG=yaml.safe_load((ROOT/"config.yaml").read_text())

def load_chunks(size):
    path=ROOT/"data"/f"chunks_{size}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python ingest.py --chunk-words {size}")
    return json.loads(path.read_text())

def evaluate(size, questions, model, top_k, hybrid):
    chunks=load_chunks(size)
    vp=ROOT/"data"/f"embeddings_{size}.npy"
    if vp.exists(): vectors=np.load(vp)
    else:
        vectors=model.encode([c["text"] for c in chunks],normalize_embeddings=True,
                             convert_to_numpy=True,show_progress_bar=True).astype("float32")
        np.save(vp,vectors)
    bm=BM25([c["text"] for c in chunks])
    details=[]; hits=0
    for q in questions:
        qv=model.encode([q["question"]],normalize_embeddings=True,convert_to_numpy=True)[0]
        dense=vectors@qv
        if hybrid:
            lexical=np.asarray(bm.score(q["question"]),dtype="float32")
            if lexical.max()>0: lexical/=lexical.max()
            scores=.75*dense+.25*lexical
        else: scores=dense
        idx=np.argsort(scores)[::-1][:top_k]
        retrieved=[chunks[i] for i in idx]
        page_hit=any(c["page"] in q["gold_pages"] for c in retrieved)
        term_hit=any(any(t.lower() in c["text"].lower() for t in q["gold_terms"]) for c in retrieved)
        hits+=int(page_hit)
        details.append({"id":q["id"],"question":q["question"],"gold_pages":q["gold_pages"],
                        "retrieved_pages":[c["page"] for c in retrieved],
                        "retrieved_chunk_ids":[c["chunk_id"] for c in retrieved],
                        "page_hit":page_hit,"gold_term_hit":term_hit,
                        "top_passages":[c["text"] for c in retrieved]})
    return {"chunk_words":size,"top_k":top_k,"hybrid":hybrid,"n":len(questions),
            "recall_at_5":hits/len(questions),"term_coverage_at_5":sum(d["gold_term_hit"] for d in details)/len(details),
            "details":details}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--chunk-sizes",nargs="+",type=int,default=[350,700])
    ap.add_argument("--top-k",type=int,default=5)
    ap.add_argument("--dense-only",action="store_true")
    a=ap.parse_args()
    questions=json.loads((ROOT/"questions.json").read_text())
    model=SentenceTransformer(CFG["embedding_model"])
    results=[evaluate(s,questions,model,a.top_k,not a.dense_only) for s in a.chunk_sizes]
    out=ROOT/"results"; out.mkdir(exist_ok=True)
    for r in results:
        (out/f"chunk_{r['chunk_words']}_evaluation.json").write_text(json.dumps(r,indent=2))
        print(f"chunk={r['chunk_words']} Recall@{a.top_k}={r['recall_at_5']:.3f}; term coverage={r['term_coverage_at_5']:.3f}")
    print("Interpretation: smaller chunks tend to isolate answer-bearing passages; larger chunks retain context but can dilute similarity.")

if __name__=="__main__": main()
