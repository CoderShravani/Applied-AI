"""Minimal transparent RAG: dense retrieval, optional BM25, optional cross-encoder, grounded output."""
from pathlib import Path
import argparse, json, yaml, numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from bm25 import BM25

ROOT=Path(__file__).parent
CFG=yaml.safe_load((ROOT/"config.yaml").read_text())

def load_index(chunk_words):
    chunks=json.loads((ROOT/"data"/f"chunks_{chunk_words}.json").read_text())
    vec_path=ROOT/"data"/f"embeddings_{chunk_words}.npy"
    if not vec_path.exists():
        model=SentenceTransformer(CFG["embedding_model"])
        vec=model.encode([c["text"] for c in chunks],normalize_embeddings=True,
                         convert_to_numpy=True,show_progress_bar=True)
        np.save(vec_path,vec.astype("float32"))
    else: vec=np.load(vec_path)
    return chunks,vec

def retrieve(question, chunks, vectors, top_k=5, hybrid=True, rerank=False):
    embedder=SentenceTransformer(CFG["embedding_model"])
    q=embedder.encode([question],normalize_embeddings=True,convert_to_numpy=True)[0]
    dense=vectors @ q
    scores=dense.copy()
    if hybrid:
        bm=BM25([c["text"] for c in chunks]).score(question)
        bm=np.asarray(bm,dtype="float32")
        if bm.max()>0: bm=bm/(bm.max()+1e-9)
        scores=0.75*dense+0.25*bm
    candidate_ids=np.argsort(scores)[::-1][:max(top_k*4,top_k)]
    candidates=[chunks[i] for i in candidate_ids]
    if rerank:
        cross=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pair_scores=cross.predict([(question,c["text"]) for c in candidates])
        order=np.argsort(pair_scores)[::-1][:top_k]
        return [dict(candidates[i],score=float(pair_scores[i]),dense_score=float(dense[candidate_ids[i]])) for i in order]
    return [dict(chunks[i],score=float(scores[i]),dense_score=float(dense[i])) for i in candidate_ids[:top_k]]

def answer_with_ollama(question, hits, model="qwen2.5:1.5b"):
    import requests
    evidence="\n\n".join(f"[page {h['page']} | {h['chunk_id']}] {h['text']}" for h in hits)
    prompt=("Answer only from the evidence below. If the evidence does not contain the answer, "
            "say you cannot find it in the supplied document. Cite every factual claim using [page N].\n\n"
            f"EVIDENCE:\n{evidence}\n\nQUESTION: {question}")
    r=requests.post("http://localhost:11434/api/generate",
                    json={"model":model,"prompt":prompt,"stream":False,
                          "options":{"temperature":0.1}},timeout=180)
    r.raise_for_status()
    return r.json()["response"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--question",required=True)
    ap.add_argument("--chunk-words",type=int,default=CFG["default_chunk_words"])
    ap.add_argument("--top-k",type=int,default=CFG["top_k"])
    ap.add_argument("--dense-only",action="store_true")
    ap.add_argument("--rerank",action="store_true")
    ap.add_argument("--generate",action="store_true")
    ap.add_argument("--ollama-model",default="qwen2.5:1.5b")
    a=ap.parse_args()
    chunks,v=load_index(a.chunk_words)
    hits=retrieve(a.question,chunks,v,a.top_k,not a.dense_only,a.rerank)
    print("\nRETRIEVED EVIDENCE")
    for h in hits: print(f"\n[{h['page']} | {h['chunk_id']}] score={h['score']:.4f}\n{h['text'][:1000]}")
    if a.generate:
        print("\nGROUNDED ANSWER\n"+answer_with_ollama(a.question,hits,a.ollama_model))

if __name__=="__main__": main()
