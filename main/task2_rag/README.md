# Task 2 — Retrieval-augmented generation from scratch

## Corpus and measured result

The source is the official NIST AI Risk Management Framework 1.0 PDF (48 pages, January 2023): https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf

The evaluation contains 15 questions. Page-aware chunks use 60-word overlap and `sentence-transformers/all-MiniLM-L6-v2` embeddings. Normalized embeddings are ranked by NumPy cosine similarity and combined with a custom BM25 scorer using 0.75 dense / 0.25 lexical weighting. Retrieval uses top-5 passages.

| Chunk words | Recall@5 | Gold-term coverage@5 |
|---:|---:|---:|
| 350 | 60.0% (9/15) | 100% (15/15) |
| 700 | 66.7% (10/15) | 100% (15/15) |

The 700-word configuration produced one additional gold-page hit on this evaluation set. Detailed Q10 and Q13 retrieval diagnostics and targeted query-expansion/reranking improvements are recorded in `results/evaluation_summary.json`.

## Architecture

1. Download and extract page text with `pypdf`.
2. Create overlapping word-window chunks while retaining page numbers.
3. Embed with `sentence-transformers/all-MiniLM-L6-v2`.
4. Normalize vectors; rank by cosine similarity using NumPy.
5. Combine dense similarity and BM25 lexical scores.
6. Optionally rerank a candidate set with a cross-encoder.
7. Generate a grounded answer with a local Ollama model or return evidence-only output.

No LangChain, LlamaIndex, or Haystack is used.

## Run

```bash
python ingest.py --download
python evaluate.py --chunk-sizes 350 700 --top-k 5
python rag.py --question "What does the framework mean by trustworthy AI?"
```
