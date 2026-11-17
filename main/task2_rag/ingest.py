"""Download NIST source PDF, extract pages, and build overlapping page-aware chunks."""
from pathlib import Path
import argparse, json, re, requests, yaml
from pypdf import PdfReader

ROOT=Path(__file__).parent
CFG=yaml.safe_load((ROOT/"config.yaml").read_text())

def clean(text):
    text=re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text or "")
    text=re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_page(text, page, size, overlap):
    words=text.split()
    if not words: return []
    chunks=[]; start=0
    while start<len(words):
        end=min(start+size,len(words))
        piece=" ".join(words[start:end])
        if len(piece.split()) >= 35:
            chunks.append({"page":page,"text":piece,"word_start":start,"word_end":end})
        if end==len(words): break
        start=max(start+1,end-overlap)
    return chunks

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--download",action="store_true")
    ap.add_argument("--chunk-words",type=int,default=CFG["default_chunk_words"])
    args=ap.parse_args()
    pdf=ROOT/CFG["pdf_path"]; pdf.parent.mkdir(parents=True,exist_ok=True)
    if args.download or not pdf.exists():
        print("Downloading official NIST PDF...")
        r=requests.get(CFG["source_url"],timeout=90); r.raise_for_status()
        pdf.write_bytes(r.content)
    reader=PdfReader(str(pdf))
    pages=[]
    for i,p in enumerate(reader.pages, start=1):
        text=clean(p.extract_text() or "")
        if text: pages.append({"page":i,"text":text})
    (ROOT/CFG["pages_path"]).write_text(json.dumps(pages,indent=2))
    chunks=[]
    for p in pages:
        for c in chunk_page(p["text"],p["page"],args.chunk_words,CFG["chunk_overlap_words"]):
            c["chunk_id"]=f"p{p['page']:03d}_c{len(chunks):05d}"
            chunks.append(c)
    out=ROOT/"data"/f"chunks_{args.chunk_words}.json"
    out.write_text(json.dumps(chunks,indent=2))
    print(f"PDF pages: {len(reader.pages)} | text-bearing pages: {len(pages)} | chunks: {len(chunks)}")
    if len(reader.pages)<30: raise RuntimeError("Source unexpectedly has fewer than 30 pages; verify the PDF.")
    print(f"Saved {out}")

if __name__=="__main__": main()
