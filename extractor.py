# extractor.py
import httpx, os, io
import pdfplumber
from typing import List, Dict

def download_pdf(url: str, out_path: str) -> str:
    local_path = os.path.join(out_path, "doc.pdf")
    with httpx.stream("GET", url, timeout=30.0) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return local_path

def extract_text_from_pdf(path: str):
    chunks = []
    with pdfplumber.open(path) as pdf:
        for p, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # naive split by double newlines for paragraphs
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
            for i, para in enumerate(paras):
                chunks.append({
                    "doc_id": os.path.basename(path),
                    "page": p,
                    "chunk_id": f"{os.path.basename(path)}_p{p}_c{i}",
                    "text": para
                })
    return chunks

def download_and_extract_text(url: str, tmpdir: str):
    pdf_path = download_pdf(url, tmpdir)
    return extract_text_from_pdf(pdf_path)
