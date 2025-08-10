# extractor.py
import httpx, os, io
import pdfplumber
import docx
import magic  # python-magic
from bs4 import BeautifulSoup
from typing import List, Dict

def download_file(url: str, out_path: str) -> str:
    filename = os.path.basename(url.split("?")[0])
    local_path = os.path.join(out_path, filename)
    with httpx.stream("GET", url, timeout=30.0) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return local_path

# PDF extraction
def extract_pdf(path: str):
    chunks = []
    with pdfplumber.open(path) as pdf:
        for p, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            paras = [para.strip() for para in text.split("\n\n") if para.strip()]
            for i, para in enumerate(paras):
                chunks.append({
                    "doc_id": os.path.basename(path),
                    "page": p,
                    "chunk_id": f"{os.path.basename(path)}_p{p}_c{i}",
                    "text": para
                })
    return chunks

# DOCX extraction
def extract_docx(path: str):
    chunks = []
    doc = docx.Document(path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for i, para in enumerate(paras):
        chunks.append({
            "doc_id": os.path.basename(path),
            "page": None,
            "chunk_id": f"{os.path.basename(path)}_c{i}",
            "text": para
        })
    return chunks

# TXT extraction
def extract_txt(path: str):
    chunks = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    for i, para in enumerate(paras):
        chunks.append({
            "doc_id": os.path.basename(path),
            "page": None,
            "chunk_id": f"{os.path.basename(path)}_c{i}",
            "text": para
        })
    return chunks

# HTML/Email extraction
def extract_html(path: str):
    chunks = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    text = soup.get_text(separator="\n")
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    for i, para in enumerate(paras):
        chunks.append({
            "doc_id": os.path.basename(path),
            "page": None,
            "chunk_id": f"{os.path.basename(path)}_c{i}",
            "text": para
        })
    return chunks

# Main dispatcher
def extract_text(path: str):
    mime_type = magic.from_file(path, mime=True)
    if mime_type == "application/pdf":
        return extract_pdf(path)
    elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        return extract_docx(path)
    elif mime_type.startswith("text/"):
        return extract_txt(path)
    elif mime_type in ["text/html", "application/xhtml+xml"]:
        return extract_html(path)
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")

def download_and_extract_text(url: str, tmpdir: str):
    file_path = download_file(url, tmpdir)
    return extract_text(file_path)
