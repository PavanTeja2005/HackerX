# app.py
import os
import tempfile
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from extractor import download_and_extract_text
from retriever import chunk_and_build_index, retrieve_topk_for_question
from llm_agent import answer_question_with_context

app = FastAPI(title="HackRx API", version="1.0")

# Simple auth: expect Bearer token in Authorization header
API_KEY = "bearer"

class RunRequest(BaseModel):
    documents: str | list
    questions: list[str]

class RunResponse(BaseModel):
    answers: list[str]

@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def hackrx_run(payload: RunRequest, authorization: str = Header(None)):
    # Auth check
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    # 1. Download & extract text
    tmpdir = tempfile.mkdtemp()
    try:
        docs = payload.documents
        if isinstance(docs, str):
            docs = [docs]
        text_chunks = []
        for doc in docs:
            text_chunks.extend(download_and_extract_text(doc, tmpdir))
        if not text_chunks:
            raise HTTPException(status_code=400, detail="Document extraction failed or empty")
        
        # 2. Embed & index
        index, metadata_list = chunk_and_build_index(text_chunks)
        
        # 3. Answer each question
        answers = []
        for q in payload.questions:
            topk = retrieve_topk_for_question(q, index, metadata_list, k=5)
            ans_text = answer_question_with_context(q, topk)
            answers.append(ans_text)
        
        return {"answers": answers}
    finally:
        pass  # Optional cleanup

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)