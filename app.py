# app.py
import os
import tempfile
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel
from extractor import download_and_extract_text
from retriever import chunk_and_build_index, retrieve_topk_for_question
from llm_agent import answer_question_with_context

app = FastAPI(title="HackRx /hackrx/run - Prototype")

# Simple auth: expect Bearer token in Authorization header
API_KEY = os.getenv("HACKRX_API_KEY", "testkey123")

class RunRequest(BaseModel):
    documents: str
    questions: list

class RunResponse(BaseModel):
    answers: list

@app.post("/hackrx/run", response_model=RunResponse)
async def hackrx_run(payload: RunRequest, authorization: str = Header(None)):
    # Auth
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")
    # 1. download & extract text
    tmpdir = tempfile.mkdtemp()
    try:
        text_chunks = download_and_extract_text(payload.documents, tmpdir)
        if not text_chunks:
            raise HTTPException(status_code=400, detail="Document extraction failed or empty")
        # 2. embed & index
        index, metadata_list = chunk_and_build_index(text_chunks)
        answers = []
        for q in payload.questions:
            topk = retrieve_topk_for_question(q, index, metadata_list, k=6)
            # 3. call LLM to generate final answer using topk contexts
            ans_text = answer_question_with_context(q, topk)
            answers.append(ans_text)
        return {"answers": answers}
    finally:
        # optional: cleanup tmpdir
        pass

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
