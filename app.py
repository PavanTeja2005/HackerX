import os
import tempfile
import logging
import gc
from contextlib import asynccontextmanager
from typing import List, Any

import shutil
import psutil
import threading
import time


import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

from config import Config
from retriever import ChromaRetriever
from llm_agent import llm_service
from extractor import document_extractor
from embeddings import cleanup_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_memory_usage(interval=10):
    """Log memory usage of the current process at regular intervals."""
    process = psutil.Process(os.getpid())
    with open("memory_usage.txt", "w") as log_file:
        pass
        
    while True:
        mem_info = process.memory_info()
        logger.info(f"Memory usage: RSS={mem_info.rss / (1024 ** 2):.2f} MB, VMS={mem_info.vms / (1024 ** 2):.2f} MB")
        time.sleep(interval)
        with open("memory_usage.txt", "a") as log_file:
            log_file.write(f"{time.time()}: RSS={mem_info.rss / (1024 ** 2):.2f} MB, VMS={mem_info.vms / (1024 ** 2):.2f} MB\n")

# Start memory logging in a background thread
# threading.Thread(target=log_memory_usage, args=(10,), daemon=True).start()

# Request/Response models
class RunRequest(BaseModel):
    documents: str | List[str]
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# Global retriever instance
retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with proper cleanup"""
    global retriever
    
    # Startup
    logger.info("Starting HackRx API server...")
    retriever = ChromaRetriever()
    
    yield
    
    # Shutdown
    logger.info("Shutting down HackRx API server...")
    if retriever:
        retriever.cleanup()
    cleanup_embeddings()
    gc.collect()

# Create FastAPI app with lifespan management
app = FastAPI(
    title="HackRx API - Memory Optimized", 
    version="2.0",
    lifespan=lifespan
)

def verify_token(authorization: str = Header(None)) -> str:
    """Verify Bearer token"""
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    
    token = authorization.split(" ", 1)[1]
    if token != Config.BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    return token

@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def hackrx_run(
    payload: RunRequest, 
    token: str = Depends(verify_token)
):
    """Main API endpoint with comprehensive memory management"""
    collection_name = None
    temp_dir = None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="hackrx_")
        logger.info(f"Processing request with {len(payload.questions)} questions")
        
        # 1. Process documents
        documents = payload.documents if isinstance(payload.documents, list) else [payload.documents]
        all_chunks = []
        
        for doc_url in documents:
            logger.info(f"Processing document: {doc_url}")
            try:
                chunks = document_extractor.extract_from_url(doc_url, temp_dir)
                all_chunks.extend(chunks)
                
                # Force garbage collection after each document
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to process document {doc_url}: {e}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Failed to process document: {str(e)}"
                )
        
        if not all_chunks:
            raise HTTPException(status_code=400, detail="No content extracted from documents")
        
        logger.info(f"Extracted {len(all_chunks)} total chunks")
        
        # 2. Create collection and add documents
        collection_name = f"session_{abs(hash(str(documents)))}"
        retriever.create_collection(collection_name)
        retriever.add_documents(all_chunks)
        
        # Clear chunks from memory
        del all_chunks
        gc.collect()
        
        # 3. Process questions
        answers = []
        for i, question in enumerate(payload.questions):
            logger.info(f"Processing question {i+1}/{len(payload.questions)}: {question[:100]}...")
            
            try:
                # Search for relevant chunks
                similar_chunks = retriever.search(question, top_k=Config.TOP_K_RESULTS)
                similar_chunks = similar_chunks[:Config.TOP_K_RESULTS]
                # with open("chunnks.txt", "w") as f:
                #     for chunk in similar_chunks:
                #         f.write(f"{chunk['text']}\n"+"***"*100+"\n")
                
                # Generate answer
                if similar_chunks:
                    answer = llm_service.generate_answer(question, similar_chunks)
                else:
                    answer = "Not stated in provided document."
                
                answers.append(answer)
                
                # Periodic garbage collection
                if i % 3 == 0:
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to process question: {e}")
                answers.append("Error processing question")
        
        logger.info(f"Successfully processed all {len(answers)} questions")
        return {"answers": answers}
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
    finally:
        # Comprehensive cleanup
        try:
            # Cleanup retriever collection
            if retriever and collection_name:
                retriever.cleanup_collection()
            
            # Cleanup temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "HackRx API - Memory Optimized", "version": "2.0"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # Disable reload in production for better memory management
        log_level="info"
    )