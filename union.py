# config.py - Centralized configuration
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ")
    BEARER_TOKEN = os.getenv("bearer", "default-token-for-testing")
    
    # Model Settings
    LLM_MODEL = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "all-MiniLM-L12-v2"
    
    # Processing Settings
    CHUNK_SIZE = 1000
    TOP_K_RESULTS = 5
    MAX_ANSWER_LENGTH = 120
    
    # Storage Settings
    CHROMA_PERSIST_DIR = "./chroma_data"
    TEMP_DIR = "./temp_files"
    
    # Memory Settings
    EMBEDDING_BATCH_SIZE = 16  # Reduced for memory efficiency
    MAX_DOCUMENT_SIZE_MB = 100
    CLEANUP_AFTER_REQUESTS = True

# =============================================================================

# embeddings.py - Memory-optimized embedding service
import gc
import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Memory-efficient embedding service with automatic cleanup"""
    
    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self.model_name = Config.EMBEDDING_MODEL
    
    def _load_model(self) -> SentenceTransformer:
        """Load model only when needed"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with memory management"""
        if not texts:
            return np.array([])
        
        try:
            model = self._load_model()
            
            # Process in batches to manage memory
            all_embeddings = []
            batch_size = Config.EMBEDDING_BATCH_SIZE
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = model.encode(
                    batch, 
                    convert_to_numpy=True, 
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                all_embeddings.append(batch_embeddings)
                
                # Force garbage collection after each batch
                if i % (batch_size * 4) == 0:  # Every 4 batches
                    gc.collect()
            
            return np.vstack(all_embeddings) if all_embeddings else np.array([])
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def cleanup(self):
        """Explicitly cleanup model from memory"""
        if self._model is not None:
            logger.info("Cleaning up embedding model")
            del self._model
            self._model = None
            gc.collect()
    
    def __del__(self):
        """Ensure cleanup on destruction"""
        self.cleanup()

# Global instance with cleanup registration
embedding_service = EmbeddingService()

def cleanup_embeddings():
    """Function to cleanup embeddings globally"""
    global embedding_service
    embedding_service.cleanup()

# Register cleanup on exit
import atexit
atexit.register(cleanup_embeddings)

# =============================================================================

# retriever.py - ChromaDB-based retriever
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
import hashlib
import json
from config import Config

logger = logging.getLogger(__name__)

class ChromaRetriever:
    """ChromaDB-based document retriever with memory optimization"""
    
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIR
        self.client = None
        self.current_collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with persistence"""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"ChromaDB initialized at: {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def create_collection(self, collection_name: str) -> chromadb.Collection:
        """Create or get collection with optimized settings"""
        try:
            # Clean collection name (ChromaDB has naming restrictions)
            clean_name = self._clean_collection_name(collection_name)
            
            # Delete existing collection to avoid conflicts
            try:
                self.client.delete_collection(name=clean_name)
                logger.info(f"Deleted existing collection: {clean_name}")
            except:
                pass  # Collection doesn't exist
            
            # Create new collection
            self.current_collection = self.client.create_collection(
                name=clean_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 100,
                    "hnsw:M": 16
                }
            )
            
            logger.info(f"Created collection: {clean_name}")
            return self.current_collection
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise
    
    def _clean_collection_name(self, name: str) -> str:
        """Clean collection name for ChromaDB compatibility"""
        # ChromaDB collection names must be alphanumeric + underscores/hyphens
        clean = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)
        return clean[:50]  # Limit length
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add documents to collection with memory optimization"""
        if not chunks or not self.current_collection:
            logger.warning("No chunks to add or no collection selected")
            return
        
        try:
            # Process in batches to manage memory
            batch_size = 100  # Smaller batches for memory efficiency
            total_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size else 0)
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                
                # Prepare batch data
                documents = [chunk["text"] for chunk in batch]
                metadatas = [
                    {
                        "doc_id": chunk["doc_id"],
                        "page": chunk.get("page"),
                        "chunk_id": chunk["chunk_id"]
                    }
                    for chunk in batch
                ]
                ids = [chunk["chunk_id"] for chunk in batch]
                
                # Add to ChromaDB (it handles embeddings automatically)
                self.current_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                # Force garbage collection every few batches
                if batch_num % 3 == 0:
                    gc.collect()
            
            logger.info(f"Added {len(chunks)} chunks to collection")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def search(self, query_text: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.current_collection:
            logger.warning("No collection available for search")
            return []
        
        top_k = top_k or Config.TOP_K_RESULTS
        
        try:
            # Query ChromaDB
            results = self.current_collection.query(
                query_texts=[query_text],
                n_results=min(top_k, 100),  # Limit results for memory
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['ids'][0]:  # Check if we have results
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        "score": max(0, 1 - results['distances'][0][i]),  # Convert distance to similarity
                        "chunk_id": results['metadatas'][0][i]["chunk_id"],
                        "doc_id": results['metadatas'][0][i]["doc_id"],
                        "page": results['metadatas'][0][i].get("page"),
                        "text": results['documents'][0][i]
                    })
            
            logger.info(f"Found {len(formatted_results)} similar chunks")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def cleanup_collection(self):
        """Clean up current collection"""
        if self.current_collection:
            try:
                collection_name = self.current_collection.name
                self.client.delete_collection(name=collection_name)
                logger.info(f"Deleted collection: {collection_name}")
            except Exception as e:
                logger.error(f"Failed to delete collection: {e}")
            finally:
                self.current_collection = None
    
    def cleanup(self):
        """Full cleanup of retriever"""
        self.cleanup_collection()
        if self.client:
            self.client = None
        gc.collect()

# =============================================================================

# llm_agent.py - Memory-optimized LLM service
from groq import Groq
import logging
from typing import List, Dict, Any
from config import Config

logger = logging.getLogger(__name__)

class LLMService:
    """Memory-optimized LLM service"""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Groq client"""
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        try:
            self.client = Groq(api_key=Config.GROQ_API_KEY)
            logger.info("Groq client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
    
    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer with context, optimized for memory"""
        if not context_chunks:
            return "Not stated in provided document."
        
        try:
            # Build context string efficiently
            context_parts = []
            total_length = 0
            max_context_length = 3000  # Limit context to manage memory
            
            for chunk in context_chunks:
                chunk_text = f"- [{chunk['chunk_id']}] (score={chunk['score']:.3f}) {chunk['text']}"
                if total_length + len(chunk_text) > max_context_length:
                    break
                context_parts.append(chunk_text)
                total_length += len(chunk_text)
            
            context_str = "\n".join(context_parts)
            
            # Create prompt
            prompt = self._create_prompt(question, context_str)
            
            # Generate response
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_ANSWER_LENGTH + 50,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Log chunk IDs for debugging
            chunk_ids = [chunk['chunk_id'] for chunk in context_chunks[:3]]
            logger.info(f"Generated answer using chunks: {chunk_ids}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Fallback to first chunk if available
            if context_chunks:
                return context_chunks[0]["text"][:400] + "..."
            return "Error generating answer"
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create optimized prompt"""
        return f"""You are a precise document-answering assistant.
Use ONLY the provided document clauses to answer the question.
DO NOT add information not found in the clauses.

Question: {question}

Context clauses (top relevant chunks):
{context}

Instructions:
- If the answer is present, extract and combine all relevant facts, conditions, and exclusions.
- Include key eligibility requirements, waiting periods, limits, and exceptions exactly as stated.
- If multiple clauses contain relevant info, merge them into one coherent answer.
- If the answer is not in the clauses, return exactly: "Not stated in provided document."
- Cite all clause_ids used in square brackets, e.g., [doc.pdf_p3_c2].
- Keep the answer concise (<= {Config.MAX_ANSWER_LENGTH} words) but complete.

Return ONLY the answer text (no extra commentary)."""

# Global LLM service instance
llm_service = LLMService()

# =============================================================================

# extractor.py - Memory-optimized document extractor
import httpx
import os
import io
import tempfile
import pdfplumber
import docx
import magic
from bs4 import BeautifulSoup
from typing import List, Dict, Generator
import logging
from config import Config

logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Memory-optimized document extractor"""
    
    def __init__(self):
        self.max_file_size = Config.MAX_DOCUMENT_SIZE_MB * 1024 * 1024  # Convert to bytes
    
    def download_file(self, url: str, output_dir: str) -> str:
        """Download file with memory optimization"""
        try:
            filename = os.path.basename(url.split("?")[0]) or "document"
            local_path = os.path.join(output_dir, filename)
            
            # Stream download to avoid loading entire file in memory
            with httpx.stream("GET", url, timeout=60.0) as response:
                response.raise_for_status()
                
                # Check file size
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_file_size:
                    raise ValueError(f"File too large: {int(content_length) / 1024 / 1024:.1f}MB")
                
                # Stream to file
                with open(local_path, "wb") as f:
                    downloaded = 0
                    for chunk in response.iter_bytes(chunk_size=8192):
                        downloaded += len(chunk)
                        if downloaded > self.max_file_size:
                            raise ValueError(f"File too large: >{Config.MAX_DOCUMENT_SIZE_MB}MB")
                        f.write(chunk)
            
            logger.info(f"Downloaded: {filename} ({downloaded / 1024 / 1024:.1f}MB)")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            raise
    
    def extract_text_streaming(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """Extract text in streaming fashion to manage memory"""
        try:
            mime_type = magic.from_file(file_path, mime=True)
            filename = os.path.basename(file_path)
            
            if mime_type == "application/pdf":
                yield from self._extract_pdf_streaming(file_path, filename)
            elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                yield from self._extract_docx_streaming(file_path, filename)
            elif mime_type.startswith("text/"):
                yield from self._extract_text_streaming(file_path, filename)
            elif mime_type in ["text/html", "application/xhtml+xml"]:
                yield from self._extract_html_streaming(file_path, filename)
            else:
                logger.warning(f"Unsupported file type: {mime_type}")
                return
                
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            raise
    
    def _extract_pdf_streaming(self, file_path: str, filename: str) -> Generator[Dict[str, Any], None, None]:
        """Stream PDF extraction page by page"""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue
                    
                    # Split into paragraphs
                    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                    
                    for chunk_num, paragraph in enumerate(paragraphs):
                        if len(paragraph) < 50:  # Skip very short chunks
                            continue
                            
                        yield {
                            "doc_id": filename,
                            "page": page_num,
                            "chunk_id": f"{filename}_p{page_num}_c{chunk_num}",
                            "text": paragraph
                        }
                    
                    # Force garbage collection every 10 pages
                    if page_num % 10 == 0:
                        gc.collect()
                        
        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {e}")
            raise
    
    def _extract_docx_streaming(self, file_path: str, filename: str) -> Generator[Dict[str, Any], None, None]:
        """Stream DOCX extraction paragraph by paragraph"""
        try:
            doc = docx.Document(file_path)
            
            for chunk_num, paragraph in enumerate(doc.paragraphs):
                text = paragraph.text.strip()
                if len(text) < 50:  # Skip very short paragraphs
                    continue
                
                yield {
                    "doc_id": filename,
                    "page": None,
                    "chunk_id": f"{filename}_c{chunk_num}",
                    "text": text
                }
                
        except Exception as e:
            logger.error(f"DOCX extraction failed for {file_path}: {e}")
            raise
    
    def _extract_text_streaming(self, file_path: str, filename: str) -> Generator[Dict[str, Any], None, None]:
        """Stream text file extraction"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            
            for chunk_num, paragraph in enumerate(paragraphs):
                if len(paragraph) < 50:
                    continue
                
                yield {
                    "doc_id": filename,
                    "page": None,
                    "chunk_id": f"{filename}_c{chunk_num}",
                    "text": paragraph
                }
                
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            raise
    
    def _extract_html_streaming(self, file_path: str, filename: str) -> Generator[Dict[str, Any], None, None]:
        """Stream HTML extraction"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f, "html.parser")
            
            text = soup.get_text(separator="\n")
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            for chunk_num, paragraph in enumerate(paragraphs):
                if len(paragraph) < 50:
                    continue
                
                yield {
                    "doc_id": filename,
                    "page": None,
                    "chunk_id": f"{filename}_c{chunk_num}",
                    "text": paragraph
                }
                
        except Exception as e:
            logger.error(f"HTML extraction failed for {file_path}: {e}")
            raise
    
    def extract_from_url(self, url: str, temp_dir: str) -> List[Dict[str, Any]]:
        """Extract text from URL with full memory management"""
        file_path = None
        try:
            # Download file
            file_path = self.download_file(url, temp_dir)
            
            # Extract text in streaming fashion
            chunks = list(self.extract_text_streaming(file_path))
            
            logger.info(f"Extracted {len(chunks)} chunks from {url}")
            return chunks
            
        finally:
            # Cleanup downloaded file
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {file_path}: {e}")

# Global extractor instance
document_extractor = DocumentExtractor()

# =============================================================================

# app.py - Memory-optimized FastAPI application
import os
import tempfile
import logging
import gc
from contextlib import asynccontextmanager
from typing import List

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
                import shutil
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