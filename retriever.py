import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
import hashlib
import json
from config import Config
import gc
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

