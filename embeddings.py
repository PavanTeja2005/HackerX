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

