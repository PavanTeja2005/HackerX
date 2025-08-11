import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ")
    BEARER_TOKEN = os.getenv("bearer", "default-token-for-testing")
    
    # Model Settings
    LLM_MODEL = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
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

