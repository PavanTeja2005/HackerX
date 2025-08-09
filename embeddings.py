# embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

_MODEL = None

def get_embedding_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL

def embed_texts(texts):
    model = get_embedding_model()
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # normalize for cosine via inner product FAISS
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    emb = emb / norms
    return emb
