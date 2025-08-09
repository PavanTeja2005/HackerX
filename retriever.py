# retriever.py
import faiss
import numpy as np
from embeddings import embed_texts

def chunk_and_build_index(chunks):
    # chunks: list of {doc_id, page, chunk_id, text}
    texts = [c["text"] for c in chunks]
    emb = embed_texts(texts)
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb.astype('float32'))
    return index, chunks  # metadata_list = chunks

def retrieve_topk_for_question(question, index, metadata_list, k=6):
    from embeddings import embed_texts
    q_emb = embed_texts([question])
    D, I = index.search(q_emb.astype('float32'), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata_list):
            continue
        meta = metadata_list[idx]
        results.append({
            "score": float(score),
            "doc_id": meta["doc_id"],
            "page": meta["page"],
            "chunk_id": meta["chunk_id"],
            "text": meta["text"]
        })
    return results
