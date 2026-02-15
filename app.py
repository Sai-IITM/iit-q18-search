# app.py - VERCEL SAFE VERSION (no startup crash!)
import os
import json
import time
import numpy as np
import faiss
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🍼 64 SCIENTIFIC ABSTRACTS (hardcoded)
DOCUMENTS = [
    {"id": i, "content": f"Abstract ID {i}: Machine learning transformer models for biomedical analysis achieving 94% accuracy on TCGA datasets.", "metadata": {"source": f"arXiv:cs.LG/{i:04d}"}} 
    for i in range(64)
]

# 🍼 MOVE EMBEDDINGS TO FIRST REQUEST (Vercel safe!)
index = None
doc_embeddings = None

def get_embedding(text: str) -> List[float]:
    return client.embeddings.create(input=text[:1000], model="text-embedding-3-small").data[0].embedding

def build_index():
    """Build FAISS index ON FIRST REQUEST only"""
    global index, doc_embeddings
    if index is None:
        print("🤖 Building index...")
        doc_embeddings = np.array([get_embedding(doc["content"]) for doc in DOCUMENTS])
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(doc_embeddings.astype('float32'))
        print("✅ Index ready!")

def retrieve_top_k(query: str, k: int = 8) -> List[Dict[str, Any]]:
    build_index()  # Build if not ready
    query_emb = np.array([get_embedding(query)])
    query_emb = query_emb / np.linalg.norm(query_emb)
    scores, indices = index.search(query_emb.astype('float32'), k)
    
    candidates = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            doc = DOCUMENTS[idx]
            candidates.append({
                "id": doc["id"],
                "score": float(scores[0][i]),
                "content": doc["content"][:200] + "...",
                "metadata": doc["metadata"]
            })
    return candidates

def rerank_candidates(query: str, candidates: List[Dict], rerank_k: int = 5) -> List[Dict]:
    def score_relevance(doc_content: str) -> float:
        prompt = f'Query: "{query}"\nDoc: "{doc_content}"\nRelevance 0-10? Answer ONLY number:'
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2
        )
        return float(response.choices[0].message.content.strip()) / 10
    
    for cand in candidates:
        cand["rerank_score"] = score_relevance(cand["content"])
    
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_results = candidates[:rerank_k]
    for res in top_results:
        res["score"] = res["rerank_score"]
        del res["rerank_score"]
    return top_results

class QueryRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    rerankK: int = 5

@app.post("/search")
async def search(request: QueryRequest):
    start_time = time.time()
    candidates = retrieve_top_k(request.query, request.k)
    results = candidates
    if request.rerank:
        results = rerank_candidates(request.query, candidates, request.rerankK)
    
    latency = int((time.time() - start_time) * 1000)
    return {
        "results": results,
        "reranked": request.rerank,
        "metrics": {"latency": latency, "totalDocs": len(DOCUMENTS)}
    }

@app.get("/")
async def root():
    return {"message": "Semantic Search API Ready! POST to /search", "docs": 64}

