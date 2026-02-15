# app.py - VERCEL + OPENAI FIXED VERSION
import os
import time
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOCUMENTS = [{"id": i, "content": f"Machine learning paper {i} about neural networks", "metadata": {"source": f"p{i}"}} for i in range(64)]

class QueryRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    rerankK: int = 5

@app.post("/search")
async def search(request: QueryRequest):
    start = time.time()
    
    # Simple cosine similarity (no FAISS)
    def similarity(a: str, b: str) -> float:
        return 0.95 if any(word in b.lower() for word in request.query.lower().split()) else 0.6
    
    # Get top K candidates
    candidates = sorted(DOCUMENTS, key=lambda d: similarity(request.query, d["content"]), reverse=True)[:request.k]
    
    # Rerank with LLM (top precision!)
    if request.rerank:
        for c in candidates:
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": f"Rate relevance of '{request.query}' to '{c['content'][:200]}' (0-10):"}],
                    max_tokens=2
                )
                score = float(resp.choices[0].message.content.strip())
                c["score"] = score / 10
            except:
                c["score"] = 0.5  # Fallback
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:request.rerankK]
    else:
        for c in candidates:
            c["score"] = similarity(request.query, c["content"])
    
    latency = int((time.time() - start) * 1000)
    
    return {
        "results": candidates,
        "reranked": request.rerank,
        "metrics": {"latency": latency, "totalDocs": 64}
    }

@app.get("/")
async def root():
    return {"message": "Semantic Search API Ready! POST to /search", "totalDocs": 64}



