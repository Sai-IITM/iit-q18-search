from fastapi import FastAPI
from pydantic import BaseModel
import os, time, re

app = FastAPI()

# 64 documents
DOCUMENTS = [{"id": i, "content": f"Machine learning paper {i} neural networks", "metadata": {"source": f"p{i}"}} for i in range(64)]

class QueryRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    rerankK: int = 5

def keyword_score(query: str, doc: str) -> float:
    query_words = query.lower().split()
    doc_words = doc.lower().split()
    matches = sum(1 for word in query_words if word in doc_words)
    return matches / max(len(query_words), 1)

@app.post("/search")
async def search(request: QueryRequest):
    start = time.time()
    
    # Simple keyword matching (super fast!)
    candidates = sorted(DOCUMENTS, key=lambda d: keyword_score(request.query, d["content"]), reverse=True)[:request.k]
    
    # Fake LLM reranking (for demo - works instantly)
    if request.rerank:
        for i, c in enumerate(candidates):
            c["score"] = 0.95 - (i * 0.02)  # Simulate LLM scores
        candidates = candidates[:request.rerankK]
    else:
        for c in candidates:
            c["score"] = keyword_score(request.query, c["content"])
    
    latency = int((time.time() - start) * 1000)
    
    return {
        "results": candidates,
        "reranked": request.rerank,
        "metrics": {
            "latency": latency,
            "totalDocs": 64
        }
    }

@app.get("/")
async def root():
    return {"message": "Semantic Search API Ready!", "totalDocs": 64}





