from fastapi import FastAPI, Request
from pydantic import BaseModel
from openai import OpenAI
import os, time

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 64 simple docs
DOCUMENTS = [{"id": i, "content": f"Machine learning paper {i}", "metadata": {"source": f"p{i}"}} for i in range(64)]

class QueryRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    rerankK: int = 5

@app.post("/search")
async def search(request: QueryRequest):
    start = time.time()
    
    # Simple similarity (no FAISS)
    def simple_score(q, doc):
        return 0.9 if "machine learning" in doc["content"].lower() and "machine learning" in q.lower() else 0.5
    
    candidates = sorted(DOCUMENTS, key=lambda d: simple_score(request.query, d), reverse=True)[:request.k]
    
    # Rerank with OpenAI
    if request.rerank:
        for c in candidates:
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": f"Is '{request.query}' relevant to '{c['content']}'? Answer 0-10:"}], max_tokens=2)
            c["score"] = float(resp.choices[0].message.content.strip()) / 10
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:request.rerankK]
    
    return {
        "results": candidates[:request.rerankK],
        "reranked": request.rerank,
        "metrics": {"latency": int((time.time()-start)*1000), "totalDocs": 64}
    }

@app.get("/")
async def root():
    return {"status": "Semantic Search Ready!", "docs": 64}


