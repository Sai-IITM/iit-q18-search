from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os, time

app = FastAPI()
client = None  # Lazy init

DOCUMENTS = [{"id": i, "content": f"Machine learning paper {i}", "metadata": {"source": f"p{i}"}} for i in range(64)]

class QueryRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    rerankK: int = 5

def get_client():
    global client
    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client

@app.post("/search")
async def search(request: QueryRequest):
    start = time.time()
    
    # Simple keyword similarity
    def score_doc(doc):
        return sum(1 for word in request.query.lower().split() if word in doc["content"].lower()) / len(request.query.split())
    
    candidates = sorted(DOCUMENTS, key=score_doc, reverse=True)[:request.k]
    
    # Rerank with OpenAI
    if request.rerank:
        openai_client = get_client()
        for c in candidates:
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": f"Rate '{request.query}' vs '{c['content']}' (0-10):"}],
                    max_tokens=2
                )
                c["score"] = float(resp.choices[0].message.content) / 10
            except:
                c["score"] = 0.5
        candidates.sort(key=lambda x: x["score"], reverse=True)[:request.rerankK]
    
    latency = int((time.time() - start) * 1000)
    return {
        "results": candidates,
        "reranked": request.rerank,
        "metrics": {"latency": latency, "totalDocs": 64}
    }

@app.get("/")
async def root():
    return {"message": "Semantic Search API Ready!", "totalDocs": 64}




