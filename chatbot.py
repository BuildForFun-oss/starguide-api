import os
import json
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel
import pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# ===== CONFIG =====
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "starguide-knowledge-v2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

with open("knowledge_graph.json", "r") as f:
    kg = json.load(f)

SYSTEM_PROMPT = """
You are StarGuide, a wise astrological counselor grounded in Norvell's teachings.
You empower users, never predict doom, and always respect free will.
Use markdown, emojis, and bullet points. Be warm and practical.
"""

INTENT_MAP = {
    "character": "character",
    "compatibility": "romance",
    "attraction": "romance",
    "wealth": "wealth",
    "career": "general",
    "health": "health",
    "parenting": "parenting",
    "social": "social",
    "spiritual": "spiritual",
    "general": "general"
}

def classify_intent(msg: str) -> str:
    msg_lower = msg.lower()
    if any(w in msg_lower for w in ["planet", "ruling", "element"]):
        return "factual"
    for intent, keywords in {
        "character": ["personality", "traits", "character", "nature"],
        "compatibility": ["compatible", "match", "relationship", "love"],
        "attraction": ["attract", "please", "win", "charm"],
        "wealth": ["money", "rich", "wealth", "finance"],
        "career": ["career", "job", "profession", "business"],
        "health": ["health", "body", "disease", "exercise"],
        "parenting": ["child", "parent", "raise", "kid"],
        "social": ["friend", "social", "popularity"],
        "spiritual": ["spirit", "purpose", "ego", "destiny"],
    }.items():
        if any(kw in msg_lower for kw in keywords):
            return intent
    return "general"

def query_knowledge_graph(msg: str) -> Optional[str]:
    msg_lower = msg.lower()
    for sign in kg["signs"]:
        if sign in msg_lower:
            planet = kg["signs"][sign].get("ruling_planet")
            element = kg["signs"][sign].get("element")
            if planet and "planet" in msg_lower:
                return f"The ruling planet of {sign.capitalize()} is {planet.capitalize()}."
            if element and "element" in msg_lower:
                return f"{sign.capitalize()} is a {element} sign."
    return None

def get_embedding(text: str):
    return embedding_model.encode(text).tolist()

def retrieve_chunks(query: str, sign: Optional[str] = None, domain: Optional[str] = None, top_k: int = 10):
    qvec = get_embedding(query)
    filter_dict = {}
    if sign: filter_dict["sign"] = {"$eq": sign.lower()}
    if domain: filter_dict["domain"] = {"$eq": domain}
    res = index.query(vector=qvec, top_k=top_k, filter=filter_dict, include_metadata=True)
    return [{"text": m.metadata.get("text", ""), "metadata": m.metadata, "score": m.score} for m in res.matches]

def rerank_chunks(query: str, chunks: List[Dict], top_n: int = 5):
    if not chunks: return []
    scores = reranker.predict([(query, c["text"]) for c in chunks])
    for i, c in enumerate(chunks):
        c["combined_score"] = 0.6 * c["score"] + 0.4 * scores[i]
    return sorted(chunks, key=lambda x: x["combined_score"], reverse=True)[:top_n]

def hybrid_retrieve(query: str, sign=None, domain=None):
    initial = retrieve_chunks(query, sign, domain)
    return rerank_chunks(query, initial)

app = FastAPI()
security = HTTPBearer()

class ChatRequest(BaseModel):
    message: str
    user_sign: Optional[str] = None
    session_id: Optional[str] = None
    history: Optional[List[Dict]] = None

@app.post("/api/chat", dependencies=[Depends(security)])
async def chat(request: ChatRequest):
    try:
        # KG lookup
        kg_answer = query_knowledge_graph(request.message)
        if kg_answer:
            return {"response": kg_answer, "intent": "factual", "sources": ["kg"], "retrieved_chunks": []}

        intent = classify_intent(request.message)
        domain = INTENT_MAP.get(intent, "general")
        chunks = hybrid_retrieve(request.message, sign=request.user_sign, domain=domain)
        context = "\n\n".join([c["text"] for c in chunks])

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if request.history:
            messages.extend(request.history[-5:])

        user_content = f"User's sign: {request.user_sign or 'unknown'}\n\n"
        if chunks:
            user_content += f"Insights:\n{context}\n\n"
        user_content += f"Question: {request.message}"
        messages.append({"role": "user", "content": user_content})

        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=1000
        )
        answer = resp.choices[0].message.content

        if intent in ["health", "wealth"]:
            answer += "\n\n*Disclaimer: Not professional medical/financial advice.*"

        # Safety
        if "you will die" in answer.lower() or "doomed" in answer.lower():
            answer += "\n\nRemember, the stars incline, they do not compel."

        return {
            "response": answer,
            "intent": intent,
            "sources": [c["metadata"].get("source") for c in chunks],
            "retrieved_chunks": [{"text": c["text"][:200] + "...", "score": c.get("combined_score")} for c in chunks]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}