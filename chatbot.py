import os
import json
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
import pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ===== CONFIG =====
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "starguide-knowledge-v2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Comma-separated list of valid client API keys, e.g. "key1,key2"
VALID_API_KEYS = set(k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip())
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

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
This response is AI-generated.
"""

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "hurt myself", "self harm", "self-harm"
]

CRISIS_RESPONSE = (
    "I'm really glad you reached out, and I want to make sure you get real support right now — "
    "this isn't something the stars can help with, but people can.\n\n"
    "**If you're in the US:** call or text **988** (Suicide & Crisis Lifeline), available 24/7.\n"
    "**Outside the US:** please contact your local emergency number or a crisis line for your country.\n\n"
    "If you're in immediate danger, please contact emergency services now."
)

# Maps classify_intent() output -> Pinecone metadata "domain" field.
# NOTE: previously "career" incorrectly mapped to "general", which meant
# career-domain chunks were never retrieved for career questions.
INTENT_MAP = {
    "character": "character",
    "compatibility": "romance",
    "attraction": "romance",
    "wealth": "wealth",
    "career": "career",
    "health": "health",
    "parenting": "parenting",
    "social": "social",
    "spiritual": "spiritual",
    "factual": "general",
    "general": "general"
}


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> str:
    """Actually validate the bearer token (previous version only extracted it)."""
    if not VALID_API_KEYS:
        raise HTTPException(status_code=500, detail="Server auth not configured")
    if credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return credentials.credentials


def contains_crisis_language(msg: str) -> bool:
    msg_lower = msg.lower()
    return any(kw in msg_lower for kw in CRISIS_KEYWORDS)


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
            if planet and ("planet" in msg_lower or "ruling" in msg_lower):
                return f"The ruling planet of {sign.capitalize()} is {planet.capitalize()}."
            if element and "element" in msg_lower:
                return f"{sign.capitalize()} is a {element.capitalize()} sign."
    return None


def get_embedding(text: str):
    return embedding_model.encode(text).tolist()


def retrieve_chunks(query: str, qvec, sign: Optional[str] = None, domain: Optional[str] = None, top_k: int = 10):
    filter_dict = {}
    if sign:
        filter_dict["sign"] = {"$eq": sign.lower()}
    if domain:
        filter_dict["domain"] = {"$eq": domain}
    res = index.query(vector=qvec, top_k=top_k, filter=filter_dict, include_metadata=True)
    return [{"text": m.metadata.get("text", ""), "metadata": m.metadata, "score": m.score} for m in res.matches]


def _rerank_sync(query: str, chunks: List[Dict], top_n: int = 5):
    if not chunks:
        return []
    scores = reranker.predict([(query, c["text"]) for c in chunks])
    for i, c in enumerate(chunks):
        c["combined_score"] = 0.6 * c["score"] + 0.4 * scores[i]
    return sorted(chunks, key=lambda x: x["combined_score"], reverse=True)[:top_n]


async def hybrid_retrieve(query: str, sign=None, domain=None):
    # Embedding + rerank are CPU-bound and blocking; run off the event loop
    # so one slow request doesn't stall every other concurrent user.
    qvec = await run_in_threadpool(get_embedding, query)
    initial = await run_in_threadpool(retrieve_chunks, query, qvec, sign, domain)
    return await run_in_threadpool(_rerank_sync, query, initial)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves static/index.html (the chat UI) at the site root, so visiting the
# Railway URL in a browser shows the chatbot interface instead of a 404.
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return f.read()
    return HTMLResponse("<h1>StarGuide API</h1><p>No UI found at static/index.html.</p>", status_code=200)


class ChatRequest(BaseModel):
    message: str
    user_sign: Optional[str] = None
    location: Optional[str] = None
    session_id: Optional[str] = None
    history: Optional[List[Dict]] = None


@app.post("/api/chat")
async def chat(request: ChatRequest, _token: str = Depends(verify_token)):
    try:
        if contains_crisis_language(request.message):
            return {
                "response": CRISIS_RESPONSE,
                "intent": "crisis",
                "sources": [],
                "retrieved_chunks": []
            }

        kg_answer = query_knowledge_graph(request.message)
        if kg_answer:
            return {"response": kg_answer, "intent": "factual", "sources": ["kg"], "retrieved_chunks": []}

        intent = classify_intent(request.message)
        domain = INTENT_MAP.get(intent, "general")
        chunks = await hybrid_retrieve(request.message, sign=request.user_sign, domain=domain)
        context = "\n\n".join([c["text"] for c in chunks])

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if request.history:
            messages.extend(request.history[-5:])

        user_content = f"User's sign: {request.user_sign or 'unknown'}\n"
        if request.location:
            user_content += f"User's location: {request.location}\n"
        user_content += "\n"
        if chunks:
            user_content += f"Insights:\n{context}\n\n"
        user_content += f"Question: {request.message}"
        messages.append({"role": "user", "content": user_content})

        resp = await run_in_threadpool(
            lambda: openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=1000
            )
        )
        answer = resp.choices[0].message.content

        if intent in ["health", "wealth"]:
            answer += "\n\n*Disclaimer: Not professional medical/financial advice.*"

        if "you will die" in answer.lower() or "doomed" in answer.lower():
            answer += "\n\nRemember, the stars incline, they do not compel."

        return {
            "response": answer,
            "intent": intent,
            "sources": [c["metadata"].get("source") for c in chunks],
            "retrieved_chunks": [{"text": c["text"][:200] + "...", "score": c.get("combined_score")} for c in chunks]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
