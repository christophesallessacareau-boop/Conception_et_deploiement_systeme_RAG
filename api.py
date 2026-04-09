# api.py

from contextlib import asynccontextmanager
import os
import secrets
import time

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_mistralai import ChatMistralAI

from RAG.retrieval   import telecharger_evenements
from RAG.embedding   import generer_embeddings
from RAG.vectorstore import construire_vectorstore_langchain, creer_retriever
from RAG.rag         import construire_chaine_rag


# ── Environnement ──────────────────────────────────────────
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ADMIN_KEY       = os.getenv("ADMIN_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY non défini dans .env")
if not ADMIN_KEY:
    raise ValueError("ADMIN_KEY non défini dans .env")

client = Mistral(api_key=MISTRAL_API_KEY)
llm    = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model="mistral-small-latest",   # ✅ nom corrigé
    temperature=0.2
)


# ── Cache global ───────────────────────────────────────────
vectorstore = None
retriever   = None
rag_chain   = None


# ── Pipeline ───────────────────────────────────────────────
def build_pipeline():
    global vectorstore, retriever, rag_chain

    print("⏳ Reconstruction du pipeline RAG...")

    evenements = telecharger_evenements()
    if not evenements:
        raise ValueError("Aucun événement récupéré")

    resultats = generer_embeddings(evenements, client)
    if not resultats:
        raise ValueError("Aucun embedding généré")

    vectorstore = construire_vectorstore_langchain(resultats, client)
    retriever   = creer_retriever(vectorstore)
    rag_chain   = construire_chaine_rag(retriever, llm)

    print("✅ Pipeline prêt")


# ── Démarrage ──────────────────────────────────────────────
@asynccontextmanager                  # fonction asynchrone
async def lifespan(app: FastAPI):
    try:
        build_pipeline()
    except Exception as e:
        print(" Erreur au démarrage :", e)
    yield                                   # l'app tourne ici


app = FastAPI(
    title="RAG Occitanie API",
    description="Posez vos questions sur les événements en Occitanie",
    lifespan=lifespan
)


# ── Schémas ────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str


# ── Sécurité ───────────────────────────────────────────────
def verify_admin(x_api_key: str = Header(None)):  # ✅ une seule définition
    if not x_api_key or not secrets.compare_digest(x_api_key, ADMIN_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


last_rebuild = 0

def rate_limit_rebuild():
    global last_rebuild
    now = time.time()
    if now - last_rebuild < 60:
        raise HTTPException(status_code=429, detail="Too many rebuilds. Attendez 1 minute.")
    last_rebuild = now


# ── Endpoint /ask ──────────────────────────────────────────
@app.post("/ask")
def ask(request: QuestionRequest):
    """Pose une question au système RAG et reçoit une réponse augmentée."""

    if not request.question or request.question.strip() == "":
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    if rag_chain is None:
        raise HTTPException(status_code=503, detail="Pipeline non initialisé, réessayez dans quelques secondes")

    try:
        response, docs = rag_chain(request.question)
        return {
            "question": request.question,
            "answer":   response,
            "sources": [
                {
                    "titre": d.metadata.get("titre"),
                    "ville": d.metadata.get("ville"),
                    "date":  d.metadata.get("date"),
                }
                for d in docs
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint /admin/rebuild ────────────────────────────────
@app.post("/admin/rebuild")
def rebuild(_: None = Depends(verify_admin)):
    """Reconstruit le pipeline RAG (protégé par clé API)."""
    rate_limit_rebuild()
    try:
        build_pipeline()
        return {"status": "✅ Pipeline reconstruit avec succès"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check ───────────────────────────────────────────
@app.get("/")
def root():
    """Vérifie que l'API est en ligne et que le pipeline est prêt."""
    return {
        "message":  "API RAG Occitanie opérationnelle",
        "pipeline": "prêt" if rag_chain is not None else "non initialisé"
    }