"""
API du RAG : deux portes d'entrée sur la même logique de recherche (rag_core).

  POST /question  question -> réponse rédigée + sources   (chatbot)
  POST /search    question -> extraits bruts + sources     (intégrations)
  GET  /health    état du service
  /mcp            serveur MCP (HTTP streamable) pour les agents : Claude Code, Codex, ...

Authentification : en-tête "Authorization: Bearer <jeton>" sur toutes les routes sauf /health.
Les jetons sont créés avec create_token.py et stockés en hash dans la table api_tokens.

Lancement en local :
  uvicorn api:app --reload
  puis http://localhost:8000/docs pour tester depuis le navigateur.
"""

from __future__ import annotations

import hashlib
import logging
import time
from contextlib import asynccontextmanager

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

import rag_core

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag-api")


# --------------------------------------------------------------------------
# Authentification (partagée par l'API et le serveur MCP)
# --------------------------------------------------------------------------

def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def user_for_token(token: str) -> str | None:
    """Renvoie l'e-mail associé à un jeton actif, ou None. Met à jour la date de dernière utilisation."""
    with rag_core.get_pool().connection() as conn:
        row = conn.execute(
            """update api_tokens set last_used_at = now()
               where token_hash = %s and revoked_at is null
               returning user_email""",
            (hash_token(token),),
        ).fetchone()
    return row["user_email"] if row else None


bearer = HTTPBearer(auto_error=False)


def require_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> str:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Jeton manquant",
                            headers={"WWW-Authenticate": "Bearer"})
    user = user_for_token(creds.credentials)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Jeton invalide ou révoqué",
                            headers={"WWW-Authenticate": "Bearer"})
    return user


# --------------------------------------------------------------------------
# Serveur MCP
# --------------------------------------------------------------------------

mcp = MCPServer(
    name="documentation-technique",
    instructions=(
        "Recherche dans la documentation technique de plafonds rayonnants actifs "
        "(fiches produits, prospectus, catalogues). Utilise search_documentation pour toute "
        "question sur les puissances, facteurs de correction, dimensions, raccordements, "
        "montage ou références produits, et cite le fichier et la page de chaque information."
    ),
)


@mcp.tool(
    description=(
        "Recherche hybride (sens + mots-clés) dans la documentation technique des plafonds "
        "rayonnants actifs. À utiliser pour toute question sur les puissances frigorifiques ou "
        "de chauffage, les facteurs de correction, les dimensions, les raccordements, le montage "
        "ou les références produits. Renvoie les extraits les plus pertinents avec leur fichier "
        "source et leur page. Si la question compare plusieurs produits, cite leurs codes "
        "(ex. 'C-XXX') pour que chaque produit soit cherché."
    )
)
async def search_documentation(question: str, top_k: int = 8) -> str:
    """question : la question en langage naturel, autonome (sans référence à la conversation).
    top_k : nombre d'extraits à renvoyer (1 à 20)."""
    top_k = max(1, min(int(top_k), 20))
    passages = await anyio.to_thread.run_sync(lambda: rag_core.search(question, top_k=top_k))
    if not passages:
        return "Aucun extrait trouvé dans la documentation pour cette question."
    return (
        rag_core.format_passages(passages)
        + "\n\nCite tes sources sous la forme (fichier, page). "
          "Si les extraits ne contiennent pas la réponse, dis-le."
    )


@mcp.tool(description="Liste les documents indexés, avec leur gamme de produit et leur nombre d'extraits.")
async def list_documents() -> str:
    docs = await anyio.to_thread.run_sync(rag_core.list_documents)
    return "\n".join(
        f"- {d['filename']} (gamme : {d['product_line'] or 'générale'}, {d['chunks']} extraits)"
        for d in docs
    )


# Application ASGI du MCP, en mode sans état (stateless) : chaque requête est indépendante,
# ce qui permet de faire tourner plusieurs instances derrière un load balancer.
# La protection DNS-rebinding du SDK est réservée aux serveurs sur localhost ; ici l'accès
# est contrôlé par le jeton Bearer vérifié dans le middleware ci-dessous.
mcp_app = mcp.streamable_http_app(streamable_http_path="/mcp", stateless_http=True,
                                  json_response=True, host="0.0.0.0")


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    async with mcp.session_manager.run():   # le gestionnaire de sessions MCP doit tourner avec l'app
        yield
    rag_core.get_pool().close()             # ferme proprement les connexions à l'arrêt


app = FastAPI(
    title="RAG documentation technique",
    description="Recherche hybride et réponses sourcées sur la documentation produits.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def auth_mcp_and_log(request: Request, call_next):
    start = time.perf_counter()
    if request.url.path.startswith("/mcp"):
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else ""
        user = await run_in_threadpool(user_for_token, token) if token else None
        if user is None:
            log.info("%s %s -> 401", request.method, request.url.path)
            return JSONResponse({"detail": "Jeton manquant, invalide ou révoqué"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
    response = await call_next(request)
    log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path,
             response.status_code, (time.perf_counter() - start) * 1000)
    return response


# --------------------------------------------------------------------------
# Modèles
# --------------------------------------------------------------------------

class QuestionIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000,
                          examples=["Quelle est la puissance frigorifique normée du plafond C-XXX ?"])
    language: str | None = Field(None, pattern="^(fr|de|en)$", description="Filtrer sur une langue (optionnel)")
    top_k: int = Field(8, ge=1, le=20, description="Nombre d'extraits à récupérer")


class Source(BaseModel):
    ref: int
    filename: str
    pages: list[int] | None
    section: str
    score: float


class AnswerOut(BaseModel):
    answer: str
    sources: list[Source]
    usage: dict | None = None


class Passage(BaseModel):
    ref: int
    filename: str
    pages: list[int] | None
    section: str
    context: str | None
    content: str
    score: float


class SearchOut(BaseModel):
    passages: list[Passage]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    with rag_core.get_pool().connection() as conn:
        n = conn.execute("select count(*) as n from chunks").fetchone()["n"]
    return {"status": "ok", "chunks": n}


@app.post("/question", response_model=AnswerOut)
def ask(body: QuestionIn, user: str = Depends(require_user)) -> dict:
    log.info("question de %s : %s", user, body.question[:120])
    return rag_core.answer(body.question, top_k=body.top_k, language=body.language)


@app.post("/search", response_model=SearchOut)
def search(body: QuestionIn, user: str = Depends(require_user)) -> dict:
    passages = rag_core.search(body.question, top_k=body.top_k, language=body.language)
    return {
        "passages": [
            {
                "ref": n,
                "filename": p["filename"],
                "pages": p["pages"],
                "section": " > ".join(p["headings"] or []),
                "context": p["context"],
                "content": p["content"],
                "score": round(float(p["score"]), 4),
            }
            for n, p in enumerate(passages, 1)
        ]
    }


# Le serveur MCP est monté en dernier : les routes FastAPI ci-dessus restent prioritaires.
app.mount("/", mcp_app)
