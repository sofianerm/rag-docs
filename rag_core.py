"""
Cœur du RAG : recherche dans la base et rédaction de la réponse.

Partagé par l'API (chatbot) et, plus tard, par le serveur MCP (agents) :
une seule logique de recherche, deux portes d'entrée.
"""

from __future__ import annotations

import logging
import os
import re
import time
from functools import lru_cache

import numpy as np
import voyageai
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

load_dotenv()
log = logging.getLogger("rag-core")

FTS_CONFIG = {"fr": "french", "de": "german", "en": "english"}

SYSTEM_PROMPT = """Tu es l'assistant technique d'un fabricant de plafonds rayonnants actifs \
(rafraîchissants et chauffants). Tu réponds aux questions des collaborateurs, des bureaux \
d'études et des installateurs.

Règles :
- Réponds UNIQUEMENT à partir des extraits fournis. N'utilise pas tes connaissances générales \
pour donner des valeurs techniques.
- Si les extraits ne contiennent pas la réponse, dis-le clairement ("Je ne trouve pas cette \
information dans la documentation") et n'invente rien.
- Cite tes sources après chaque information, au format [n] où n est le numéro de l'extrait.
- Donne toujours les valeurs avec leurs unités et, si c'est indiqué, la norme ou les conditions \
(ex. "85 W/m² selon DIN 4715").
- Réponds dans la langue de la question, de façon concise et structurée."""


# --------------------------------------------------------------------------
# Connexions (créées une seule fois, réutilisées entre les requêtes)
# --------------------------------------------------------------------------

@lru_cache
def get_pool() -> ConnectionPool:
    return ConnectionPool(
        os.environ["DATABASE_URL"],
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True, "row_factory": dict_row},
        configure=register_vector,
        open=True,
    )


@lru_cache
def get_voyage() -> voyageai.Client:
    return voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])


@lru_cache
def get_llm() -> OpenAI:
    return OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])


# --------------------------------------------------------------------------
# Recherche
# --------------------------------------------------------------------------

EMBED_RETRIES = int(os.environ.get("EMBED_RETRIES", "4"))


def embed_query(question: str) -> np.ndarray:
    """Vectorise la question, avec nouvelles tentatives si l'API limite le débit (HTTP 429)."""
    for attempt in range(1, EMBED_RETRIES + 1):
        try:
            result = get_voyage().embed(
                [question],
                model=os.environ.get("EMBEDDING_MODEL", "voyage-4"),
                input_type="query",   # les documents ont été vectorisés avec input_type="document"
                output_dimension=int(os.environ.get("EMBEDDING_DIM", "1024")),
            )
            return np.array(result.embeddings[0], dtype=np.float32)
        except voyageai.error.RateLimitError:
            if attempt == EMBED_RETRIES:
                raise
            wait = 10 * attempt
            log.warning("limite de débit Voyage, nouvel essai dans %s s (%s/%s)", wait, attempt, EMBED_RETRIES)
            time.sleep(wait)
    raise RuntimeError("inaccessible")


# Format des codes produits, ex. "PX-NOVA". À adapter au catalogue via PRODUCT_CODE_REGEX.
PRODUCT_CODE = re.compile(os.environ.get("PRODUCT_CODE_REGEX", r"\b[A-Z]{1,3}-[A-Z]{2,5}\b"))


def detect_products(question: str) -> list[str]:
    """Codes produits cités dans la question, dans l'ordre, sans doublon (ex. ['PX-NOVA', 'PX-LUNA'])."""
    return list(dict.fromkeys(PRODUCT_CODE.findall(question.upper())))


def _hybrid(conn, question: str, vector: np.ndarray, top_k: int, fts_config: str,
            language: str | None, product: str | None) -> list[dict]:
    return conn.execute(
        """
        select h.id, h.content, h.context, h.headings, h.pages, h.score,
               d.filename, d.title, d.product_line
        from hybrid_search(%s, %s, %s, %s::regconfig, %s, 50, %s) h
        join documents d on d.id = h.document_id
        order by h.score desc
        """,
        (question, vector, top_k, fts_config, language, product),
    ).fetchall()


def search(question: str, top_k: int = 8, language: str | None = None) -> list[dict]:
    """Recherche hybride (sens + mots-clés). Renvoie les extraits avec leur source.

    Si la question cite plusieurs produits (comparaison), on fait une recherche par produit,
    filtrée sur sa fiche, et on répartit les places : sinon le produit le plus « proche »
    de la question prend tous les résultats et l'autre n'est jamais trouvé."""
    vector = embed_query(question)
    fts_config = FTS_CONFIG.get(language or "fr", "simple")
    products = detect_products(question)

    with get_pool().connection() as conn:
        if len(products) < 2:
            return _hybrid(conn, question, vector, top_k, fts_config, language, None)

        per_product = max(4, top_k // len(products))
        results: dict[int, dict] = {}
        for product in products:
            rows = _hybrid(conn, question, vector, per_product, fts_config, language, product)
            if not rows:
                # pas de fiche dédiée à ce produit : on cherche dans toute la base avec son nom
                rows = _hybrid(conn, f"{product} {question}", vector, per_product, fts_config, language, None)
            for row in rows:
                results.setdefault(row["id"], row)
        return list(results.values())


def list_documents() -> list[dict]:
    with get_pool().connection() as conn:
        return conn.execute(
            """select d.filename, d.product_line, d.language, count(c.id) as chunks
               from documents d left join chunks c on c.document_id = d.id
               group by d.id order by d.filename"""
        ).fetchall()


def format_passages(passages: list[dict]) -> str:
    blocks = []
    for n, p in enumerate(passages, 1):
        pages = ", ".join(str(x) for x in (p["pages"] or [])) or "?"
        header = f"[{n}] {p['filename']}, page(s) {pages}"
        if p["context"]:
            header += f"\nContexte : {p['context']}"
        blocks.append(f"{header}\n{p['content']}")
    return "\n\n---\n\n".join(blocks)


def sources_of(passages: list[dict]) -> list[dict]:
    return [
        {
            "ref": n,
            "filename": p["filename"],
            "pages": p["pages"],
            "section": " > ".join(p["headings"] or []),
            "score": round(float(p["score"]), 4),
        }
        for n, p in enumerate(passages, 1)
    ]


# --------------------------------------------------------------------------
# Réponse
# --------------------------------------------------------------------------

def answer(question: str, top_k: int = 8, language: str | None = None) -> dict:
    passages = search(question, top_k=top_k, language=language)
    if not passages:
        return {
            "answer": "Je ne trouve aucune information sur ce sujet dans la documentation.",
            "sources": [],
            "usage": None,
        }

    response = get_llm().chat.completions.create(
        model=os.environ["LLM_MODEL"],
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extraits de la documentation :\n\n{format_passages(passages)}"
                                        f"\n\nQuestion : {question}"},
        ],
    )
    usage = response.usage
    return {
        "answer": (response.choices[0].message.content or "").strip(),
        "sources": sources_of(passages),
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        } if usage else None,
    }
