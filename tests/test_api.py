"""
Tests d'intégration : base Postgres + pgvector réelle, embeddings et LLM simulés.

Nécessite une base vide avec le schéma (sql/*.sql) et DATABASE_URL, par exemple :
  docker compose up -d
  DATABASE_URL=postgresql://rag:rag@localhost:5432/rag pytest
La CI GitHub Actions fait tourner ces tests à chaque push.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psycopg
import pytest
from pgvector.psycopg import register_vector

os.environ.setdefault("VOYAGE_API_KEY", "test")
os.environ.setdefault("LLM_BASE_URL", "http://llm.invalid")
os.environ.setdefault("LLM_API_KEY", "test")
os.environ.setdefault("LLM_MODEL", "test")

import ingest  # noqa: E402
import rag_core  # noqa: E402

TOKEN = "rag_test_token"
BASE = np.random.default_rng(0).random(1024).astype(np.float32)


class FakeLLM:
    calls: list[dict] = []

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                FakeLLM.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Réponse simulée [1]"))],
                    usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5),
                )


@pytest.fixture(scope="session", autouse=True)
def database():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    register_vector(conn)
    conn.execute("truncate documents, chunks, api_tokens restart identity cascade")
    # Deux produits : PX-NOVA est volontairement plus « proche » de toutes les questions.
    for product, offset in [("PX-NOVA", 0.01), ("PX-LUNA", 0.5)]:
        chunks = [
            ingest.Chunk(content=f"{product} puissance frigorifique {80 + i} W/m2", pages=[i + 1],
                         headings=["Puissance"], context=f"Fiche du plafond {product}")
            for i in range(6)
        ]
        chunks.append(ingest.Chunk(content=f"{product} flexible coudé référence 0{2430 + offset * 10:.0f}",
                                   pages=[9], headings=["Raccords"]))
        ingest.store(conn, Path(f"Fiche {product} fr.pdf"), "sha", "fr", product, product,
                     chunks, [BASE + offset] * len(chunks))
    conn.execute("insert into api_tokens (user_email, token_hash) values (%s, %s)",
                 ("test@exemple.ch", hashlib.sha256(TOKEN.encode()).hexdigest()))
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def fake_providers(monkeypatch):
    monkeypatch.setattr(rag_core, "embed_query", lambda q: BASE)
    monkeypatch.setattr(rag_core, "get_llm", lambda: FakeLLM)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    import api
    with TestClient(api.app) as c:
        yield c


AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_reimport_is_idempotent(database):
    assert ingest.already_imported(database, "Fiche PX-NOVA fr.pdf", "sha")
    assert database.execute("select count(*) from documents").fetchone()[0] == 2


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "chunks": 14}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer faux"}])
def test_routes_require_valid_token(client, headers):
    for path in ("/question", "/search"):
        assert client.post(path, json={"question": "test test"}, headers=headers).status_code == 401
    assert client.post("/mcp", json={}, headers=headers).status_code == 401


def test_question_returns_answer_with_sources(client):
    r = client.post("/question", json={"question": "Puissance du PX-NOVA ?"}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Réponse simulée [1]"
    assert body["sources"][0]["filename"] == "Fiche PX-NOVA fr.pdf"
    prompt = FakeLLM.calls[-1]["messages"][1]["content"]
    assert "[1] Fiche PX-NOVA fr.pdf" in prompt and "Contexte :" in prompt


def test_keyword_search_finds_exact_reference(client):
    # Le vecteur ne sait pas trouver un numéro : c'est la partie mots-clés qui le remonte.
    r = client.post("/search", json={"question": "référence 02435"}, headers=AUTH)
    top3 = [p["content"] for p in r.json()["passages"][:3]]
    assert any(c.endswith("02435") for c in top3)


def test_comparison_retrieves_both_products(client):
    r = client.post("/search", json={"question": "compare le PX-NOVA et le PX-LUNA"}, headers=AUTH)
    files = [p["filename"] for p in r.json()["passages"]]
    assert files.count("Fiche PX-NOVA fr.pdf") == 4
    assert files.count("Fiche PX-LUNA fr.pdf") == 4


def test_input_validation(client):
    assert client.post("/question", json={"question": "a"}, headers=AUTH).status_code == 422


def test_token_batches_respect_budget():
    batches = ingest.token_batches(["x" * 3000] * 40, 8000)
    assert all(sum(len(t) // 3 + 1 for t in b) <= 8000 for b in batches)
    assert sum(len(b) for b in batches) == 40
