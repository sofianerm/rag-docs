# RAG documentation technique

Assistant qui répond aux questions techniques à partir de fiches produits PDF (puissances, facteurs de correction, raccordements, montage), en **citant le document et la page** de chaque information, et qui dit clairement quand l'information n'est pas dans la documentation.

Deux portes d'entrée sur la même logique de recherche :
- une **API REST** pour un chatbot ;
- un **serveur MCP** pour les agents (Claude Code, Codex, Cursor…).

```mermaid
flowchart LR
    subgraph Import["Import (batch, une fois par document)"]
        PDF[PDF] --> DL[Docling<br/>texte + tableaux]
        DL --> CH[Découpage<br/>par sections]
        CH --> CX[Contexte<br/>par LLM]
        CX --> EM[Embeddings<br/>Voyage]
    end
    EM --> DB[(PostgreSQL<br/>+ pgvector)]
    subgraph Question["À chaque question"]
        Q[Question] --> QE[Embedding<br/>de la question]
        QE --> HS[Recherche hybride<br/>vecteurs + plein texte]
        HS --> LLM[LLM<br/>réponse sourcée]
    end
    DB --- HS
    API["API /question"] --> Q
    MCP["Serveur MCP"] --> HS
```

## Choix techniques

| Étape | Choix | Pourquoi |
|---|---|---|
| Extraction | **Docling** (modèle de mise en page + TableFormer) | Conserve la structure : titres, sections, tableaux. |
| Découpage | `HybridChunker` de Docling, ~500 tokens | Coupe selon la structure du document plutôt qu'à taille fixe ; chaque morceau garde son titre de section. |
| Tableaux | Sérialisation **Markdown** | Le format par défaut (« en-tête = valeur ») cassait les tableaux horizontaux (voir plus bas). |
| Contexte | **Contextual retrieval** : une phrase générée par LLM situe chaque morceau dans son document | Plusieurs fiches contiennent des tableaux presque identiques ; sans contexte, la recherche les confond. |
| Embeddings | **Voyage `voyage-4`**, 1024 dimensions, `input_type` document / query | Multilingue (FR / DE / EN). Alternative 100 % locale de même dimension : `BAAI/bge-m3`. |
| Stockage | **PostgreSQL + pgvector** (Supabase en production, Docker en local) | Une seule base standard pour les données, les vecteurs et le plein texte, sans dépendance à un service propriétaire. |
| Recherche | **Hybride** : similarité cosinus (index HNSW) + plein texte Postgres, fusion par **Reciprocal Rank Fusion** | Le sens comprend les questions formulées librement ; les mots-clés retrouvent les références exactes (codes, numéros). |
| Comparaisons | Détection des codes produits et **une recherche filtrée par produit** | Sans cela, le produit le plus « proche » de la question prend toutes les places. |
| Génération | LLM via une **API compatible OpenAI** (DeepSeek, Claude, Azure OpenAI…), `temperature=0` | Fournisseur interchangeable par configuration, selon les exigences de localisation des données. |

## Problèmes rencontrés

**Tableaux mal lus.** En inspectant les morceaux avant la vectorisation (`--dry-run`), les tableaux horizontaux — une ligne de pourcentages, une ligne de valeurs, sous un titre fusionné — étaient aplatis : la valeur perdait le lien avec son pourcentage. Passage à une sérialisation Markdown, qui conserve la grille.

**Limites de débit de l'API d'embeddings.** Sur un compte gratuit (10 000 tokens/min), l'import échouait sur les gros documents. Les requêtes sont désormais regroupées par budget de tokens, avec nouvelles tentatives et attente progressive (*retry with backoff*). L'import est idempotent : un fichier inchangé (hash SHA-256) n'est pas retraité.

**Questions comparatives.** « Compare le produit A et le produit B » ne ramenait que le produit A : la recherche plein texte exigeait tous les mots dans un même morceau (ET), et un vecteur unique favorisait un seul produit. Passage en OU pour les mots-clés et recherche séparée par produit détecté.

## Sécurité

- Secrets uniquement en variables d'environnement, jamais dans le dépôt.
- **RLS** activé sans policy : aucune lecture possible avec les clés publiques Supabase ; la fonction de recherche n'est exécutable que par le backend.
- **Un jeton par utilisateur**, stocké en hash SHA-256, révocable individuellement, avec date de dernière utilisation. Même contrôle pour l'API et le serveur MCP.
- Image Docker exécutée avec un utilisateur non root.
- Étape suivante en entreprise : OAuth via le fournisseur d'identité (ex. Microsoft Entra ID), prévu par la spécification MCP.

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # sans GPU
pip install -r requirements-ingest.txt                 # API + Docling pour l'import
cp .env.example .env                                   # puis renseigner les clés

docker compose up -d                                   # Postgres + pgvector local, schéma appliqué

python ingest.py docs/ --dry-run                       # vérifier le découpage sans rien stocker
python ingest.py docs/ --lang fr --log import_log.txt  # import complet

python ask.py                                          # poser des questions en ligne de commande
```

## API

```bash
python create_token.py create prenom.nom@exemple.ch   # jeton affiché une seule fois
uvicorn api:app --reload                              # documentation interactive : /docs
```

| Route | Rôle |
|---|---|
| `POST /question` | réponse rédigée par le LLM, avec les sources citées |
| `POST /search` | extraits bruts avec leurs sources, sans LLM |
| `GET /health` | état du service et nombre de morceaux indexés |
| `/mcp` | serveur MCP (HTTP streamable, sans état) |

Authentification : `Authorization: Bearer <jeton>`.

## Serveur MCP

Outils exposés : `search_documentation` (recherche hybride, extraits sourcés) et `list_documents`.

```bash
claude mcp add --transport http documentation https://<hôte>/mcp \
  --header "Authorization: Bearer <jeton>"
```

L'outil renvoie les extraits, pas une réponse rédigée : c'est le modèle de l'agent qui rédige, ce qui évite de payer deux LLM et lui permet de combiner la documentation avec le reste de sa tâche.

## Évaluation

`eval.py` passe un jeu de questions de référence et vérifie, pour chacune :
- que la réponse contient les valeurs attendues ;
- que le bon extrait (fichier + page) a été retrouvé ;
- que les bons documents sont cités (questions comparatives) ;
- que le système **dit qu'il ne sait pas** quand l'information n'existe pas (test anti-hallucination).

```bash
python eval.py                  # eval/questions.json (local) ou eval/questions.example.json
python eval.py --min-score 0.8  # code de sortie non nul sous le seuil
```

Le jeu réel reste local (`eval/questions.json`, non versionné) ; `eval/questions.example.json` montre le format avec des produits fictifs.

## Tests et CI

Tests d'intégration sur une vraie base Postgres + pgvector, avec embeddings et LLM simulés : authentification de l'API et du MCP, réponse sourcée, recherche d'une référence exacte, comparaison de deux produits, validation des entrées, regroupement par budget de tokens.

```bash
DATABASE_URL=postgresql://rag:rag@localhost:5432/rag pytest
```

GitHub Actions lance ces tests à chaque push (service `pgvector/pgvector`) et vérifie la construction de l'image Docker.

## Déploiement

L'API est déployée sur **Vercel** (FastAPI en fonction serverless, déploiement automatique à chaque push). Seules les dépendances de l'API (`requirements.txt`) sont installées ; Docling reste en local, l'import se faisant en batch, hors du serveur.

En serverless, chaque instance ouvre ses propres connexions : `DATABASE_URL` pointe vers le pooler Supabase en **mode transaction** (port 6543), et les requêtes préparées sont désactivées pour rester compatibles.

Variables d'environnement à définir dans Vercel : `DATABASE_URL`, `VOYAGE_API_KEY`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `PRODUCT_CODE_REGEX`.

Le `Dockerfile` (image légère, utilisateur non root) permet le même déploiement sur un serveur classique ou un service de conteneurs (Azure Container Apps, Cloud Run…).

## Structure

```
ingest.py          import : extraction, découpage, contexte, embeddings, stockage
rag_core.py        recherche hybride et génération (partagé par l'API, le MCP et ask.py)
api.py             API FastAPI + serveur MCP (point d'entrée Vercel)
ask.py             client en ligne de commande
create_token.py    gestion des jetons d'accès
eval.py            évaluation sur jeu de questions de référence
sql/               schéma, index et fonction de recherche hybride
tests/             tests d'intégration
```

## Pistes

- Reranking des extraits (cross-encoder) avant la génération
- Reformulation des questions de suivi pour le mode conversation
- Description des figures par un modèle de vision
- Comparaison mesurée avec l'approche « tous les documents dans le contexte »
