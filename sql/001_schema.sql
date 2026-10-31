-- Schéma de la base documentaire (Postgres + pgvector).
-- Fonctionne à l'identique en local (Docker Compose) et sur Supabase.

create extension if not exists vector;

-- Un document = un fichier source.
create table if not exists documents (
    id            bigserial primary key,
    filename      text        not null unique,
    title         text,
    language      text        not null default 'fr',   -- 'fr', 'de', 'en'
    product_line  text,                                -- gamme de produit, pour filtrer
    sha256        text        not null,                -- détecte les fichiers modifiés
    created_at    timestamptz not null default now()
);

-- Un chunk = un morceau de document, avec son vecteur.
create table if not exists chunks (
    id           bigserial primary key,
    document_id  bigint      not null references documents(id) on delete cascade,
    chunk_index  int         not null,
    kind         text        not null default 'text',  -- 'text' ou 'figure'
    content      text        not null,                  -- texte du morceau (avec ses titres de section)
    context      text,                                  -- phrase de contexte générée par le LLM
    headings     text[],
    pages        int[],
    language     text        not null default 'fr',
    embedding    vector(1024) not null,
    fts          tsvector    not null,                  -- recherche par mots-clés
    unique (document_id, chunk_index)
);

create index if not exists chunks_embedding_idx on chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_fts_idx       on chunks using gin (fts);
create index if not exists chunks_document_idx  on chunks (document_id);

-- Jetons d'accès à l'API et au serveur MCP : un par utilisateur, stocké en hash.
create table if not exists api_tokens (
    id            bigserial primary key,
    user_email    text        not null,
    token_hash    text        not null unique,          -- sha256 du jeton, jamais le jeton en clair
    created_at    timestamptz not null default now(),
    last_used_at  timestamptz,
    revoked_at    timestamptz
);

-- Sécurité Supabase : RLS activé sans aucune policy.
-- Les clés publiques (anon / authenticated) ne peuvent donc rien lire :
-- seul le backend, connecté directement à Postgres, accède aux données.
alter table documents  enable row level security;
alter table chunks     enable row level security;
alter table api_tokens enable row level security;

-- Recherche hybride : sens (vecteurs) + mots-clés (plein texte),
-- fusionnés par Reciprocal Rank Fusion.
create or replace function hybrid_search(
    query_text       text,
    query_embedding  vector(1024),
    match_count      int      default 8,
    fts_config       regconfig default 'french',
    filter_language  text     default null,
    rrf_k            int      default 50
)
returns table (
    id           bigint,
    document_id  bigint,
    content      text,
    context      text,
    headings     text[],
    pages        int[],
    score        double precision
)
language sql stable
as $$
with semantic as (
    select c.id,
           row_number() over (order by c.embedding <=> query_embedding) as rank
    from chunks c
    where filter_language is null or c.language = filter_language
    order by c.embedding <=> query_embedding
    limit match_count * 4
),
keyword as (
    select c.id,
           row_number() over (
               order by ts_rank_cd(c.fts, websearch_to_tsquery(fts_config, query_text)) desc
           ) as rank
    from chunks c
    where c.fts @@ websearch_to_tsquery(fts_config, query_text)
      and (filter_language is null or c.language = filter_language)
    order by ts_rank_cd(c.fts, websearch_to_tsquery(fts_config, query_text)) desc
    limit match_count * 4
)
select c.id, c.document_id, c.content, c.context, c.headings, c.pages,
       coalesce(1.0 / (rrf_k + s.rank), 0) + coalesce(1.0 / (rrf_k + k.rank), 0) as score
from semantic s
full outer join keyword k on s.id = k.id
join chunks c on c.id = coalesce(s.id, k.id)
order by score desc
limit match_count;
$$;
