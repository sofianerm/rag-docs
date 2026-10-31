-- Recherche hybride v2
--  * mots-clés en logique OU : une question qui cite deux produits ("compare le PX-NOVA et le PX-LUNA")
--    ne trouvait rien en ET, car aucun morceau ne contient les deux noms. Le classement
--    ts_rank_cd favorise toujours les morceaux qui contiennent le plus de termes.
--  * filtre optionnel strict par gamme de produit (utilisé pour les comparaisons).

drop function if exists hybrid_search(text, vector, int, regconfig, text, int);

create or replace function hybrid_search(
    query_text       text,
    query_embedding  vector(1024),
    match_count      int       default 8,
    fts_config       regconfig default 'french',
    filter_language  text      default null,
    rrf_k            int       default 50,
    filter_product   text      default null
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
set search_path = public, extensions
as $$
with q as (
    select replace(plainto_tsquery(fts_config, query_text)::text, '&', '|')::tsquery as tsq
),
candidates as (
    select c.*
    from chunks c
    join documents d on d.id = c.document_id
    where (filter_language is null or c.language = filter_language)
      and (filter_product is null or d.product_line = filter_product)
),
semantic as (
    select c.id, row_number() over (order by c.embedding <=> query_embedding) as rank
    from candidates c
    order by c.embedding <=> query_embedding
    limit match_count * 4
),
keyword as (
    select c.id, row_number() over (order by ts_rank_cd(c.fts, q.tsq) desc) as rank
    from candidates c, q
    where c.fts @@ q.tsq
    order by ts_rank_cd(c.fts, q.tsq) desc
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

revoke execute on function hybrid_search(text, vector, int, regconfig, text, int, text) from public;
-- Sur Supabase : retirer aussi l'accès aux rôles des clés publiques
do $$ begin
    if exists (select 1 from pg_roles where rolname = 'anon') then
        execute 'revoke execute on function hybrid_search(text, vector, int, regconfig, text, int, text) from anon, authenticated';
    end if;
end $$;
