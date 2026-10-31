"""
Import de documents PDF dans la base vectorielle.

Pipeline, pour chaque PDF :
  1. Extraction        : Docling lit le texte, les titres et les tableaux.
  2. Figures (option)  : un modèle de vision décrit chaque image ou graphique.
  3. Découpage         : HybridChunker coupe en suivant la structure (titres, sections, tableaux).
  4. Contexte          : le LLM écrit une phrase qui situe chaque morceau dans le document.
  5. Vectorisation     : Voyage AI calcule le vecteur de chaque morceau.
  6. Stockage          : Postgres + pgvector (tables documents et chunks).

Usage :
  python ingest.py data/fiches/*.pdf --lang fr
  python ingest.py docs/ --lang fr                       (tous les PDF d'un dossier)
  python ingest.py docs/fiche.pdf --product-line PX-NOVA
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psycopg
import voyageai
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector

load_dotenv()

FTS_CONFIG = {"fr": "french", "de": "german", "en": "english"}
EMBED_BATCH = 64
MAX_DOC_CHARS_FOR_CONTEXT = 120_000  # au-delà, on tronque le document envoyé au LLM


@dataclass
class Chunk:
    content: str
    headings: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    kind: str = "text"
    context: str = ""

    def text_for_embedding(self) -> str:
        return f"{self.context}\n\n{self.content}" if self.context else self.content


# --------------------------------------------------------------------------
# 1 à 3. Extraction et découpage
# --------------------------------------------------------------------------

def convert_and_chunk(pdf_path: Path, max_tokens: int, describe_figures: bool):
    """Convertit le PDF avec Docling et renvoie (document Docling, liste de Chunk)."""
    from docling.chunking import HybridChunker
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
    from transformers import AutoTokenizer

    class MarkdownTables(ChunkingSerializerProvider):
        """Sérialise les tableaux en Markdown pour garder la grille lignes / colonnes.
        Le format par défaut ("en-tête = valeur") casse les tableaux horizontaux,
        comme les facteurs de correction où une ligne de % est suivie d'une ligne de valeurs."""
        def get_serializer(self, doc):
            return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())

    options = PdfPipelineOptions()
    options.do_table_structure = True
    # Les prospectus contiennent du vrai texte : l'OCR est inutile et ralentit l'import.
    # À réactiver (--ocr) pour des documents scannés.
    options.do_ocr = os.environ.get("INGEST_OCR") == "1"
    if describe_figures:
        options.generate_picture_images = True
        options.images_scale = 2.0

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    doc = converter.convert(str(pdf_path)).document

    # Le tokenizer sert seulement à mesurer la taille des morceaux.
    # bge-m3 est multilingue : une bonne approximation pour FR et DE.
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-m3"),
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True, serializer_provider=MarkdownTables())

    chunks: list[Chunk] = []
    for ch in chunker.chunk(dl_doc=doc):
        pages = sorted({p.page_no for item in ch.meta.doc_items for p in item.prov})
        chunks.append(
            Chunk(
                # contextualize() ajoute les titres de section au texte du morceau
                content=chunker.contextualize(chunk=ch),
                headings=list(ch.meta.headings or []),
                pages=pages,
            )
        )

    if describe_figures:
        chunks.extend(describe_pictures(doc))

    return doc, chunks


def describe_pictures(doc) -> list[Chunk]:
    """Fait décrire chaque image du document par un modèle de vision."""
    client = OpenAI(base_url=os.environ["VISION_BASE_URL"], api_key=os.environ["VISION_API_KEY"])
    model = os.environ["VISION_MODEL"]
    out: list[Chunk] = []

    for i, pic in enumerate(doc.pictures):
        image = pic.get_image(doc)
        if image is None:
            continue
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        page = pic.prov[0].page_no if pic.prov else None

        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Cette image provient d'une documentation technique. "
                        "Décris-la de façon factuelle et complète : type de figure, "
                        "axes et unités pour un graphique, valeurs lisibles, légendes, "
                        "cotes et repères pour un schéma. Réponds en français, sans introduction."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
        )
        description = (resp.choices[0].message.content or "").strip()
        if description:
            out.append(Chunk(
                content=f"Figure {i + 1} (page {page}) : {description}",
                pages=[page] if page else [],
                kind="figure",
            ))
    return out


# --------------------------------------------------------------------------
# 4. Contexte (contextual retrieval)
# --------------------------------------------------------------------------

CONTEXT_PROMPT = """<document>
{document}
</document>

Voici un extrait de ce document :
<extrait>
{chunk}
</extrait>

Écris une ou deux phrases courtes qui situent cet extrait dans le document
(produit ou gamme concernés, sujet de la section), pour améliorer sa recherche.
Réponds uniquement avec ces phrases, en français."""


def add_context(doc_markdown: str, chunks: list[Chunk]) -> None:
    client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
    model = os.environ["LLM_MODEL"]
    document = doc_markdown[:MAX_DOC_CHARS_FOR_CONTEXT]

    for n, ch in enumerate(chunks, 1):
        # Le document est toujours en tête du prompt : les fournisseurs qui mettent
        # en cache les préfixes (DeepSeek, Claude...) ne le refacturent pas en entier.
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=150,
            messages=[{"role": "user", "content": CONTEXT_PROMPT.format(document=document, chunk=ch.content)}],
        )
        ch.context = (resp.choices[0].message.content or "").strip()
        print(f"    contexte {n}/{len(chunks)}", end="\r")
    print()


# --------------------------------------------------------------------------
# 5. Vectorisation
# --------------------------------------------------------------------------

def token_batches(texts: list[str], max_tokens: int) -> list[list[str]]:
    """Regroupe les textes en paquets sous un budget de tokens (estimation : 1 token ~ 3 caractères).
    Nécessaire pour respecter les limites de débit (ex. 10 000 tokens/min sur un compte Voyage gratuit)."""
    batches: list[list[str]] = [[]]
    used = 0
    for t in texts:
        size = len(t) // 3 + 1
        if batches[-1] and (used + size > max_tokens or len(batches[-1]) >= EMBED_BATCH):
            batches.append([])
            used = 0
        batches[-1].append(t)
        used += size
    return batches


def embed(chunks: list[Chunk]) -> list[np.ndarray]:
    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    model = os.environ.get("EMBEDDING_MODEL", "voyage-4")
    dim = int(os.environ.get("EMBEDDING_DIM", "1024"))
    max_tokens = int(os.environ.get("EMBED_BATCH_TOKENS", "8000"))

    vectors: list[np.ndarray] = []
    batches = token_batches([c.text_for_embedding() for c in chunks], max_tokens)
    for n, batch in enumerate(batches, 1):
        for attempt in range(1, 7):
            try:
                result = vo.embed(
                    batch,
                    model=model,
                    input_type="document",   # les questions utiliseront input_type="query"
                    output_dimension=dim,
                )
                break
            except voyageai.error.RateLimitError:
                wait = 20 * attempt
                print(f"    limite de débit Voyage, nouvel essai dans {wait} s ({attempt}/6)")
                time.sleep(wait)
        else:
            raise RuntimeError("Voyage : limite de débit toujours atteinte après 6 essais")
        vectors.extend(np.array(v, dtype=np.float32) for v in result.embeddings)
        print(f"    vectorisation {n}/{len(batches)}", end="\r")
    print()
    return vectors


# --------------------------------------------------------------------------
# 6. Stockage
# --------------------------------------------------------------------------

def store(conn, pdf_path: Path, sha: str, lang: str, product_line: str | None,
          title: str | None, chunks: list[Chunk], vectors: list[np.ndarray]) -> None:
    fts_config = FTS_CONFIG.get(lang, "simple")
    with conn.transaction():
        # Réimport : on supprime l'ancienne version (les chunks suivent en cascade).
        conn.execute("delete from documents where filename = %s", (pdf_path.name,))
        doc_id = conn.execute(
            """insert into documents (filename, title, language, product_line, sha256)
               values (%s, %s, %s, %s, %s) returning id""",
            (pdf_path.name, title, lang, product_line, sha),
        ).fetchone()[0]

        for idx, (ch, vec) in enumerate(zip(chunks, vectors)):
            conn.execute(
                """insert into chunks
                     (document_id, chunk_index, kind, content, context, headings, pages,
                      language, embedding, fts)
                   values (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                           to_tsvector(%s::regconfig, %s))""",
                (doc_id, idx, ch.kind, ch.content, ch.context, ch.headings, ch.pages,
                 lang, vec, fts_config, ch.text_for_embedding()),
            )


class Tee:
    """Écrit à la fois dans la console (progression en direct) et dans le journal."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def detect_product_line(filename: str) -> str | None:
    """Déduit la gamme du nom de fichier, ex. 'Fiche PX-NOVA V2 fr.pdf' -> 'PX-NOVA'."""
    pattern = os.environ.get("PRODUCT_CODE_REGEX", r"\b[A-Z]{1,3}-[A-Z]{2,5}\b")
    match = re.search(pattern, filename)
    return match.group(0) if match else None


def expand_paths(paths: list[Path]) -> list[Path]:
    """Accepte des fichiers et des dossiers (tous les PDF du dossier sont pris)."""
    out: list[Path] = []
    for p in paths:
        out.extend(sorted(p.glob("*.pdf")) if p.is_dir() else [p])
    return out


def already_imported(conn, filename: str, sha: str) -> bool:
    row = conn.execute("select sha256 from documents where filename = %s", (filename,)).fetchone()
    return row is not None and row[0] == sha


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Importe des PDF dans la base vectorielle.")
    parser.add_argument("pdfs", nargs="+", type=Path, help="fichiers PDF ou dossiers")
    parser.add_argument("--lang", default="fr", choices=sorted(FTS_CONFIG))
    parser.add_argument("--product-line", default=None)
    parser.add_argument("--force", action="store_true", help="réimporter même si le fichier n'a pas changé")
    parser.add_argument("--no-context", action="store_true", help="sauter l'étape de contexte (tests rapides)")
    parser.add_argument("--dry-run", action="store_true",
                        help="extraire et découper seulement, afficher les morceaux sans rien stocker")
    parser.add_argument("--ocr", action="store_true", help="activer l'OCR (documents scannés)")
    parser.add_argument("--log", type=Path, default=None,
                        help="écrire aussi la sortie dans ce fichier (en plus de l'affichage)")
    args = parser.parse_args()
    if args.log:
        log_file = open(args.log, "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)
    if args.ocr:
        os.environ["INGEST_OCR"] = "1"

    max_tokens = int(os.environ.get("CHUNK_MAX_TOKENS", "500"))
    describe_figures = bool(os.environ.get("VISION_MODEL"))

    conn = None
    if not args.dry_run:
        conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
        register_vector(conn)

    pdfs = expand_paths(args.pdfs)
    print(f"{len(pdfs)} fichier(s) à traiter")
    failures: list[str] = []

    for i, pdf_path in enumerate(pdfs, 1):
        if not pdf_path.is_file():
            print(f"[ignoré] {pdf_path} introuvable", file=sys.stderr)
            continue
        sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        if conn and not args.force and already_imported(conn, pdf_path.name, sha):
            print(f"[{i}/{len(pdfs)}] à jour : {pdf_path.name}")
            continue

        print(f"[{i}/{len(pdfs)}] import : {pdf_path.name}")
        try:
            doc, chunks = convert_and_chunk(pdf_path, max_tokens, describe_figures)
            print(f"    {len(chunks)} morceaux")

            if args.dry_run:
                for i, ch in enumerate(chunks):
                    print(f"\n--- morceau {i} | pages {ch.pages} | {ch.kind} ---\n{ch.content}")
                continue

            if not args.no_context:
                add_context(doc.export_to_markdown(), chunks)
            vectors = embed(chunks)
            product_line = args.product_line or detect_product_line(pdf_path.name)
            store(conn, pdf_path, sha, args.lang, product_line, pdf_path.stem, chunks, vectors)
            print(f"    stocké ({len(chunks)} morceaux, gamme {product_line or '-'})")
        except Exception as exc:  # un fichier en erreur ne bloque pas les autres
            print(f"    [ERREUR] {type(exc).__name__}: {exc}", file=sys.stderr)
            failures.append(pdf_path.name)

    if conn:
        conn.close()
    if failures:
        print(f"\n{len(failures)} fichier(s) en erreur : {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)
    print("\nImport terminé.")


if __name__ == "__main__":
    main()
