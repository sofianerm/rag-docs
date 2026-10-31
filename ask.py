"""
Poser des questions à la documentation depuis le terminal.

  python ask.py                         -> mode conversation (Entrée vide pour quitter)
  python ask.py "Ta question ici"       -> une seule question
"""

from __future__ import annotations

import sys

import rag_core


def show(question: str) -> None:
    print("\nRecherche...\n")
    result = rag_core.answer(question)
    print(result["answer"])
    if result["sources"]:
        print("\nSources :")
        for s in result["sources"]:
            pages = ", ".join(str(p) for p in (s["pages"] or [])) or "?"
            section = f" - {s['section']}" if s["section"] else ""
            print(f"  [{s['ref']}] {s['filename']}, p. {pages}{section}")
    if result["usage"]:
        u = result["usage"]
        print(f"\n({u['prompt_tokens']} tokens envoyés, {u['completion_tokens']} tokens générés)")


def main() -> None:
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
        return

    print("Pose ta question sur la documentation (Entrée vide pour quitter).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        try:
            show(question)
        except Exception as exc:
            print(f"[ERREUR] {type(exc).__name__}: {exc}")
    rag_core.get_pool().close()


if __name__ == "__main__":
    main()
