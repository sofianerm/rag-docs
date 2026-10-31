"""
Évaluation du RAG sur un jeu de questions de référence.

  python eval.py                         -> eval/questions.json (local, non versionné) ou, à défaut,
                                            eval/questions.example.json
  python eval.py --questions mon_jeu.json
  python eval.py --min-score 0.8         -> code de sortie 1 si le score est inférieur (pour la CI)

Chaque question peut définir :
  must_contain      expressions régulières qui doivent toutes apparaître dans la réponse
  must_cite         fichiers qui doivent tous figurer dans les sources
  source            {filename, page} : le bon extrait doit être retrouvé (mesure du retrieval)
  expect_no_answer  true : le système doit dire qu'il ne sait pas (test anti-hallucination)

Les résultats détaillés sont écrits dans eval/results/<date>.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import rag_core

NO_ANSWER = re.compile(
    r"ne (trouve|dispose|contien)|pas (d'|cette |l')information|aucune information|n'est pas (mentionn|indiqu|précis)",
    re.IGNORECASE,
)


def evaluate(item: dict) -> dict:
    start = time.perf_counter()
    result = rag_core.answer(item["question"])
    elapsed = time.perf_counter() - start
    answer, sources = result["answer"], result["sources"]
    cited = {s["filename"] for s in sources}
    checks: dict[str, bool] = {}

    if item.get("expect_no_answer"):
        checks["dit qu'il ne sait pas"] = bool(NO_ANSWER.search(answer))
    for pattern in item.get("must_contain", []):
        checks[f"contient /{pattern}/"] = bool(re.search(pattern, answer))
    for filename in item.get("must_cite", []):
        checks[f"cite {filename}"] = filename in cited
    retrieved = None
    if "source" in item:
        src = item["source"]
        retrieved = any(s["filename"] == src["filename"] and src["page"] in (s["pages"] or [])
                        for s in sources)
        checks["bon extrait retrouvé"] = retrieved

    return {
        "id": item["id"],
        "question": item["question"],
        "expected": item.get("expected"),
        "answer": answer,
        "sources": sources,
        "checks": checks,
        "passed": all(checks.values()),
        "retrieved": retrieved,
        "seconds": round(elapsed, 2),
        "tokens": result["usage"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue le RAG sur un jeu de questions de référence.")
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    path = args.questions or next(p for p in (Path("eval/questions.json"), Path("eval/questions.example.json"))
                                  if p.exists())
    items = json.loads(path.read_text(encoding="utf-8"))
    print(f"Jeu de questions : {path} ({len(items)} questions)\n")

    results = []
    for item in items:
        try:
            r = evaluate(item)
        except Exception as exc:
            r = {"id": item["id"], "question": item["question"], "passed": False, "error": True,
                 "checks": {f"erreur technique {type(exc).__name__}: {str(exc)[:120]}": False},
                 "retrieved": None, "seconds": 0, "tokens": None}
        results.append(r)
        mark = "OK    " if r["passed"] else ("ERREUR" if r.get("error") else "ECHEC ")
        print(f"[{mark}] {r['id']:<28} {r['seconds']:>5.1f} s")
        for name, ok in r["checks"].items():
            if not ok:
                print(f"         x {name}")
        if not r["passed"] and r.get("answer"):
            print(f"         réponse : {r['answer'][:200]!r}")

    passed = sum(r["passed"] for r in results)
    retrieval = [r["retrieved"] for r in results if r["retrieved"] is not None]
    tokens = [r["tokens"]["prompt_tokens"] + r["tokens"]["completion_tokens"] for r in results if r.get("tokens")]
    score = passed / len(results)

    errors = sum(1 for r in results if r.get("error"))
    print(f"\nScore : {passed}/{len(results)} ({score:.0%})")
    if errors:
        print(f"Dont {errors} erreur(s) technique(s) (API indisponible, limite de débit...) : à relancer")
    if retrieval:
        print(f"Retrieval (bon extrait dans les sources) : {sum(retrieval)}/{len(retrieval)}")
    print(f"Temps moyen : {sum(r['seconds'] for r in results) / len(results):.1f} s par question")
    if tokens:
        print(f"Tokens moyens : {sum(tokens) // len(tokens)} par question")

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now():%Y-%m-%d_%H%M%S}.json"
    out.write_text(json.dumps({"questions": str(path), "score": score, "results": results},
                              ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Détails : {out}")

    rag_core.get_pool().close()
    if args.min_score is not None and score < args.min_score:
        sys.exit(1)


if __name__ == "__main__":
    main()
