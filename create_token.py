"""
Gestion des jetons d'accès à l'API / au serveur MCP.

  python create_token.py create prenom.nom@exemple.ch   -> crée un jeton et l'affiche UNE SEULE fois
  python create_token.py list                            -> liste les jetons (sans les jetons eux-mêmes)
  python create_token.py revoke prenom.nom@exemple.ch    -> révoque tous les jetons de cet utilisateur

Seule l'empreinte SHA-256 du jeton est stockée : une fuite de la base ne donne accès à rien.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets

import psycopg
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.add_argument("email")
    sub.add_parser("list")
    r = sub.add_parser("revoke"); r.add_argument("email")
    args = parser.parse_args()

    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        if args.cmd == "create":
            token = "rag_" + secrets.token_urlsafe(32)
            conn.execute(
                "insert into api_tokens (user_email, token_hash) values (%s, %s)",
                (args.email, hashlib.sha256(token.encode()).hexdigest()),
            )
            print(f"Jeton créé pour {args.email}. Copie-le maintenant, il ne sera plus affiché :\n\n{token}\n")

        elif args.cmd == "list":
            rows = conn.execute(
                """select id, user_email, created_at, last_used_at, revoked_at
                   from api_tokens order by created_at"""
            ).fetchall()
            for row in rows:
                state = "révoqué" if row[4] else "actif"
                last = row[3].strftime("%Y-%m-%d %H:%M") if row[3] else "jamais"
                print(f"#{row[0]:<4} {row[1]:<35} {state:<8} créé {row[2]:%Y-%m-%d}  dernier usage {last}")

        elif args.cmd == "revoke":
            n = conn.execute(
                "update api_tokens set revoked_at = now() where user_email = %s and revoked_at is null",
                (args.email,),
            ).rowcount
            print(f"{n} jeton(s) révoqué(s) pour {args.email}")


if __name__ == "__main__":
    main()
