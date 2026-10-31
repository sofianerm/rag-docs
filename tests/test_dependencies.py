"""Vérifie que pyproject.toml (lu par Vercel) et requirements.txt (Docker) déclarent les mêmes dépendances."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _names(specs):
    return {re.split(r"[\[<>=]", s.strip())[0].lower() for s in specs if s.strip() and not s.startswith("#")}


def test_pyproject_and_requirements_match():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert _names(pyproject) == _names(requirements)
