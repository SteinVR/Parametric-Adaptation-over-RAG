"""
Data loading utilities for corpus and goldset.
"""

import json
from pathlib import Path
from typing import Any


def load_goldset(path: Path) -> list[dict[str, Any]]:
    """Load benchmark goldset and return list of reference dicts."""
    with open(path) as f:
        data = json.load(f)
    return data["references"]


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load questions list."""
    with open(path) as f:
        return json.load(f)


def load_json(path: Path) -> Any:
    """Generic JSON loader."""
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    """Save data as JSON with indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def list_corpus_pdfs(corpus_dir: Path) -> list[Path]:
    """Return sorted list of PDF files in corpus directory."""
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    return pdfs
