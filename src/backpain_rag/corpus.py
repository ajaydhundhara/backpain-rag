"""Corpus governance: a sha256-pinned allow-list of source documents.

A file sitting in data/raw/ is not part of the corpus. A file *listed in the
manifest with a matching sha256* is. Dropping an uncurated file into data/raw/
halts ingestion with a named error instead of quietly widening what the
assistant will answer from -- and a source swapped for a different version
fails the same way, so "which document did this answer come from?" always has
an auditable answer.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from .config import get_settings
from .models import SourceDoc

SUPPORTED = {".pdf", ".txt", ".md"}


class CorpusError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _doc_id(filename: str) -> str:
    stem = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "doc"


def _title(filename: str) -> str:
    return re.sub(r"[_\-]+", " ", Path(filename).stem).strip().title()


def raw_files() -> list[Path]:
    raw = get_settings().raw_dir
    if not raw.exists():
        return []
    return sorted(p for p in raw.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)


def load_manifest() -> list[SourceDoc]:
    path = get_settings().manifest_path
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    return [SourceDoc(**d) for d in data.get("documents") or []]


def save_manifest(docs: list[SourceDoc]) -> None:
    path = get_settings().manifest_path
    text = path.read_text() if path.exists() else ""
    existing = yaml.safe_load(text) if text else None
    header = existing if isinstance(existing, dict) else {}
    header["documents"] = [d.model_dump() for d in docs]
    # Keep the leading comment block: yaml.safe_dump would silently drop it.
    comments = "".join(f"{line}\n" for line in text.splitlines() if line.startswith("#"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(comments + yaml.safe_dump(header, sort_keys=False, allow_unicode=True))


def scan(rehash: bool = False) -> tuple[list[SourceDoc], list[str]]:
    """Add every new file in data/raw/ to the manifest; keep existing entries.

    With `rehash`, also re-pin the sha256 of listed files that changed on disk.
    That is deliberately opt-in: a silently re-pinned hash would defeat the
    point of pinning one. Use it after regenerating a derived file (e.g. OCR).

    `url`, `publisher` and `version` are left blank rather than guessed: they
    end up in citations, and blank provenance is visible where invented
    provenance is not. Fill them in by hand.
    """
    docs = load_manifest()
    known = {d.filename for d in docs}
    added: list[str] = []
    if rehash:
        raw = get_settings().raw_dir
        for i, d in enumerate(docs):
            path = raw / d.filename
            if path.exists() and (h := sha256_file(path)) != d.sha256:
                docs[i] = d.model_copy(update={"sha256": h})
                added.append(f"{d.filename} (re-pinned)")
    for path in raw_files():
        if path.name in known:
            continue
        doc_id = _doc_id(path.name)
        taken = {d.doc_id for d in docs}
        base, n = doc_id, 2
        while doc_id in taken:
            doc_id, n = f"{base}-{n}", n + 1
        docs.append(SourceDoc(
            doc_id=doc_id, title=_title(path.name), filename=path.name,
            sha256=sha256_file(path),
        ))
        added.append(path.name)
    save_manifest(docs)
    return docs, added


def verify() -> list[SourceDoc]:
    """Every listed file exists with the pinned hash, and nothing unlisted is present."""
    docs = load_manifest()
    if not docs:
        raise CorpusError(
            "corpus manifest is empty -- put documents in data/raw/ and run `rag corpus scan`"
        )
    raw = get_settings().raw_dir
    problems: list[str] = []
    for d in docs:
        path = raw / d.filename
        if not path.exists():
            problems.append(f"missing: {d.filename}")
        elif sha256_file(path) != d.sha256:
            problems.append(f"sha256 mismatch: {d.filename} (file changed since it was listed)")
    listed = {d.filename for d in docs}
    for path in raw_files():
        if path.name not in listed:
            problems.append(f"unlisted file in data/raw/: {path.name} (run `rag corpus scan`)")
    if problems:
        raise CorpusError("corpus verification failed:\n  " + "\n  ".join(problems))
    return docs
