"""Source files -> ordered sections with heading paths -> token-bounded chunks.

Chunk quality is mostly a function of section detection: a chunk labelled
"Management > Non-pharmacological > Exercise" retrieves far better than a naked
paragraph, and it lets a citation point at a place a human can check.

Three decisions worth naming:

1. **Headings are found structurally** in PDFs (font size relative to body
   text, plus bold single-line blocks), not by matching a fixed list of titles,
   so it survives documents whose section names differ.
2. **A chunk never spans two sections.** Slightly smaller chunks are a cheap
   price for never citing the wrong section heading.
3. **Front matter is dropped.** A table of contents is dense with section names
   and page numbers, so it matches many queries lexically while containing
   nothing that answers them.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import tiktoken

from .config import get_settings
from .models import Chunk, SourceDoc

_ENC = tiktoken.get_encoding("cl100k_base")

#: Below this many characters per page, a PDF is images and needs OCR first.
MIN_CHARS_PER_PAGE = 100
#: Shallow heading levels mostly repeat the document title; keep the specific end.
MAX_PATH_DEPTH = 4
#: A section shorter than this merges into its predecessor.
RUNT_SECTION_CHARS = 200

FRONT_MATTER_RE = re.compile(
    r"^\s*(table\s+of\s+)?contents?\s*$|^\s*(acknowledge?ments?|foreword|preface"
    r"|abbreviations?|glossary|list\s+of\s+(tables?|figures?|contributors?|abbreviations?)"
    r"|contributors?|guideline\s+committee|copyright|disclaimer|declarations?\s+of\s+interest"
    r"|references?|further\s+reading|bibliography|appendix\s+\w*)\s*$",
    re.IGNORECASE,
)


class ParseError(RuntimeError):
    pass


@dataclass
class Section:
    path: str
    text: str
    page_start: int
    page_end: int


# --- parsing ----------------------------------------------------------------


@dataclass
class _Line:
    text: str
    page: int
    size: float
    bold: bool
    block_lines: int


def _pdf_lines(path: Path) -> list[_Line]:
    try:
        import pymupdf as fitz
    except ImportError as e:
        raise ParseError(
            "PDF parsing needs pymupdf: pip install -e '.[ingest]'"
        ) from e

    lines: list[_Line] = []
    with fitz.open(path) as doc:
        total_chars = 0
        for pno, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                blines = block.get("lines", [])
                for line in blines:
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = " ".join(s["text"].strip() for s in spans)
                    total_chars += len(text)
                    size = max(s["size"] for s in spans)
                    bold = all(s["flags"] & 16 for s in spans)
                    lines.append(_Line(text, pno, round(size, 1), bold, len(blines)))
        if total_chars < MIN_CHARS_PER_PAGE * max(1, len(doc)):
            raise ParseError(
                f"{path.name}: almost no extractable text ({total_chars} chars over "
                f"{len(doc)} pages). It is probably scanned -- OCR it to .txt first."
            )
    return lines


def _looks_like_heading(line: _Line, body_size: float) -> bool:
    t = line.text.strip()
    if len(t) > 90 or len(t) < 3 or t.endswith((".", ",", ";")):
        return False
    if re.fullmatch(r"[\d\s.\-–]+", t):  # page numbers, rules
        return False
    if line.size >= body_size * 1.12:
        return True
    return line.bold and line.block_lines <= 2


def parse_pdf(path: Path) -> list[Section]:
    lines = _pdf_lines(path)
    sizes = Counter(round(l.size) for l in lines for _ in range(max(1, len(l.text) // 20)))
    body = float(sizes.most_common(1)[0][0]) if sizes else 10.0
    heading_sizes = sorted(
        {round(l.size) for l in lines if _looks_like_heading(l, body)}, reverse=True
    )
    level_of = {sz: i for i, sz in enumerate(heading_sizes)}

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    start = end = 1

    def flush() -> None:
        text = re.sub(r"[ \t]+", " ", " ".join(buf)).strip()
        if text:
            path_ = " > ".join(h for _, h in stack[-MAX_PATH_DEPTH:]) or "Document"
            sections.append(Section(path_, text, start, end))

    for line in lines:
        if _looks_like_heading(line, body):
            flush()
            buf = []
            level = level_of.get(round(line.size), len(heading_sizes))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, line.text.strip()))
            start = end = line.page
        else:
            if not buf:
                start = line.page
            buf.append(line.text)
            end = line.page
    flush()
    return sections


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
#: Explicit printed page number, e.g. from tools/ocr_book.py: <!-- page 41 -->
_PAGE_MARKER = re.compile(r"^\s*<!--\s*page\s+(\S+)\s*-->\s*$", re.IGNORECASE)


def parse_text(path: Path) -> list[Section]:
    """Plain text or Markdown. `#` headings define sections.

    Page evidence, in order of preference:
      * `<!-- page N -->` markers set the page explicitly (a book's printed
        numbers, which need not start at 1 or be contiguous; "n/a" -> 0);
      * otherwise form feeds (\\f) count pages from 1;
      * with neither, pages are 0 ("n/a") rather than invented.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    has_pages = "\f" in raw or bool(re.search(r"<!--\s*page\s", raw, re.IGNORECASE))
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    page = 1
    start = last = 1

    def flush() -> None:
        text = re.sub(r"\s+", " ", " ".join(buf)).strip()
        if text:
            p = " > ".join(h for _, h in stack[-MAX_PATH_DEPTH:]) or "Document"
            sections.append(Section(p, text, start if has_pages else 0,
                                    last if has_pages else 0))

    # split("\n"), not splitlines(): splitlines() treats \f as a line break
    # and swallows it, which would erase every page boundary.
    for line in raw.split("\n"):
        page += line.count("\f")
        line = line.replace("\f", "")
        marker = _PAGE_MARKER.match(line)
        if marker:
            label = marker.group(1)
            page = int(label) if label.isdigit() else 0
            continue
        m = _MD_HEADING.match(line)
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
            start = last = page
        elif line.strip():
            if not "".join(buf).strip():
                start = page
            buf.append(line)
            last = page
        else:
            buf.append(line)
    flush()
    return sections


def parse(path: Path) -> list[Section]:
    sections = parse_pdf(path) if path.suffix.lower() == ".pdf" else parse_text(path)
    return _tidy(sections)


def _tidy(sections: list[Section]) -> list[Section]:
    """Drop front matter, merge runts into their predecessor."""
    out: list[Section] = []
    for s in sections:
        leaf = s.path.split(" > ")[-1]
        if FRONT_MATTER_RE.match(leaf):
            continue
        if out and len(s.text) < RUNT_SECTION_CHARS and out[-1].path.split(" > ")[0] == s.path.split(" > ")[0]:
            prev = out[-1]
            out[-1] = Section(prev.path, f"{prev.text} {s.text}", prev.page_start, s.page_end)
            continue
        out.append(s)
    return out


# --- chunking ---------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?:])\s+(?=[A-Z(•\-\d])")


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]


def pack(sentences: list[str], target: int, overlap: int) -> list[str]:
    """Greedy pack sentences to `target` tokens, carrying `overlap` tokens of tail."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for sent in sentences:
        st = n_tokens(sent)
        if st > target and not cur:
            # An oversized sentence becomes its own chunk, never truncated.
            chunks.append(sent)
            continue
        if cur and cur_tok + st > target:
            chunks.append(" ".join(cur))
            tail: list[str] = []
            tail_tok = 0
            for prev in reversed(cur):
                pt = n_tokens(prev)
                if tail_tok + pt > overlap:
                    break
                tail.insert(0, prev)
                tail_tok += pt
            cur, cur_tok = tail, tail_tok
        cur.append(sent)
        cur_tok += st
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_sections(sections: list[Section], doc: SourceDoc) -> list[Chunk]:
    s = get_settings()
    ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
    out: list[Chunk] = []
    for seq, section in enumerate(sections):
        pieces = pack(_sentences(section.text), s.chunk_target_tokens, s.chunk_overlap_tokens)
        if len(pieces) > 1 and n_tokens(pieces[-1]) < s.chunk_min_tokens:
            tail = pieces.pop()
            pieces[-1] = f"{pieces[-1]} {tail}"
        for i, body in enumerate(pieces):
            nt = n_tokens(body)
            if nt < s.chunk_drop_below_tokens:
                continue
            out.append(Chunk(
                chunk_id=f"{doc.doc_id}::{seq:04d}::{i:02d}",
                doc_id=doc.doc_id, title=doc.title, source_url=doc.url,
                publisher=doc.publisher, section_path=section.path,
                page_start=section.page_start, page_end=section.page_end,
                text=body, n_tokens=nt, doc_version=doc.version,
                index_version=s.index_version, ingested_at=ingested_at,
            ))
    return out


# --- pipeline ---------------------------------------------------------------


def ingest_all(docs: list[SourceDoc], on_doc=None) -> dict[str, int]:
    """Parse and chunk every manifest document into data/chunks/<doc_id>.jsonl."""
    s = get_settings()
    s.chunks_dir.mkdir(parents=True, exist_ok=True)
    for stale in s.chunks_dir.glob("*.jsonl"):
        if stale.stem not in {d.doc_id for d in docs}:
            stale.unlink()
    counts: dict[str, int] = {}
    for doc in docs:
        chunks = chunk_sections(parse(s.raw_dir / doc.filename), doc)
        path = s.chunks_dir / f"{doc.doc_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json() + "\n")
        counts[doc.doc_id] = len(chunks)
        if on_doc:
            on_doc(doc, len(chunks))
    return counts


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(get_settings().chunks_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            chunks.extend(Chunk(**json.loads(line)) for line in f if line.strip())
    return chunks
