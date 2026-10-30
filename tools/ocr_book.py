"""OCR a scanned book PDF into the Markdown-with-page-markers text `rag ingest` reads.

Built for scans where every PDF page is a photo of a two-page book spread (and
no text layer exists). For each PDF page it:

  1. renders at 300 dpi and finds the gutter (the least-inked column near the
     middle), splitting the spread into its left and right book pages;
  2. OCRs each half with Apple's Vision framework (macOS only, via `ocrmac`) --
     offline, no model download;
  3. reads the printed page number from the footer ("FORTY-ONE", "vii") with
     fuzzy matching, then repairs misreads by requiring consecutive numbering;
  4. drops chapter thumb tabs, figure captions and footers, re-joins
     hyphenated line breaks, and turns headings into Markdown:
        CHAPTER 4 / EXERCISES          -> # Chapter 4 – Exercises
        EXERCISE 2 / Lying Face Down   -> ## Exercise 2 – Lying Face Down
        OSTEOPOROSIS                   -> ## Osteoporosis
  5. writes pages separated by a form feed and a `<!-- page N -->` marker, so
     citations point at the BOOK's page numbers, not the PDF's.

OCR results are cached per PDF page, so re-running after tweaking the
post-processing takes seconds. The output is a derived file: the manifest
pins its sha256, so this script is what regenerates it.

    pip install -e '.[ocr]'
    python tools/ocr_book.py ~/Downloads/backpain.pdf data/raw/treat-your-own-back.txt \
        --skip-front 3
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- page numbers -------------------------------------------------------------

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
          "seventeen", "eighteen", "nineteen"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_WORDS = {w: i for i, w in enumerate(_UNITS)} | _TENS | {"hundred": 100}
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}


def roman_value(s: str) -> int | None:
    s = s.lower()
    if not s or not re.fullmatch(r"[ivxlc]+", s):
        return None
    total = 0
    for a, b in zip(s, s[1:] + " ", strict=True):
        v = _ROMAN[a]
        total += -v if b != " " and _ROMAN.get(b, 0) > v else v
    return total


def words_value(s: str) -> int | None:
    """'SEVENTI-THAEE' -> 73. Each token is fuzzy-matched to a number word."""
    tokens = [t for t in re.split(r"[\s\-–]+", s.lower()) if t]
    if not tokens or len(tokens) > 3:
        return None
    total = 0
    for t in tokens:
        match = difflib.get_close_matches(t, _WORDS, n=1, cutoff=0.6)
        if not match:
            return None
        v = _WORDS[match[0]]
        total = total * 100 if v == 100 else total + v
    return total or None


@dataclass
class Label:
    text: str            # what was printed, e.g. "FORTY-ONE" or "vii"
    number: int | None   # arabic page number; None for roman front matter
    roman: int | None = None


def parse_label(raw: str) -> Label | None:
    raw = raw.strip()
    if raw.isdigit():
        return Label(raw, int(raw))
    r = roman_value(raw)
    if r is not None and raw.islower():
        return Label(raw, None, r)
    w = words_value(raw)
    return Label(raw, w) if w else None


def repair_numbers(labels: list[Label | None]) -> list[Label | None]:
    """Fix misread or missing arabic page numbers. Gaps are allowed: scans
    often skip blank or full-page-figure pages.

    A reading is *trusted* when a raw neighbour confirms it (n-1 before or
    n+1 after). Then, left to right, every other reading is accepted only if it
    fits between the last assigned page and the next trusted one (leaving room
    for the pages in between); otherwise it becomes last assigned + 1.
    Roman (front matter) pages are left as read.
    """
    nums = [lab.number if lab else None for lab in labels]
    trusted = [False] * len(nums)
    for i, n in enumerate(nums):
        if n is None:
            continue
        prev_ok = i > 0 and nums[i - 1] == n - 1
        next_ok = i + 1 < len(nums) and nums[i + 1] == n + 1
        trusted[i] = prev_ok or next_ok

    out: list[Label | None] = []
    last: tuple[int, int] | None = None  # (index, assigned page)
    for i, lab in enumerate(labels):
        if lab is not None and lab.roman is not None:
            out.append(lab)
            continue
        n = nums[i]
        if not trusted[i]:
            nxt = next((j for j in range(i + 1, len(nums)) if trusted[j]), None)
            lo = last[1] + (i - last[0]) if last else 1
            hi = nums[nxt] - (nxt - i) if nxt is not None else None
            if n is None or n < lo or (hi is not None and n > hi):
                if last is None:
                    out.append(lab)  # nothing to anchor on yet (front matter)
                    continue
                n = last[1] + 1
        out.append(Label(lab.text if lab else "?", n))
        last = (i, n)
    return out


# --- OCR ------------------------------------------------------------------------


@dataclass
class Line:
    text: str
    conf: float
    x: float   # normalised, left edge
    y: float   # normalised, origin at the BOTTOM (Vision convention)
    w: float
    h: float


def render(page, dpi: int) -> np.ndarray:
    import pymupdf

    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)


def find_gutter(img: np.ndarray) -> int:
    w = img.shape[1]
    lo, hi = int(w * 0.38), int(w * 0.62)
    ink = (img[:, lo:hi] < 110).sum(axis=0).astype(float)
    return lo + int(np.argmin(np.convolve(ink, np.ones(25), "same")))


def ocr(img: np.ndarray) -> list[Line]:
    from ocrmac import ocrmac
    from PIL import Image

    res = ocrmac.OCR(Image.fromarray(img), recognition_level="accurate",
                     language_preference=["en-US"]).recognize()
    return [Line(t, float(c), *map(float, box)) for t, c, box in res]


def ocr_pdf(pdf: Path, cache_dir: Path, dpi: int) -> list[list[list[Line]]]:
    """[pdf_page][half] -> lines. Cached per PDF page as JSON."""
    import pymupdf

    cache_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf)
    pages = []
    for i, page in enumerate(doc, start=1):
        cpath = cache_dir / f"{pdf.stem}-p{i:03d}-{dpi}.json"
        if cpath.exists():
            halves = [[Line(**d) for d in half] for half in json.loads(cpath.read_text())]
        else:
            img = render(page, dpi)
            g = find_gutter(img)
            halves = [ocr(img[:, :g]), ocr(img[:, g:])]
            cpath.write_text(json.dumps([[l.__dict__ for l in h] for h in halves]))
        pages.append(halves)
        print(f"\r  OCR {i}/{len(doc)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return pages


# --- cleanup --------------------------------------------------------------------

FOOTER_Y = 0.065            # lines below this are the running footer
_TAB = re.compile(r"^[0-9O]$|^[0-9]{1,2}$")  # chapter thumb tab: a lone digit
_CAPTION = re.compile(r"^Fig\.?\s*\d", re.IGNORECASE)
# Table-of-contents leaders that bleed across the gutter: ".. 35 .. 40", "•44", "- . 62".
_LEADER = re.compile(r"^(?=.*\d)(?=.*[.•·…])[\s.•·…\-\d]+$")
_KEEP_HYPHEN = {"self", "non", "well", "long", "short", "co", "pre", "post", "over", "under",
                "mid", "semi", "anti", "multi", "one", "two", "three", "four", "five", "six",
                "seven", "eight", "nine", "ten", "first", "low", "high", "follow", "lying",
                "x", "re", "night", "pain", "half", "full", "day"}


def _is_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 3 and sum(c.isupper() for c in letters) / len(letters) > 0.9


def split_footer(lines: list[Line]) -> tuple[list[Line], str | None]:
    body, footer = [], None
    for l in lines:
        if l.y < FOOTER_Y:
            footer = l.text
        else:
            body.append(l)
    return body, footer


def reading_order(lines: list[Line]) -> list[Line]:
    """Top to bottom; lines whose vertical centres nearly coincide go left to right."""
    lines = sorted(lines, key=lambda l: -(l.y + l.h / 2))
    out: list[Line] = []
    for l in lines:
        if out and abs((out[-1].y + out[-1].h / 2) - (l.y + l.h / 2)) < 0.006 and l.x < out[-1].x:
            out.insert(len(out) - 1, l)
        else:
            out.append(l)
    return out


def text_margins(lines: list[Line]) -> list[float]:
    """Left edges of the text columns on a page: where long body lines start."""
    xs = sorted(round(l.x, 2) for l in lines if len(l.text) >= 30)
    margins: list[float] = []
    for x in xs:
        if not margins or x - margins[-1] > 0.05:
            margins.append(x)
    return margins


def to_markdown(lines: list[Line], state: dict) -> list[str]:
    """Body lines of one book page -> Markdown lines.

    Heading rules, learned from where this book's layout fooled simpler ones:
      * a heading starts at the left edge of a text column; capitalised text
        floating elsewhere is a diagram label ("VERTEBRAL BODY") and is dropped;
      * consecutive heading lines at the same margin are one heading;
      * a numbered capitalised line ("1. SITTING FOR PROLONGED PERIODS") is a
        subheading -- unless the page has 3+ of them, which makes it a list
        (the Panic Page instructions);
      * "EXERCISE N" pairs with a following subtitle line.
    `state` carries the chapter across pages, so a title split over a page
    break still pairs.
    """
    margins = text_margins(lines) or [state.get("margin", 0.15)]
    if margins:
        state["margin"] = margins[0]

    def at_margin(l: Line) -> bool:
        return any(abs(l.x - m) <= 0.035 for m in margins)

    numbered_caps = [l for l in lines if re.match(r"^\d+[.)]\s", l.text) and _is_caps(l.text)]
    caps_is_list = len(numbered_caps) >= 3

    out: list[str] = []
    in_caps_item = False
    last_heading: tuple[int, Line] | None = None   # (index in out, line) of a mergeable heading
    pending_title: str | None = None

    def emit_text(t: str, l: Line) -> None:
        if out and out[-1] and not out[-1].startswith("#"):
            prev = out[-1]
            if prev.endswith("-") and t[:1].islower():
                stem = re.findall(r"([A-Za-z]+)-$", prev)
                joiner = "-" if stem and stem[0].lower() in _KEEP_HYPHEN else ""
                out[-1] = prev[:-1] + joiner + t
                return
            if l.x > state.get("margin", 1) + 0.025 and at_margin_indent(l):
                out.extend(["", t])  # indented first line: new paragraph
                return
            out[-1] = f"{prev} {t}"
            return
        out.append(t)

    def at_margin_indent(l: Line) -> bool:
        return any(0.015 < l.x - m <= 0.06 for m in margins)

    def heading(level: str, title: str, l: Line, mergeable: bool = True) -> None:
        nonlocal last_heading
        out.extend(["", f"{level} {title}", ""])
        last_heading = (len(out) - 2, l) if mergeable else None

    for l in lines:
        t = l.text.strip()
        if not t or _TAB.match(t) or _CAPTION.match(t) or _LEADER.match(t):
            continue
        if (l.x < 0.1 or l.x > 0.9) and len(t) <= 2:
            continue

        # Photo label ("Exercise 1" beside a picture): short, well right of every
        # text column, not a sentence. Dropped before it can break a list.
        if (len(t.split()) <= 3 and not t.endswith(".") and margins
                and l.x - max(m for m in margins if m <= l.x + 0.035 or m == margins[0]) > 0.2):
            continue

        caps = _is_caps(t)
        # Continuation of the heading just emitted (same margin, next line down).
        if (caps and last_heading and abs(l.x - last_heading[1].x) < 0.03
                and 0 < last_heading[1].y - l.y < 0.045 and len(t) <= 60
                and not t.endswith((".", ","))):
            i, prev_line = last_heading
            out[i] = f"{out[i]} {t.title()}"
            last_heading = (i, l)
            continue
        if not caps or not at_margin(l):
            last_heading = None if not caps else last_heading

        if pending_title is not None:
            is_sub = (not t.endswith((".", ",", ":")) and len(t) < 60 and t[:1].isupper()
                      and not caps and not re.match(r"exercise\s+\d", t, re.IGNORECASE))
            heading("##", f"{pending_title.title()} – {t}" if is_sub else pending_title.title(),
                    l, mergeable=False)
            pending_title = None
            if is_sub:
                continue

        m = re.fullmatch(r"CHAPTER\s+(\d+)", t)
        if m:
            state["chapter_no"] = m.group(1)
            state["await_chapter_title"] = True
            continue
        if state.get("await_chapter_title") and caps:
            heading("#", f"Chapter {state['chapter_no']} – {t.title()}", l)
            state["await_chapter_title"] = False
            continue
        # Case-insensitive with a mostly-capitals check: OCR reads "ExERCISE 4",
        # while the Title-case "Exercise 1" photo labels must not match.
        if (re.fullmatch(r"EXERCISE\s+\d+", t, re.IGNORECASE)
                and sum(c.isupper() for c in t) >= 6 and at_margin(l)):
            pending_title = t
            continue

        starts_item = bool(re.match(r"^\d+[.)]\s", t))
        if caps and starts_item:
            if caps_is_list:
                in_caps_item = True
                out.append(t)
            elif len(t) <= 50 and not t.endswith("."):
                heading("##", t.title(), l)
            else:
                out.append(t)
            continue
        if caps and in_caps_item:
            out[-1] = f"{out[-1]} {t}"
            continue
        in_caps_item = False
        in_column = any(-0.035 <= l.x - m <= 0.07 for m in margins)  # margin or indent
        if (caps and not in_column and len(t.split()) <= 4
                and not t.endswith(".") and not any(c.isdigit() for c in t)):
            continue  # floating diagram label
        if (caps and len(t) <= 70 and not t.endswith((".", ",", ":", ";"))
                and "," not in t and not t[:1].isdigit()):
            heading("##", "Introduction" if t.upper() == "INTRODUCTION" else t.title(), l)
            continue
        last_heading = None
        emit_text(t, l)
    if pending_title is not None:
        out += ["", f"## {pending_title.title()}", ""]
    return out


# --- word-level OCR correction -------------------------------------------------
# Single-letter confusions Vision makes on this kind of print, observed in
# practice: esercises, chronie, deceloped, folloting, mumbness, perforn.
CONFUSIONS = [("s", "x"), ("e", "c"), ("c", "e"), ("c", "v"), ("c", "w"), ("t", "v"),
              ("t", "w"), ("n", "u"), ("u", "n"), ("n", "m"), ("m", "n"), ("i", "l"),
              ("l", "i"), ("o", "a"), ("a", "o"), ("e", "a"), ("t", "f")]
DICT_PATH = Path("/usr/share/dict/words")


def build_vocab(text: str) -> tuple[set[str], set[str], dict[str, int]]:
    """(dictionary words, all accepted words, book word frequencies)."""
    freq: dict[str, int] = {}
    for w in re.findall(r"[A-Za-z]+", text):
        freq[w.lower()] = freq.get(w.lower(), 0) + 1
    words = {w.strip().lower() for w in DICT_PATH.read_text().split()} if DICT_PATH.exists() else set()
    words |= COMMON_FORMS
    # A word the book itself uses 5+ times is vocabulary even if the dictionary
    # lacks it ("McKenzie", "centralisation").
    return words, words | {w for w, n in freq.items() if n >= 5}, freq


#: Irregular forms the old Webster list in /usr/share/dict lacks. Without them a
#: correct "became" looks unknown and gets "fixed" to "become".
COMMON_FORMS = {"became", "began", "begun", "brought", "bought", "caught", "children", "chose",
                "done", "drove", "fell", "felt", "forgot", "found", "gave", "given", "gone",
                "grew", "held", "kept", "knew", "known", "lain", "lay", "led", "left", "lost",
                "made", "meant", "met", "paid", "ran", "said", "sat", "saw", "seen", "slept",
                "sought", "spent", "stood", "taught", "thought", "told", "took", "understood",
                "went", "women", "wore", "worn", "wrote", "written", "feet", "teeth", "lying"}

_SUFFIXES = ("s", "es", "ed", "d", "ing", "ly", "er", "ers", "ness", "al", "ally")


def _known(w: str, vocab: set[str]) -> bool:
    """Loose: used to decide an ORIGINAL word is fine and must be left alone."""
    w = w.lower()
    return any(c in vocab for c in (w, w[:-1], w[:-2], w[:-3], w[:-2] + "e", w[:-3] + "e",
                                     w[:-3] + "y", w[:-4]) if len(c) >= 3)


def _known_strict(w: str, vocab: set[str]) -> bool:
    """Strict: a CANDIDATE fix must be a dictionary word or one plus a real suffix."""
    if w in vocab:
        return True
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            stem = w[: -len(suf)]
            if stem in vocab or stem + "e" in vocab or (stem.endswith("i") and stem[:-1] + "y" in vocab):
                return True
            if len(stem) > 3 and stem[-1] == stem[-2] and stem[:-1] in vocab:  # stopped -> stop
                return True
    return False


def correct_words(text: str) -> tuple[str, dict[str, str]]:
    """Fix unknown words by ONE confusion substitution that yields a known word.

    Gated three ways, so a correct-but-unusual word is left alone rather than
    'fixed' into a different one: the original must be unknown, the candidate
    must be known, and exactly one candidate may survive (ties broken only by
    the book's own usage). Digits are never touched -- the numeric-provenance
    check downstream depends on them being exactly as printed.
    """
    dictionary, vocab, freq = build_vocab(text)
    fixes: dict[str, str] = {}

    def fix(m: re.Match) -> str:
        w = m.group(0)
        if len(w) < 4 or _known(w, dictionary):
            return w
        # Book frequency alone protects a word -- unless a one-letter variant is
        # at least twice as common, which marks a *consistent* OCR misread
        # ("MeKenzie" x6 next to "McKenzie" x16).
        own = freq.get(w.lower(), 0)
        if w in fixes:
            return fixes[w]
        cands = set()
        low = w.lower()
        for i, ch in enumerate(low):
            for a, b in CONFUSIONS:
                if ch == a:
                    c = low[:i] + b + low[i + 1:]
                    if _known_strict(c, vocab):
                        cands.add(c)
        if own >= 5:
            cands = {c for c in cands if freq.get(c, 0) >= 2 * own}
        if len(cands) > 1:
            best = sorted(cands, key=lambda c: -freq.get(c, 0))
            cands = {best[0]} if freq.get(best[0], 0) > freq.get(best[1], 0) else set()
        if len(cands) != 1:
            return w
        c = cands.pop()
        out = "".join(o.upper() if o.isupper() else n for o, n in zip(w, c, strict=True))
        fixes[w] = out
        return out

    return re.sub(r"(?<![\w-])[A-Za-z]+(?![\w])", fix, text), fixes


# --- main ----------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pdf", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--cache", type=Path, default=Path("data/ocr_cache"))
    p.add_argument("--skip-front", type=int, default=0,
                   help="omit the first N book pages (e.g. author bio, table of contents)")
    a = p.parse_args()

    spreads = ocr_pdf(a.pdf, a.cache, a.dpi)
    halves = [h for spread in spreads for h in spread]

    bodies, labels, confs = [], [], []
    for lines in halves:
        body, footer = split_footer(lines)
        bodies.append(reading_order(body))
        labels.append(parse_label(footer) if footer else None)
        confs += [l.conf for l in body]
    raw_labels = list(labels)
    labels = repair_numbers(labels)

    state: dict = {}
    parts: list[str] = []
    report = []
    gaps: list[int] = []         # never seen: absent from the scan
    figure_only: list[int] = []  # seen, but nothing left after dropping captions
    prev_no = 0
    for idx, (body, lab, raw) in enumerate(zip(bodies, labels, raw_labels, strict=True)):
        printed = lab.text if lab else "?"
        if idx < a.skip_front:
            report.append(f"  skip  book page #{idx + 1} (front matter)")
            continue
        page_no = lab.number if lab and lab.number else 0
        md = "\n".join(to_markdown(body, state)).strip()
        md = re.sub(r"\n{3,}", "\n\n", md)
        if page_no and prev_no and page_no > prev_no + 1:
            gaps.extend(range(prev_no + 1, page_no))
        prev_no = page_no or prev_no
        if not md:
            figure_only.append(page_no)
            continue
        fixed = "" if raw is None or (lab and raw.number == lab.number) else "  (repaired)"
        report.append(f"  page {page_no or printed:<5} footer {raw.text if raw else '-'!r}{fixed}")
        parts.append(f"<!-- page {page_no or 'n/a'} -->\n{md}\n")

    text, fixes = correct_words("\f".join(parts))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text, encoding="utf-8")
    if fixes:
        print(f"corrected {len(fixes)} OCR word forms: "
              + ", ".join(f"{k}→{v}" for k, v in sorted(fixes.items())), file=sys.stderr)
    print("\n".join(report), file=sys.stderr)
    low = sum(c < 0.5 for c in confs)
    print(f"\nwrote {a.out}: {len(parts)} book pages, {len(confs)} lines, mean OCR confidence "
          f"{statistics.mean(confs):.3f}, {low} low-confidence lines", file=sys.stderr)
    if figure_only:
        print(f"figure-only pages (no text to index): {', '.join(map(str, figure_only))}",
              file=sys.stderr)
    if gaps:
        print(f"book pages ABSENT from the scan: {', '.join(map(str, gaps))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
