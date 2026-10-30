"""The OCR tool's pure logic: page-number parsing/repair and Markdown cleanup.
(The Vision OCR call itself is macOS-only and not exercised here.)"""

import importlib.util
import sys
from pathlib import Path

from backpain_rag.ingest import parse_text

_spec = importlib.util.spec_from_file_location(
    "ocr_book", Path(__file__).resolve().parents[1] / "tools" / "ocr_book.py")
ocr_book = importlib.util.module_from_spec(_spec)
sys.modules["ocr_book"] = ocr_book  # dataclasses look their module up here
_spec.loader.exec_module(ocr_book)
Label, Line = ocr_book.Label, ocr_book.Line


def test_page_labels_survive_ocr_misreads():
    assert ocr_book.parse_label("FORTY-ONE").number == 41
    assert ocr_book.parse_label("SEVENTI-TWO").number == 72
    assert ocr_book.parse_label("SEVENTY-THAEE").number == 73
    assert ocr_book.parse_label("FORTI").number == 40
    assert ocr_book.parse_label("vii").roman == 7
    assert ocr_book.parse_label("ONE").number == 1


def test_repair_uses_consecutive_neighbours():
    labs = [Label("x", 40), Label("y", 41), None, Label("z", 99), Label("w", 44), Label("v", 45)]
    fixed = ocr_book.repair_numbers(labs)
    assert [l.number for l in fixed] == [40, 41, 42, 43, 44, 45]


def test_repair_keeps_correct_reads_next_to_scan_gaps():
    # Real sequence from the book: 33, "TWINTY-FOUR" (34), 35, [36-37 not scanned], 38, 39
    labs = [Label("a", 32), Label("b", 33), Label("c", 24), Label("d", 35),
            Label("e", 38), Label("f", 39)]
    assert [l.number for l in ocr_book.repair_numbers(labs)] == [32, 33, 34, 35, 38, 39]


def _line(text, y, x=0.15):
    return Line(text, 1.0, x, y, 0.6, 0.02)


def test_markdown_headings_tabs_captions_and_hyphens():
    lines = [
        _line("4", 0.80, x=0.07),                 # thumb tab
        _line("EXERCISE 2", 0.93),
        _line("Lying Face Down in Extension", 0.90),
        _line("Remain face down. Place your elbows under your", 0.87),
        _line("shoulders and relax com-", 0.85),
        _line("pletely for two minutes.", 0.83),
        _line("Fig. 4.2a", 0.40),
        _line("OSTEOPOROSIS", 0.30),
        _line("From middle age many women are affected.", 0.28),
    ]
    md = "\n".join(ocr_book.to_markdown(lines, {}))
    assert "## Exercise 2 – Lying Face Down in Extension" in md
    assert "relax completely for two minutes." in md
    assert "Fig." not in md and "\n4\n" not in md
    assert "## Osteoporosis" in md


def test_caps_instruction_list_is_not_a_heading():
    # The Panic Page: several numbered capitalised instructions on one page.
    lines = [_line("1. IMMEDIATELY LIE FACE DOWN", 0.80), _line("If this is impossible, rest.", 0.78),
             _line("2. USE A ROLLED TOWEL OR A NIGHT ROLL", 0.74), _line("AROUND YOUR WAIST.", 0.72),
             _line("3. PERFORM EXERCISES 1, 2 AND 3", 0.68)]
    md = "\n".join(ocr_book.to_markdown(lines, {}))
    assert "#" not in md
    assert "2. USE A ROLLED TOWEL OR A NIGHT ROLL AROUND YOUR WAIST." in md


def test_panic_page_keeps_indented_continuations_across_photo_labels():
    # Real layout: items at x=0.16, continuations indented to 0.20, photo labels at 0.54.
    lines = [_line("1. IMMEDIATELY LIE FACE DOWN", 0.667, x=0.16),
             _line("4. IF THE PAIN IS MORE TO", 0.469, x=0.16),
             _line("AWAY FROM THE", 0.416, x=0.20),
             _line("Exercise 1", 0.411, x=0.54),
             _line("PAINFUL SIDE AND DO", 0.395, x=0.20),
             _line("EXERCISES 2 AND 3.", 0.374, x=0.20),
             _line("6. DO NOT BEND FORWARD", 0.265, x=0.16),
             _line("Exercise 2", 0.253, x=0.54),
             _line("FOR 3-4 DAYS.", 0.243, x=0.20)]
    md = "\n".join(ocr_book.to_markdown(lines, {}))
    assert "4. IF THE PAIN IS MORE TO AWAY FROM THE PAINFUL SIDE AND DO EXERCISES 2 AND 3." in md
    assert "6. DO NOT BEND FORWARD FOR 3-4 DAYS." in md
    assert "Exercise 1" not in md and "Exercise 2" not in md


def test_single_numbered_caps_line_is_a_subheading():
    lines = [_line("Prolonged sitting is the most common postural cause of pain.", 0.80),
             _line("1. SITTING FOR PROLONGED PERIODS", 0.75),
             _line("Most people who sit for long periods slouch after a while.", 0.72)]
    assert "## 1. Sitting For Prolonged Periods" in "\n".join(ocr_book.to_markdown(lines, {}))


def test_floating_diagram_labels_are_dropped_and_split_headings_join():
    lines = [_line("In animals that walk on all fours the load is even and spread.", 0.90, x=0.11),
             _line("FUNCTIONS OF THE", 0.84, x=0.12),
             _line("VERTEBRAL BODY", 0.83, x=0.59),     # label on the spine drawing
             _line("LUMBAR SPINE", 0.82, x=0.12),
             _line("The spine carries the weight of the upper body at all times.", 0.78, x=0.11)]
    md = "\n".join(ocr_book.to_markdown(lines, {}))
    assert "## Functions Of The Lumbar Spine" in md
    assert "Vertebral" not in md


def test_exercise_heading_tolerates_ocr_case_slip_but_not_photo_labels():
    lines = [_line("ExERCISE 4", 0.93), _line("Extension in Standing", 0.90),
             _line("Stand upright with your feet slightly apart.", 0.87),
             _line("Exercise 1", 0.40, x=0.70)]
    md = "\n".join(ocr_book.to_markdown(lines, {}))
    assert "## Exercise 4 – Extension in Standing" in md
    assert md.count("##") == 1


def test_contents_leaders_bleeding_across_the_gutter_are_dropped():
    lines = [_line(".. 35 .. 35 ..40", 0.95), _line("In about 1956, in my clinic, I", 0.90),
             _line(".. 51", 0.89), _line("observed by chance a remarkable event.", 0.88),
             _line("•44", 0.86), _line("- . 62", 0.85)]
    md = "\n".join(ocr_book.to_markdown(lines, {}))
    assert md == "In about 1956, in my clinic, I observed by chance a remarkable event."


def test_chapter_heading_pairs_across_lines():
    lines = [_line("CHAPTER 7", 0.9), _line("SPECIAL SITUATIONS", 0.87), _line("Some text here.", 0.8)]
    assert "# Chapter 7 – Special Situations" in "\n".join(ocr_book.to_markdown(lines, {}))


def test_page_markers_give_book_page_numbers(tmp_path):
    p = tmp_path / "book.txt"
    p.write_text("<!-- page n/a -->\n## Introduction\nIntro text.\n\f"
                 "<!-- page 40 -->\n# Chapter 4 – Exercises\n## Exercise 1\nLie face down.\n\f"
                 "<!-- page 41 -->\nKeep lying.\n## Exercise 2\nPress up.\n")
    secs = {s.path: s for s in parse_text(p)}
    assert secs["Introduction"].page_start == 0
    ex1 = secs["Chapter 4 – Exercises > Exercise 1"]
    assert (ex1.page_start, ex1.page_end) == (40, 41)
    assert secs["Chapter 4 – Exercises > Exercise 2"].page_start == 41


def test_word_correction_is_dictionary_gated(monkeypatch, tmp_path):
    d = tmp_path / "words"
    d.write_text("exercise\nchronic\nnumbness\npain\nfree\nback\nthe\nfor\n")
    monkeypatch.setattr(ocr_book, "DICT_PATH", d)
    text = ("McKenzie " * 12) + ("MeKenzie " * 5) + ("centralisation " * 5) + "esercises chronie mumbness painfree 3-4 days"
    fixed, fixes = ocr_book.correct_words(text)
    assert "MeKenzie" not in fixed and fixed.count("McKenzie") == 17
    assert fixed.count("centralisation") == 5   # frequent, no common variant: protected
    assert "exercises" in fixed and "chronic" in fixed and "numbness" in fixed
    assert "painfree" in fixed and "3-4" in fixed   # unusual-but-right word, digits untouched


def test_word_correction_leaves_irregular_words_and_unfixable_junk(monkeypatch, tmp_path):
    d = tmp_path / "words"
    d.write_text("become\nwhen\nperform\nsehen\n")
    monkeypatch.setattr(ocr_book, "DICT_PATH", d)
    fixed, _ = ocr_book.correct_words("it became worse when pertormed")
    assert fixed == "it became worse when performed"
