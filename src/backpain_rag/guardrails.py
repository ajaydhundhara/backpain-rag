"""The three gates that decide when NOT to answer.

  Gate 1  intent       -- stops questions that should never be answered
  Gate 2  confidence   -- stops questions the corpus cannot answer
  Gate 3  output       -- stops answers the corpus does not support

Each catches a different failure, and every one of them fails CLOSED: an
unavailable classifier, reranker or judge produces a refusal, never a pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .config import get_settings
from .llm import LLMError, note_degraded, parse
from .models import (
    ConfidenceAction, GroundedAnswer, Intent, IntentVerdict, RefusalReason, Retrieved,
)
from .policy import ADVICE_PATTERNS, EMERGENCY_PATTERNS, INTENT_GUIDE, SCOPE_STATEMENT

# =============================================================================
# Gate 1 -- intent
# =============================================================================
# Rules run first and may only ever ESCALATE to a refusal; they are never allowed
# to stamp a query in_scope. If they could, any phrasing the regexes failed to
# anticipate would bypass the model entirely -- a blocklist turned allowlist.
# Retrieval never runs for a refused query.

_COMPILED = {
    Intent.EMERGENCY: [(re.compile(p, re.IGNORECASE), why) for p, why in EMERGENCY_PATTERNS],
    Intent.PERSONALIZED_ADVICE: [(re.compile(p, re.IGNORECASE), why) for p, why in ADVICE_PATTERNS],
}


def rule_screen(query: str) -> IntentVerdict | None:
    """A refusal verdict, or None meaning 'undecided -- ask the model'."""
    # Emergency first: a query can be both personal and urgent; urgent must win.
    for intent in (Intent.EMERGENCY, Intent.PERSONALIZED_ADVICE):
        for pattern, why in _COMPILED[intent]:
            if pattern.search(query):
                return IntentVerdict(intent=intent, reason=why,
                                     matched_rule=pattern.pattern[:60])
    return None


class _IntentOut(BaseModel):
    intent: Intent
    reason: str = Field(description="one short clause justifying the label")


def classify(query: str, use_model: bool = True) -> IntentVerdict:
    ruled = rule_screen(query)
    if ruled is not None:
        return ruled
    if not use_model:
        return IntentVerdict(intent=Intent.IN_SCOPE, reason="rules only; model screen skipped")
    try:
        out = parse(_IntentOut, f"{SCOPE_STATEMENT}\n\n{INTENT_GUIDE}", query)
    except LLMError as e:
        note_degraded("intent_classifier")
        return IntentVerdict(
            intent=Intent.OUT_OF_DOMAIN,
            reason=f"intent classifier unavailable ({e}); refusing rather than assuming safe"[:240],
            matched_rule="fail-closed",
        )
    return IntentVerdict(intent=out.intent, reason=out.reason)


# =============================================================================
# Gate 2 -- retrieval confidence (three-way)
# =============================================================================
# Reads the ABSOLUTE rerank score of the best passage.
#   top >= confidence_threshold                        -> PROCEED
#   corrective_threshold <= top < confidence_threshold -> CORRECT (retry once)
#   below                                              -> REFUSE
# The bar never moves: a correction buys a second attempt, never a lower bar.


@dataclass
class ConfidenceVerdict:
    passed: bool
    top_score: float | None
    threshold: float
    reason: str
    kept: list[Retrieved]
    #: Defaults to REFUSE so a verdict built without one fails closed.
    action: ConfidenceAction = ConfidenceAction.REFUSE


def evaluate_confidence(
    results: list[Retrieved],
    threshold: float | None = None,
    allow_correction: bool = True,
) -> ConfidenceVerdict:
    s = get_settings()
    threshold = s.confidence_threshold if threshold is None else threshold
    floor = s.corrective_threshold
    correcting = allow_correction and s.corrective_enabled

    if not results:
        return ConfidenceVerdict(False, None, threshold, "retrieval returned nothing", [])

    scored = [r for r in results if r.rerank_score is not None]
    if not scored:
        # No retry: an unscored pool means the reranker is down, and a retry
        # would pay twice to reach the same fail-closed answer.
        return ConfidenceVerdict(False, None, threshold,
                                 "reranker produced no scores; refusing rather than guessing", [])

    top = max(r.rerank_score or 0.0 for r in scored)
    if top < threshold:
        if correcting and top >= floor:
            return ConfidenceVerdict(
                False, top, threshold,
                f"best passage scored {top:.1f}, inside corrective band "
                f"[{floor:.1f}, {threshold:.1f}) -- retrying retrieval",
                [], ConfidenceAction.CORRECT,
            )
        return ConfidenceVerdict(False, top, threshold,
                                 f"best passage scored {top:.1f} < {threshold:.1f}", [])

    # Keep only passages that clear the bar: padding the context with weak
    # passages invites the generator to cite them.
    kept = [r for r in scored if (r.rerank_score or 0.0) >= threshold][: s.rerank_top_n]
    for i, r in enumerate(kept, start=1):
        r.marker = f"C{i}"
    return ConfidenceVerdict(True, top, threshold, f"top passage scored {top:.1f}", kept,
                             ConfidenceAction.PROCEED)


# =============================================================================
# Gate 3 -- output validation
# =============================================================================
# Four checks, cheapest first, short-circuiting:
#   1. citation validity   (deterministic proof)
#   2. numeric provenance  (deterministic proof)
#   3. claim support       (LLM judge)
#   4. framing / safety    (LLM judge)
# Cheap proofs are never gated behind expensive opinions.
#
# Check 2 matters most: a model that invents a dose or a duration while citing a
# real passage passes citation validation and reads fluently. Comparing every
# number in the answer to the numbers in the cited passages catches that free.

_MARKER = re.compile(r"\[(?:C\d+)(?:\s*,\s*C\d+)*\]")
_LIST_NUM = re.compile(r"(^|[.!?]\s+)\d+[.)]\s", re.MULTILINE)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Thousands separators inside a number (10,000 and Indian 1,00,000). The
# lookahead needs 2-3 digits so "C1,2" style enumerations are not joined.
_GROUPING = re.compile(r"(?<=\d),(?=\d{2,3}\b)")


def strip_markers(text: str) -> str:
    return _LIST_NUM.sub(r"\1 ", _MARKER.sub(" ", text))


_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
          "eighteen", "nineteen"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_NUM_WORD = re.compile(
    r"\b(?:(" + "|".join(_TENS) + r")(?:[\s-](" + "|".join(_UNITS[1:10]) + r"))?|("
    + "|".join(_UNITS) + r")|(hundred))\b", re.IGNORECASE)


def _word_numbers(text: str) -> set[str]:
    """Number words as digits: "ten times" -> 10, "twenty-four" -> 24.

    Applied to BOTH sides of the check. Sources written in words ("six to eight
    times a day") must not make a faithful "6 to 8 times" look fabricated, and
    an answer must not slip an invented "twelve" past a digits-only check.
    """
    out: set[str] = set()
    for m in _NUM_WORD.finditer(text):
        tens, unit, single, hundred = (g.lower() if g else None for g in m.groups())
        if tens:
            out.add(str(_TENS[tens] + (_UNITS.index(unit) if unit else 0)))
        elif single:
            out.add(str(_UNITS.index(single)))
        elif hundred:
            out.add("100")
    return out


def numbers_in(text: str, answer_side: bool = False) -> set[str]:
    """Quantities in `text`. On the answer side the word "one" is not counted:
    in prose it is rarely a quantity ("one of the most useful exercises"), and
    flagging it would refuse correct answers."""
    digits = {n.lstrip("0") or "0" for n in _NUMBER.findall(_GROUPING.sub("", text))}
    words = _word_numbers(text)
    if answer_side and not re.search(r"\b1\b", text):
        words.discard("1")
    return digits | words


#: Above this share of unsupported claims the answer is refused, not salvaged.
MAX_UNSUPPORTED_FRACTION = 0.4
#: Share of a claim's words a sentence must contain to count as carrying it.
SENTENCE_CARRIES_CLAIM = 0.6

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _carried_by(claim: str, sentences: list[str]) -> bool:
    cw = _words(claim)
    return bool(cw) and any(len(_words(s) & cw) / len(cw) >= SENTENCE_CARRIES_CLAIM
                            for s in sentences)


def repair(answer: GroundedAnswer, unsupported: list[str]) -> GroundedAnswer | None:
    """Drop unsupported claims and the sentences carrying them; keep the rest.

    The generator is told partial answers are fine, so refusing six verified
    claims over one unsupported sentence contradicts its instructions. A
    supported claim that shared a dropped sentence goes too -- otherwise
    `claims` would name something `answer` no longer says. A claim that cannot
    be located in either half is left alone: failing to find it is not evidence
    that its sentence was removed. Returns None when nothing usable remains.
    """
    bad = set(unsupported)
    kept_claims = [c for c in answer.claims if c.text not in bad]
    if not kept_claims:
        return None
    bad_words = [_words(u) for u in unsupported]
    kept, dropped = [], []
    for sentence in _SENTENCE.split(answer.answer.strip()):
        sw = _words(sentence)
        if not sw:
            continue
        overlap = max((len(sw & bw) / max(1, len(bw)) for bw in bad_words), default=0.0)
        (kept if overlap < SENTENCE_CARRIES_CLAIM else dropped).append(sentence)
    prose = " ".join(kept).strip()
    if len(_words(prose)) < 12:
        return None
    kept_claims = [c for c in kept_claims
                   if _carried_by(c.text, kept) or not _carried_by(c.text, dropped)]
    if not kept_claims:
        return None
    return GroundedAnswer(answer=prose, claims=kept_claims)


@dataclass
class OutputVerdict:
    passed: bool
    reason: RefusalReason | None = None
    detail: str = ""
    repaired: GroundedAnswer | None = None
    invalid_citations: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)


def _lookup(results: list[Retrieved]) -> dict[str, Retrieved]:
    by: dict[str, Retrieved] = {}
    for r in results:
        by[r.chunk.chunk_id] = r
        if r.marker:
            by[r.marker] = r
    return by


def check_citations(answer: GroundedAnswer, results: list[Retrieved]) -> list[str]:
    valid = _lookup(results)
    return sorted({cid for c in answer.claims for cid in c.chunk_ids if cid not in valid})


def check_numbers(answer: GroundedAnswer, results: list[Retrieved], query: str = "") -> list[str]:
    """Every number in the answer must appear in a passage it cites. Citation
    markers, list numbering and numbers echoed from the question are exempt."""
    by = _lookup(results)
    cited = {cid for c in answer.claims for cid in c.chunk_ids}
    source = "\n".join({by[c].chunk.chunk_id: by[c].chunk.text for c in cited if c in by}.values())
    allowed = numbers_in(source) | numbers_in(query)
    return sorted(n for n in numbers_in(strip_markers(answer.answer), answer_side=True)
                  if n not in allowed)


class _Support(BaseModel):
    index: int
    supported: bool
    why: str = ""


class _Supports(BaseModel):
    results: list[_Support]


_SUPPORT_SYSTEM = """You verify citations for a back pain reference assistant.

For each numbered claim you are given the passage(s) it cites. Decide whether \
those passages actually state, or directly entail, the claim.

Be strict. supported=false if the claim adds a detail the passage lacks, \
generalises beyond it, or changes a number, population, recommendation \
strength or condition. Topical similarity is not support."""


def check_support(answer: GroundedAnswer, results: list[Retrieved]) -> list[str]:
    """Texts of claims the judge could not verify. Deliberately blind to the
    question: a judge that knows what the user wanted finds reasons to allow
    an answer that goes past its sources."""
    by = _lookup(results)
    blocks = []
    for i, claim in enumerate(answer.claims):
        cited = [by[c] for c in claim.chunk_ids if c in by]
        passages = "\n".join(f"  - {r.chunk.text}" for r in cited) or "  (no passage)"
        blocks.append(f"[{i}] Claim: {claim.text}\nCited passages:\n{passages}")
    try:
        out = parse(_Supports, _SUPPORT_SYSTEM, "\n\n".join(blocks))
    except LLMError:
        note_degraded("support_judge")
        return [c.text for c in answer.claims]
    verdicts = {s.index: s.supported for s in out.results}
    return [c.text for i, c in enumerate(answer.claims) if not verdicts.get(i, False)]


class _Safety(BaseModel):
    personalized_advice: bool = Field(description="tells a specific person what to do")
    diagnostic: bool = Field(description="asserts what condition a person has")
    imperative_treatment: bool = Field(description="instructs the reader to take/do a treatment")
    emergency_instruction: bool = Field(description="directs the reader to take urgent action")
    note: str = ""


_SAFETY_SYSTEM = """You check whether a drafted answer stays within its boundary.

The assistant may report what its back pain sources (guidelines, self-help \
books, patient information) say, in the third person, attributed to a source. \
It may discuss exercises, repetitions, medicines, doses, imaging, surgery and \
referral as *source content*.

It may NOT address the reader as a patient, tell anyone what to do, assert \
what someone has, or issue instructions. The distinction is framing, not topic:

  OK  - "NICE recommends against offering paracetamol alone for low back pain [C2]."
  NOT - "Don't bother taking paracetamol for your back."

  OK  - "Loss of bladder control is listed as a red flag needing emergency referral [C1]."
  NOT - "Go to A&E right now."

  OK  - "The guideline describes this presentation as sciatica [C3]."
  NOT - "You have sciatica."

  OK  - "McKenzie's Exercise 3 is performed ten times per session [C2]."
  NOT - "Do Exercise 3 ten times every session."

Red flags, exercise instructions and referral criteria are ordinary source CONTENT. Flag \
emergency_instruction only when the text directs the reader to act, not when \
it reports what a document says. Flag only genuine boundary crossings."""


def check_framing(answer: GroundedAnswer) -> list[str]:
    try:
        out = parse(_Safety, _SAFETY_SYSTEM, answer.answer)
    except LLMError:
        note_degraded("safety_judge")
        return ["safety judge unavailable"]
    return [name for name in ("personalized_advice", "diagnostic", "imperative_treatment",
                              "emergency_instruction") if getattr(out, name)]


def validate_output(answer: GroundedAnswer, results: list[Retrieved], query: str = "") -> OutputVerdict:
    if answer.insufficient_context or not answer.answer.strip():
        return OutputVerdict(False, RefusalReason.MODEL_DECLINED,
                             "model reported insufficient context")
    if not answer.claims:
        return OutputVerdict(False, RefusalReason.UNGROUNDED_OUTPUT,
                             "answer carried no claims to verify")

    invalid = check_citations(answer, results)
    if invalid:
        return OutputVerdict(False, RefusalReason.UNGROUNDED_OUTPUT,
                             f"cited passages that were not retrieved: {', '.join(invalid)}",
                             invalid_citations=invalid)

    bad_numbers = check_numbers(answer, results, query)
    if bad_numbers:
        return OutputVerdict(False, RefusalReason.UNSAFE_OUTPUT,
                             f"numbers absent from cited passages: {', '.join(bad_numbers)}",
                             unsupported_numbers=bad_numbers)

    unsupported = check_support(answer, results)
    repaired = None
    if unsupported:
        share = len(unsupported) / len(answer.claims)
        repaired = repair(answer, unsupported) if share <= MAX_UNSUPPORTED_FRACTION else None
        if repaired is None:
            return OutputVerdict(False, RefusalReason.UNGROUNDED_OUTPUT,
                                 f"{len(unsupported)}/{len(answer.claims)} claim(s) unsupported; "
                                 "could not be salvaged", unsupported_claims=unsupported)

    final = repaired or answer
    flags = check_framing(final)
    if flags:
        return OutputVerdict(False, RefusalReason.UNSAFE_OUTPUT,
                             f"framing violations: {', '.join(flags)}", safety_flags=flags)

    detail = (f"passed after stripping {len(unsupported)} unsupported claim(s)"
              if unsupported else "all output checks passed")
    return OutputVerdict(True, None, detail, repaired=repaired, unsupported_claims=unsupported)
