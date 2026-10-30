import pytest

from backpain_rag import guardrails, llm
from backpain_rag.guardrails import (
    MAX_UNSUPPORTED_FRACTION, check_numbers, classify, evaluate_confidence, numbers_in, repair,
    rule_screen, validate_output,
)
from backpain_rag.models import (
    Claim, ConfidenceAction, GroundedAnswer, Intent, RefusalReason, Retrieved,
)

from .conftest import make_chunk

# --- gate 1 -------------------------------------------------------------------


@pytest.mark.parametrize("q", [
    "I can't control my bladder and my back is killing me",
    "I have numbness between my legs and in my bottom since this morning",
    "i fell off a ladder and hurt my back",
    "My back pain is so bad I want to end my life",
    "my legs suddenly went weak",
])
def test_rules_catch_emergencies(q):
    v = rule_screen(q)
    assert v is not None and v.intent is Intent.EMERGENCY


@pytest.mark.parametrize("q", [
    "Should I take ibuprofen for my back pain?",
    "My MRI says L4-L5 disc bulge, is that serious?",
    "can i go running with a slipped disc",
    "My lower back hurts when I bend over",
    "what should i do about sciatica",
])
def test_rules_catch_personal_advice(q):
    v = rule_screen(q)
    assert v is not None and v.intent is Intent.PERSONALIZED_ADVICE


@pytest.mark.parametrize("q", [
    "What are the red flags for cauda equina syndrome?",
    "Is loss of bladder control a sign of cauda equina syndrome?",
    "Is bed rest recommended for acute low back pain?",
    "When is MRI recommended for sciatica?",
])
def test_rules_leave_general_questions_alone(q):
    # The symptom alone must not fire: guideline questions mention red flags.
    assert rule_screen(q) is None


def test_rules_never_clear_a_query(monkeypatch):
    """With the model screen off, rules return in_scope only as 'unscreened',
    and with it on, the model decides -- the rules never stamp in_scope."""
    calls = []

    def fake_parse(schema, system, user, **kw):
        calls.append(user)
        return schema(intent=Intent.OUT_OF_DOMAIN, reason="model says no")

    monkeypatch.setattr(guardrails, "parse", fake_parse)
    v = classify("what is spinal stenosis")
    assert v.intent is Intent.OUT_OF_DOMAIN and calls == ["what is spinal stenosis"]


def test_model_not_consulted_when_rules_refuse(monkeypatch):
    monkeypatch.setattr(guardrails, "parse", lambda *a, **k: pytest.fail("model consulted"))
    assert classify("should i stretch my back").intent is Intent.PERSONALIZED_ADVICE


def test_gate1_fails_closed(monkeypatch):
    def boom(*a, **k):
        raise llm.LLMError("down")

    monkeypatch.setattr(guardrails, "parse", boom)
    v = classify("what is sciatica")
    assert v.intent is Intent.OUT_OF_DOMAIN and v.matched_rule == "fail-closed"
    assert "intent_classifier" in llm.degradations()


# --- gate 2 -------------------------------------------------------------------


def _scored(*scores):
    return [Retrieved(chunk=make_chunk(f"c{i}", "text"), rerank_score=s) for i, s in enumerate(scores)]


def test_confidence_three_way():
    assert evaluate_confidence(_scored(9, 3)).action is ConfidenceAction.PROCEED
    assert evaluate_confidence(_scored(6.5, 3)).action is ConfidenceAction.CORRECT
    assert evaluate_confidence(_scored(4, 3)).action is ConfidenceAction.REFUSE


def test_correction_band_collapses_on_second_look():
    v = evaluate_confidence(_scored(6.5), allow_correction=False)
    assert v.action is ConfidenceAction.REFUSE and not v.passed


def test_unscored_pool_refuses_without_retry():
    v = evaluate_confidence(_scored(None, None))
    assert not v.passed and v.action is ConfidenceAction.REFUSE


def test_kept_passages_clear_the_bar_and_get_markers():
    v = evaluate_confidence(_scored(10, 8, 5))
    assert [r.marker for r in v.kept] == ["C1", "C2"]


# --- gate 3 -------------------------------------------------------------------


def test_numbers_canonicalise_grouping_but_not_enumerations():
    assert numbers_in("10,000 and 1,00,000") == {"10000", "100000"}
    assert numbers_in("chunks 1,2") == {"1", "2"}


def _kept():
    r = Retrieved(chunk=make_chunk("nice::0003::00", "Consider oral NSAIDs for 2 weeks."),
                  marker="C1", rerank_score=9)
    return [r]


def test_fabricated_number_is_caught():
    ans = GroundedAnswer(answer="The guideline suggests NSAIDs for 6 weeks [C1].",
                         claims=[Claim(text="NSAIDs for 6 weeks", chunk_ids=["C1"])])
    assert check_numbers(ans, _kept()) == ["6"]


def test_number_words_count_on_both_sides():
    kept = [Retrieved(chunk=make_chunk("b::1", "Repeat ten times per session, six to eight times a day."),
                      marker="C1", rerank_score=9)]
    ok = GroundedAnswer(answer="The book describes 10 repetitions, 6 to 8 sessions a day [C1].",
                        claims=[Claim(text="10 repetitions", chunk_ids=["C1"])])
    assert check_numbers(ok, kept) == []
    bad = GroundedAnswer(answer="The book describes twelve repetitions per session [C1].",
                         claims=[Claim(text="twelve repetitions", chunk_ids=["C1"])])
    assert check_numbers(bad, kept) == ["12"]
    assert numbers_in("twenty-four hours, one hundred") >= {"24", "1", "100"}
    prose = GroundedAnswer(answer="It is one of the most useful routines in the book [C1].",
                           claims=[Claim(text="most useful", chunk_ids=["C1"])])
    assert check_numbers(prose, kept) == []


def test_numbers_from_the_question_and_markers_are_exempt():
    ans = GroundedAnswer(answer="1. The guideline lists NSAIDs for 2 weeks [C1].",
                         claims=[Claim(text="NSAIDs for 2 weeks", chunk_ids=["C1"])])
    assert check_numbers(ans, _kept(), query="what about 2 weeks") == []


def test_invalid_citation_refused():
    ans = GroundedAnswer(answer="Something [C9].", claims=[Claim(text="x", chunk_ids=["C9"])])
    v = validate_output(ans, _kept())
    assert not v.passed and v.reason is RefusalReason.UNGROUNDED_OUTPUT


def test_insufficient_context_is_model_declined():
    v = validate_output(GroundedAnswer(answer="", insufficient_context=True), _kept())
    assert v.reason is RefusalReason.MODEL_DECLINED


def test_repair_drops_the_unsupported_sentence_and_keeps_the_rest():
    ans = GroundedAnswer(
        answer=("The guideline recommends a group exercise programme for low back pain [C1]. "
                "It advises against paracetamol alone for low back pain management [C1]. "
                "Acupuncture is strongly recommended for everyone with chronic pain [C1]."),
        claims=[
            Claim(text="group exercise programme recommended for low back pain", chunk_ids=["C1"]),
            Claim(text="advises against paracetamol alone for low back pain", chunk_ids=["C1"]),
            Claim(text="Acupuncture is strongly recommended for everyone with chronic pain",
                  chunk_ids=["C1"]),
        ],
    )
    fixed = repair(ans, ["Acupuncture is strongly recommended for everyone with chronic pain"])
    assert fixed is not None
    assert "Acupuncture" not in fixed.answer and len(fixed.claims) == 2
    assert 1 / 3 <= MAX_UNSUPPORTED_FRACTION


def test_support_judge_outage_fails_closed(monkeypatch):
    def boom(*a, **k):
        raise llm.LLMError("down")

    monkeypatch.setattr(guardrails, "parse", boom)
    ans = GroundedAnswer(answer="The guideline lists NSAIDs for 2 weeks [C1].",
                         claims=[Claim(text="NSAIDs for 2 weeks", chunk_ids=["C1"])])
    v = validate_output(ans, _kept())
    assert not v.passed
