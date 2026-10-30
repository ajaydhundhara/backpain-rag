"""The domain boundary, stated once, in one place.

Everything specific to "Back Pain Help" lives in this file: what is in scope,
what every refusal says, the red-flag emergency rules, the personal-advice
rules, and the abbreviation glossary. The pipeline itself is domain-agnostic;
retargeting it to another corpus means editing this file and the eval set.

The distinction that makes the assistant useful at all
------------------------------------------------------
Back pain guidelines are *made of* recommendations -- exercise, NSAIDs, when to
image, when to refer. A filter that blocked anything resembling advice would
refuse the whole corpus. The line is attribution and addressee, not subject:

  ALLOWED    "NICE recommends against routine imaging for non-specific low back
              pain in non-specialist settings [C2]."   -> reports a document
  FORBIDDEN  "You don't need an MRI."                   -> instructs a person

Same topic. One reports; the other instructs.
"""

from __future__ import annotations

from .models import RefusalReason

DOMAIN = "back pain"

DISCLAIMER = (
    "This assistant summarises the back pain sources in its knowledge base "
    "only. It does not give personal medical advice. Speak to a "
    "doctor, physiotherapist or pharmacist about your own symptoms."
)

EMERGENCY_MESSAGE = (
    "Some back pain symptoms need urgent medical attention, and I can't assess "
    "an individual situation.\n\n"
    "**If you have new numbness around your genitals or bottom, new problems "
    "controlling your bladder or bowels, weakness in your legs, back pain after a "
    "serious fall or accident, or back pain with a high fever — contact emergency "
    "services now (911 in the US, 999 in the UK, 112 in the EU and India) or go to "
    "the nearest emergency department.**\n\n"
    "If you are thinking about harming yourself, please contact a crisis line or "
    "emergency services right away."
)

REFUSALS: dict[RefusalReason, str] = {
    RefusalReason.EMERGENCY: EMERGENCY_MESSAGE,
    RefusalReason.PERSONALIZED_ADVICE: (
        "I can't give personal medical advice, diagnose the cause of someone's "
        "pain, or recommend a treatment for a specific person — that needs a "
        "clinician who can examine you.\n\n"
        "I can tell you what my sources say in general. Try asking something "
        "like \"what exercises does the book describe for low back pain?\""
    ),
    RefusalReason.OUT_OF_DOMAIN: (
        "I only answer questions about back and spine pain from the sources in "
        "my knowledge base. That question falls outside them."
    ),
    RefusalReason.LOW_CONFIDENCE: (
        "I don't have enough information in my sources to answer this."
    ),
    RefusalReason.MODEL_DECLINED: (
        "I don't have enough information in my sources to answer this."
    ),
    RefusalReason.UNGROUNDED_OUTPUT: (
        "I don't have enough information in my sources to answer this."
    ),
    RefusalReason.UNSAFE_OUTPUT: (
        "I can't answer that in a way that stays within my boundaries — the "
        "response would have read as personal medical advice rather than a "
        "summary of what the sources say."
    ),
}


def refusal_text(reason: RefusalReason) -> str:
    return REFUSALS[reason]


SCOPE_STATEMENT = """\
You are "Back Pain Help", an assistant that explains what the back pain \
sources in its knowledge base say -- clinical guidelines, self-help books and \
patient-information material -- about back and spine pain: low back pain, \
neck pain, sciatica, their causes, posture, self-management, exercises and \
rehabilitation, physical and psychological therapies, medicines, surgery, red \
flags and referral. You are not a clinician and you are not talking to a \
patient about their own case.

ALLOWED: summarising and explaining source content in the third person, \
always attributed to a cited source.

FORBIDDEN: telling a specific person what to do; diagnosing; recommending a \
treatment, exercise or dose for an individual; interpreting someone's \
symptoms, scans or history; emergency or crisis instructions."""


# --- Gate 1 rule pass --------------------------------------------------------
# Each pattern needs a *personal* framing to fire. "Cauda equina syndrome red
# flags" is a legitimate guideline question; "I can't feel my legs" is an
# emergency. Matching the symptom alone would refuse much of the corpus.

FIRST_PERSON = r"(?:\bi\b|\bmy\b|\bme\b|\bmine\b|i'm|i've|\bim\b)"

EMERGENCY_PATTERNS: list[tuple[str, str]] = [
    (rf"{FIRST_PERSON}[^.?!]{{0,50}}\b(can'?t|cannot|unable to|lost|losing|no)\b[^.?!]{{0,20}}"
     r"\b(control|feel|feeling|sensation)\b[^.?!]{0,30}\b(bladder|bowels?|pee|poo|urinat\w*|"
     r"legs?|groin|genitals?|bottom|buttocks?|saddle)\b",
     "first-person cauda equina red flag"),
    (rf"{FIRST_PERSON}[^.?!]{{0,40}}\b(numb|numbness|tingling)\b[^.?!]{{0,40}}"
     r"\b(groin|genitals?|bottom|buttocks?|saddle|between (my|the) legs)\b",
     "first-person saddle anaesthesia"),
    (rf"{FIRST_PERSON}[^.?!]{{0,40}}\b(wet myself|incontinen\w*|can'?t pee|cannot pee|"
     r"can'?t urinate)\b", "first-person sphincter disturbance"),
    (rf"{FIRST_PERSON}[^.?!]{{0,40}}\b(legs?|feet|foot)\b[^.?!]{{0,30}}\b(gave way|giving way|"
     r"suddenly weak|went weak|paralys\w*|can'?t move)\b", "first-person acute leg weakness"),
    (rf"{FIRST_PERSON}[^.?!]{{0,60}}\b(fell|fall|fallen|car crash|accident|hit by)\b[^.?!]{{0,60}}"
     r"\b(back|spine|neck)\b", "first-person trauma"),
    (rf"{FIRST_PERSON}[^.?!]{{0,40}}\b(chest pain|can'?t breathe|cannot breathe|"
     r"passed out|unconscious|overdos\w*)\b", "first-person acute symptom"),
    (r"\b(kill myself|suicidal|suicide|end my life|harm myself|self[- ]harm)\b", "self-harm"),
    (r"\b(emergency|urgent|right now|immediately)\b[^.?!]{0,30}\b(help|what do i do|"
     r"what should i do)\b", "explicit urgency"),
]

ADVICE_PATTERNS: list[tuple[str, str]] = [
    (r"\bshould i\b", "asks what the user should do"),
    (r"\bcan i (take|use|stop|start|have|do|go|lift|run|exercise|work)\b",
     "asks permission for own action"),
    (r"\b(do|does) (i|my \w+) (have|need)\b", "asks for a personal diagnosis or need"),
    (r"\bwhat('?s| is) wrong with (me|my)\b", "asks for a diagnosis"),
    (r"\bdiagnos(e|is) (me|my)\b", "asks for a diagnosis"),
    (rf"\bhow (much|many|long|often)\b[^.?!]{{0,30}}\b(should|do|can)\s+{FIRST_PERSON}",
     "asks own dose or duration"),
    (rf"{FIRST_PERSON}\s+(have|had|has|get|got|feel|felt|am feeling|experience|'ve had)\b"
     r"[^.?!]{0,50}\b(pain|ache|aching|sore|stiff|stiffness|sciatica|spasm|twinge|tingling|"
     r"numb)\b", "describes own symptoms"),
    (r"\bmy (back|neck|spine|lower back|sciatica|disc|hip|leg)\b[^.?!]{0,40}"
     r"\b(hurts?|aches?|pain|sore|stiff|killing|is|feels|went)\b", "describes own back"),
    (r"\bis it safe for me\b", "asks personal safety"),
    (r"\bwhat (should|can) i (take|do|use|try)\b", "asks what the user should do"),
    (r"\bmy (mri|x-?ray|scan|ct|report|results?|test|blood)\b",
     "asks to interpret own results"),
]


INTENT_GUIDE = """\
Classify the user's query into exactly one label:

- "in_scope": a general question about back, neck or spine pain -- what a \
condition is, causes, risk factors, prognosis, assessment, red flags, imaging, \
self-management, exercise, physiotherapy, psychological therapies, medicines, \
injections, surgery or referral. Third-person or impersonal framing. The user \
does not have to mention guidelines: "what is sciatica" is a request for what \
the sources say about sciatica.
- "personalized_advice": asks you to advise, diagnose or recommend for a \
specific, identified person -- the user ("should I", "my back"), someone they \
identify as theirs ("my mum's disc"), or a person whose particulars they \
supply ("a 40-year-old with L5 numbness for 3 weeks, what next"). The test is \
whether answering means reasoning about one real individual's situation.
- "emergency": describes an acute or crisis situation needing immediate care.
- "out_of_domain": not about back, neck or spine pain (other medical topics \
such as diabetes or knee injuries, or non-medical topics such as sport, code, \
travel, politics), or a question about you.

An INDEFINITE person is not a specific person. "older adults", "pregnant \
women", "people with sciatica", "a manual worker" are clinical categories -- \
the vocabulary guidelines are written in -- so a question framed around one is \
in_scope. Compare: "is bed rest recommended for people with acute low back \
pain" (in_scope) vs "should I stay in bed with my bad back" (personalized).

Whether the sources actually cover the topic is NOT yours to decide -- a later \
stage refuses when retrieval finds nothing. When a back pain question is \
neither personal nor urgent, it is in_scope.

Examples:
"what is sciatica" -> in_scope
"are opioids recommended for chronic low back pain" -> in_scope
"when is an MRI recommended for low back pain" -> in_scope
"what are the red flags for cauda equina syndrome" -> in_scope
"what exercises help pregnant women with back pain" -> in_scope
"should i take ibuprofen for my back" -> personalized_advice
"my dad's sciatica is getting worse, what should he do" -> personalized_advice
"i suddenly can't feel my legs and my back hurts" -> emergency
"how do i treat a sprained ankle" -> out_of_domain
"who won the cricket match" -> out_of_domain"""


# --- Query rewrite glossary ---------------------------------------------------
# Expanded only where unambiguous in a back pain corpus. Expansions are
# *additive* ("LBP low back pain"), so BM25 keeps the exact token.

GLOSSARY: dict[str, str] = {
    "lbp": "low back pain",
    "nslbp": "non-specific low back pain",
    "clbp": "chronic low back pain",
    "nsaid": "non-steroidal anti-inflammatory drug",
    "nsaids": "non-steroidal anti-inflammatory drugs",
    "ces": "cauda equina syndrome",
    "cbt": "cognitive behavioural therapy",
    "tens": "transcutaneous electrical nerve stimulation",
    "sij": "sacroiliac joint",
    "ddd": "degenerative disc disease",
    "slr": "straight leg raise",
    "rom": "range of motion",
    "mri": "magnetic resonance imaging",
    "otc": "over the counter",
    "snri": "serotonin noradrenaline reuptake inhibitor",
    "tca": "tricyclic antidepressant",
    "rfa": "radiofrequency denervation ablation",
    "esi": "epidural steroid injection",
    "smt": "spinal manipulative therapy",
    "rcts": "randomised controlled trials",
}

#: Deliberately NOT expanded -- more than one plausible reading here.
AMBIGUOUS = frozenset({"as", "pt", "ms", "ra", "ct", "pid", "oa"})
