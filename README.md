# Back Pain Help

A retrieval-augmented assistant that answers questions about back, neck and
spine pain **only** from a curated set of sources, with citations, and refuses
everything else. It is a personal project. The current corpus is an OCR of
*Treat Your Own Back* (Robin McKenzie); see `data/corpus_manifest.yaml`.

The architecture follows the Medical RAG project walked through in
[this video](https://www.youtube.com/watch?v=ZBXMIfANEXk), which is based on
[abhijitWebDev/medic-guidelines-rag](https://github.com/abhijitWebDev/medic-guidelines-rag).
This repo reimplements that design for a different domain. The original repo
has no license, so no code was copied.

The interesting part is not the retrieval. It is the three independent gates
that decide when *not* to answer.

## Pipeline

```
query
 │
 ├─ GATE 1  intent          personalized / emergency / off-domain → refuse
 │                          (no retrieval happens for a refused query)
 ├─ rewrite                 additive abbreviation expansion (LBP → "LBP low back pain")
 ├─ HyDE                    hypothetical passage blended into the dense query vector
 ├─ retrieve                dense (local vectors) + BM25, fused with RRF
 ├─ rerank                  absolute 0–10 relevance score per passage
 ├─ GATE 2  confidence      top-1 ≥ 7.5              → proceed
 │                          top-1 in [6.0, 7.5)      → retry once (deeper, HyDE-weighted),
 │                                                     re-judge at the SAME bar
 │                          top-1 < 6.0              → "not enough information"
 ├─ generate                JSON: answer + claims[] each citing passage markers
 ├─ GATE 3  output          citations exist → numbers traceable →
 │                          claims supported (salvage minority failures) → framing safe
 └─ answer + citations + disclaimer
```

Gate 1 stops questions that should never be answered. Gate 2 stops questions
the corpus cannot answer. Gate 3 stops answers the corpus does not support.

## Design decisions

- **Rules may only escalate, never clear.** Gate 1's regexes (red flags such as
  cauda equina symptoms, and "should I…" / "my MRI…" phrasings) can refuse a
  query outright, but they can never mark it in scope. Anything they don't
  catch goes to the model. Otherwise a phrasing the patterns missed would
  bypass the check: the blocklist would become an allowlist.
- **An indefinite person is not a person.** "Is bed rest recommended for people
  with acute LBP" is in scope. "Should I stay in bed with my bad back" is not.
- **The reranker scores on an absolute scale, not relatively.** Gate 2 refuses
  on the top-1 score, and the best of five irrelevant passages still ranks
  first. So the prompt asks "how well does this passage answer the question"
  on a fixed scale, never "which is best".
- **Numbers are checked mechanically.** A fabricated duration or dose that
  cites a real passage reads fluently and passes citation checks. Gate 3
  compares every number in the answer against the cited passages. That
  catches the most dangerous failure mode at no cost.
- **Framing is the boundary, not subject matter.** "NICE advises against
  paracetamol alone [C2]" is allowed. "Don't bother with paracetamol" is not.
  The topic is the same; one reports and the other instructs.
- **Everything fails closed.** If the classifier, reranker or a judge is
  unavailable, the result is a refusal, never a pass. The exception is HyDE,
  which is a retrieval improvement, not a gate: it falls back to the plain
  query vector. Degraded responses are never cached.
- **Partial answers are salvaged.** If a minority (≤40%) of claims fail the
  support judge, those claims and the sentences carrying them are stripped and
  the rest is returned.
- **RRF fuses ranks, not scores.** BM25 scores are unbounded and cosine scores
  are bounded, so normalising them onto one scale would invent a comparison.
- **HyDE is blended, not substituted, and only on the dense half.** The
  hypothetical passage is embedded and then discarded. It never reaches the
  reranker, the generator or the citations. BM25 keeps searching the literal
  query, because exact terms ("L5", "TENS") matter there.
- **The corrective retry never lowers the bar.** It runs once, looks deeper,
  keeps the union of both passes, and re-judges at the same threshold. It is
  CRAG with the web-search fallback removed, because every claim must trace
  to a sha256-pinned source.
- **Corpus governance.** A file in `data/raw/` is not part of the corpus until
  it is listed in `data/corpus_manifest.yaml` with a matching sha256.
  `data/index_manifest.json` records which documents, chunking settings and
  embedding model built the current index.
- **One file holds the domain.** `src/backpain_rag/policy.py` contains the
  scope statement, refusal texts, emergency/advice rules and glossary.
  Retarget the assistant by editing that file and the eval set.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'     # [ingest] alone if you don't need tests
cp .env.example .env                  # set OPENAI_API_KEY
```

## Workflow

```bash
# 1. Put guideline PDFs / .txt / .md in data/raw/ (see suggestions in
#    data/corpus_manifest.yaml), then build the allow-list.
rag corpus scan          # adds new files with their sha256; fill in title/url/publisher
rag corpus verify        # every file present, hash matches, nothing unlisted

# 2. Parse and chunk. Writes data/chunks/*.jsonl -- read a few before indexing.
rag ingest

# 3. Embed into the local vector index (data/index/).
rag index

# 4. Ask.
rag ask "What do guidelines say about imaging for low back pain?"
rag ask "..." --trace      # per-gate trace
rag ask "..." --json       # full structured response

# 5. Measure and tune. Edit data/eval/questions.yaml to match your corpus first.
rag eval run --bucket personalized     # safety buckets work before any corpus exists
rag eval calibrate --write             # set the gate-2 threshold from data
rag eval run                           # full pipeline metrics
rag eval run --judge --out results.json   # + LLM-judged faithfulness/completeness/relevance

# 6. Serve the UI + API.
rag serve                # UI -> http://127.0.0.1:8000/   API docs -> /docs
```

### Scanned books (OCR)

Scanned PDFs have no text layer, and `rag ingest` rejects them with a clear
error. `tools/ocr_book.py` converts a scan of two-page book spreads into the
Markdown-with-page-markers text that ingest reads. It uses Apple Vision, so it
runs on macOS only and works offline.

```bash
pip install -e '.[ocr]'
python tools/ocr_book.py ~/Downloads/backpain.pdf data/raw/treat-your-own-back.txt --skip-front 3
rag corpus scan --rehash      # re-pin the sha256 after regenerating
```

What the tool does:

- Splits each spread at the gutter.
- Reads the printed page numbers ("SEVENTY-THAEE" → 73) and repairs misreads
  from their neighbours, so citations use the book's own page numbers.
- Turns chapter, section and exercise headings into Markdown.
- Drops thumb tabs, figure captions, photo labels, diagram labels and
  table-of-contents leaders.
- Fixes systematic one-letter OCR confusions ("esercises" → "exercises"),
  but only when the result is a dictionary word.
- Reports figure-only pages and pages missing from the scan.

OCR output is cached in `data/ocr_cache/`, so re-runs take seconds.

For other plain-text sources, put a form feed (`\f`) between pages, or add
`<!-- page N -->` markers.

## The UI

`/` is a single static page with no build step and no CDN. It calls the same
`/api/*` endpoints as any other client. Under every result is a **Pipeline**
panel showing what each stage decided, including the stages a refusal never
reached. Hovering a `[C1]` marker highlights its source card. Badges flag
`⚠ degraded` (a model was unreachable, so the refusal is about the service,
not the question) and `⚡ served from cache`.

## Optional features (all keyed off configuration)

| Set this | And you get |
|---|---|
| `DATABASE_URL` (Postgres) or `AUTH_ENABLED=true` (SQLite) | sign-up / sign-in, per-account history, per-account quota |
| `SES_SMTP_HOST` + `MAIL_FROM` | email verification (gates asking) + password reset |
| `REDIS_URL` | shared answer/embedding cache and rate-limit counters across instances |
| `SESSION_SECRET` | sessions that survive restarts (required for any deployment) |

Two stores, opposite failure modes:

|            | Redis (`cache.py`)                | Database (`db.py`)   |
|------------|-----------------------------------|----------------------|
| Holds      | answers, embeddings, HyDE, counters | users, history     |
| On failure | compute normally (fail open)      | 503 (fail closed)    |

- Passwords use stdlib `scrypt`. Each hash stores its own cost parameters, so
  raising `SCRYPT_N` re-hashes users as they next sign in.
- Session and email links are HMAC-signed, with the purpose included in the
  signature, so a verification link can never reset a password. Nothing is
  stored server-side. A password change invalidates every older session and
  reset link.
- `/forgot` returns the same response whether or not the account exists.
- The quota (default 5 questions/hour) is a spend control, not a security
  boundary. Set a hard monthly limit on a project-scoped OpenAI key as well.
  Sign-in attempts are metered separately, by IP.

```bash
rag users list
rag users add someone@example.com      # prompts for a password; pre-verified
rag users delete someone@example.com   # account and all its history
```

## Deploying to Vercel

`app.py` and `vercel.json` are ready. Build the index locally, commit
`data/chunks/`, `data/index/` and `data/index_manifest.json` (the vector index
ships inside the bundle; there is no vector database to run), then set
`OPENAI_API_KEY`, `DATABASE_URL` (the *pooled* Neon/Supabase URL),
`SESSION_SECRET` and `PUBLIC_BASE_URL` in the project settings. Without
`DATABASE_URL`, SQLite on Vercel is wiped at every cold start.

## Layout

```
src/backpain_rag/
  config.py       all tunables; gate-2 thresholds load from data/calibration.json
  models.py       shared schemas (Claim.chunk_ids must be non-empty)
  policy.py       THE DOMAIN: scope, refusals, red-flag + advice rules, glossary
  llm.py          OpenAI wrapper; raises LLMError, tracks degradations
  corpus.py       sha256 allow-list
  ingest.py       PDF/MD/TXT -> sections with heading paths -> token-bounded chunks
  index.py        embeddings, local vector store, BM25, RRF, index manifest
  retrieval.py    rewrite, HyDE, hybrid search, LLM reranker, corrective retry
  guardrails.py   gate 1 (intent), gate 2 (confidence), gate 3 (output)
  generate.py     grounded, citation-bearing generation
  assistant.py    the pipeline end to end
  evaluation.py   eval set, runner, calibration, LLM judges
  cache.py        Redis or in-process; fails open
  db.py           users + history on Postgres or SQLite; fails closed
  passwords.py    scrypt
  security.py     signed sessions + email-link tokens + rate limiting
  mailer.py       SES over SMTP, stdlib only
  api.py          FastAPI: JSON API, auth pages, serves web/index.html
  cli.py          `rag` command
tests/            65 tests; the LLM is faked, so the suite runs offline in < 1s
tools/ocr_book.py scanned book spreads -> page-numbered Markdown (macOS Vision OCR)
```

## Scope

Answers are limited to the ingested guidelines. This is not a diagnostic tool
and does not give personal medical advice.
