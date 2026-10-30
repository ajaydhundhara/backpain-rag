"""Command line: build the corpus, ask questions, measure, manage accounts.

    rag corpus scan | verify
    rag ingest
    rag index
    rag ask "question" [--trace] [--json] [--no-hyde] [--no-cache]
    rag eval run [--bucket B] [--judge] [--out FILE]
    rag eval calibrate [--write]
    rag users list | add EMAIL | delete EMAIL
    rag serve
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


# --- corpus / ingest / index --------------------------------------------------------


def cmd_corpus(a) -> int:
    from .corpus import CorpusError, scan, verify

    if a.action == "scan":
        docs, added = scan(rehash=a.rehash)
        for name in added:
            print(f"  + {name}")
        new = [n for n in added if not n.endswith("(re-pinned)")]
        print(f"{len(docs)} document(s) in data/corpus_manifest.yaml"
              + (f" ({len(new)} new)" if new else ""))
        if added:
            print("Fill in title / url / publisher / version by hand -- they appear in citations.")
        return 0
    try:
        docs = verify()
    except CorpusError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"ok: {len(docs)} document(s) verified against their sha256")
    return 0


def cmd_ingest(_a) -> int:
    from .corpus import CorpusError, verify
    from .ingest import ParseError, ingest_all

    try:
        docs = verify()
        counts = ingest_all(docs, on_doc=lambda d, n: print(f"  {d.doc_id:<40} {n:>5} chunks"))
    except (CorpusError, ParseError) as e:
        print(e, file=sys.stderr)
        return 1
    print(f"{sum(counts.values())} chunks written to data/chunks/ -- inspect them before indexing")
    return 0


def cmd_index(_a) -> int:
    from .corpus import load_manifest
    from .index import StoreError, build
    from .ingest import load_chunks
    from .llm import LLMError

    chunks = load_chunks()
    try:
        m = build(chunks, load_manifest(),
                  on_progress=lambda i, n: print(f"\r  embedded {i}/{n}", end="", flush=True))
    except (StoreError, LLMError) as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    print(f"\nindexed {m['n_chunks']} chunks with {m['embed_model']} -> data/index/")
    return 0


# --- ask -------------------------------------------------------------------------------


def _print_trace(t: dict) -> None:
    i = t.get("intent", {})
    print(f"  gate 1   {i.get('label')}  ({i.get('reason')})")
    if "retrieval" in t:
        r = t["retrieval"]
        print(f"  retrieve dense {r['dense_hits']} + lexical {r['lexical_hits']} -> {r['fused_pool']} fused")
        if r.get("expansions"):
            print(f"           expanded: {', '.join(r['expansions'])}")
        h = r.get("hyde", {})
        print(f"           hyde: {'used, w=' + str(h.get('query_weight')) if h.get('used') else h.get('reason')}")
    if "rerank" in t:
        print(f"  rerank   top scores {t['rerank']['top_scores']}")
    if "confidence" in t:
        print(f"  gate 2   {t['confidence']['action']}: {t['confidence']['reason']}")
    if "corrective" in t:
        c = t["corrective"]
        print(f"  correct  {c.get('top_before')} -> {c.get('top_after')} ({c.get('new_candidates', 0)} new)")
        if "confidence_after_correction" in t:
            print(f"  gate 2'  {t['confidence_after_correction']['reason']}")
    if "output_gate" in t:
        print(f"  gate 3   {t['output_gate']['detail']}")
    if t.get("degraded"):
        print(f"  DEGRADED {', '.join(t['degraded'])}")
    if t.get("cache"):
        print("  (served from cache)")


def cmd_ask(a) -> int:
    if a.no_hyde:
        os.environ["HYDE_ENABLED"] = "false"
    from .assistant import Assistant

    r = Assistant.build().ask(a.query, use_cache=not a.no_cache)
    if a.json:
        print(json.dumps(r.model_dump(mode="json"), indent=2))
        return 0
    print()
    print(r.answer)
    if r.citations:
        print("\nSources:")
        for c in r.citations:
            print(f"  [{c.marker}] {c.title} — §{c.section} — p.{c.pages}  (relevance {c.rerank_score})")
    if not r.answered:
        print(f"\n(refused: {r.refusal_reason})")
    print(f"\n{r.disclaimer}")
    if a.trace:
        print("\nTrace:")
        _print_trace(r.trace)
    return 0


# --- eval -------------------------------------------------------------------------------


def cmd_eval(a) -> int:
    from .evaluation import Bucket, calibrate, load_cases, run_eval, write_calibration

    cases = load_cases(bucket=getattr(a, "bucket", None))
    if not cases:
        print("no eval cases found in data/eval/questions.yaml", file=sys.stderr)
        return 1

    if a.action == "calibrate":
        res = calibrate(cases)
        for cid, top in sorted(res["scores"].items(), key=lambda kv: kv[1]):
            print(f"  {top:>5.1f}  {cid}")
        print(f"\nbest band {res['band'][0]}–{res['band'][1]} "
              f"({res['accuracy_at_band']:.0%} separation) -> threshold {res['confidence_threshold']}")
        if res["note"]:
            print(f"note: {res['note']}")
        if a.write:
            from .config import get_settings

            p = write_calibration(res, keep_corrective=get_settings().corrective_threshold)
            print(f"wrote {p}")
        return 0

    def show(r) -> None:
        mark = "PASS" if r.correct else "FAIL"
        hit = "" if r.retrieval_hit is None else f"  hit={r.retrieval_hit}"
        print(f"  {mark}  {r.case.bucket.value:<14} {r.case.id:<28} {r.detail}{hit}")

    report = run_eval(cases, judges=a.judge, on_case=show)
    print(f"\noverall accuracy     {_pct(report.accuracy())}")
    for b in Bucket:
        if report.rows(b):
            print(f"  {b.value:<18} {_pct(report.accuracy(b))}")
    print(f"safety compliance    {_pct(report.safety_compliance)}  (must be 100%)")
    print(f"false refusal rate   {_pct(report.false_refusal_rate)}")
    print(f"retrieval hit rate   {_pct(report.retrieval_hit_rate)}")
    if a.judge:
        print("\njudged quality (never part of accuracy):")
        for m in ("faithfulness", "completeness", "relevance"):
            score, unstable, n = report.judged(m)
            if n:
                print(f"  {m:<14} {score:.2f}  over {n} case(s), self-disagreed {unstable:.0%}")
        for r in report.results:
            j = r.judgements.get("completeness")
            if j and j.missed:
                print(f"  {r.case.id} missed: " + "; ".join(j.missed))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report.to_json(), f, indent=2, default=str)
        print(f"\nwrote {a.out}")
    failed_safety = report.safety_compliance is not None and report.safety_compliance < 1.0
    return 1 if failed_safety or (report.accuracy() or 0) < 1.0 else 0


# --- users -------------------------------------------------------------------------------


def cmd_users(a) -> int:
    from . import db
    from .passwords import check_strength

    if a.action == "list":
        rows = db.list_users()
        for u, n in rows:
            v = "verified" if u.verified else "unverified"
            print(f"  {u.email:<40} {v:<11} {n:>4} turns")
        print(f"{len(rows)} account(s)")
        return 0
    if not a.email:
        print("an email is required", file=sys.stderr)
        return 2
    if a.action == "add":
        pw = getpass.getpass("password: ")
        weak = check_strength(pw)
        if weak:
            print(weak, file=sys.stderr)
            return 1
        try:
            u = db.create_user(a.email, pw, verified=True)
        except db.EmailTaken as e:
            print(e, file=sys.stderr)
            return 1
        print(f"created {u.email}")
        return 0
    u = db.get_user_by_email(a.email)
    if u is None:
        print("no such account", file=sys.stderr)
        return 1
    db.delete_user(u.id)
    print(f"deleted {u.email} and all of its history")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag", description="Back Pain Help RAG")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus", help="manage the sha256 document allow-list")
    c.add_argument("action", choices=["scan", "verify"])
    c.add_argument("--rehash", action="store_true",
                   help="re-pin sha256 of listed files that changed (e.g. regenerated OCR)")
    c.set_defaults(fn=cmd_corpus)

    sub.add_parser("ingest", help="parse + chunk manifest documents").set_defaults(fn=cmd_ingest)
    sub.add_parser("index", help="embed chunks and build the vector index").set_defaults(fn=cmd_index)

    q = sub.add_parser("ask", help="ask a question")
    q.add_argument("query")
    q.add_argument("--trace", action="store_true")
    q.add_argument("--json", action="store_true")
    q.add_argument("--no-hyde", action="store_true")
    q.add_argument("--no-cache", action="store_true")
    q.set_defaults(fn=cmd_ask)

    e = sub.add_parser("eval", help="measure the pipeline")
    esub = e.add_subparsers(dest="action", required=True)
    er = esub.add_parser("run")
    er.add_argument("--bucket")
    er.add_argument("--judge", action="store_true", help="add LLM-judged quality (costs more)")
    er.add_argument("--out")
    ec = esub.add_parser("calibrate")
    ec.add_argument("--write", action="store_true")
    e.set_defaults(fn=cmd_eval)

    u = sub.add_parser("users", help="manage accounts")
    u.add_argument("action", choices=["list", "add", "delete"])
    u.add_argument("email", nargs="?")
    u.set_defaults(fn=cmd_users)

    s = sub.add_parser("serve", help="run the web UI + API")
    s.set_defaults(fn=lambda _a: (__import__("backpain_rag.api", fromlist=["main"]).main(), 0)[1])

    a, rest = p.parse_known_args(argv)
    if a.cmd == "serve":
        sys.argv = ["rag-serve", *rest]
    elif rest:
        p.error(f"unrecognized arguments: {' '.join(rest)}")
    from .db import StorageError
    from .index import StoreError
    from .retrieval import RetrievalError

    try:
        return a.fn(a)
    except (RetrievalError, StoreError, StorageError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
