"""FastAPI service: the JSON API, the auth pages, and the single-page UI.

One process serves both. The UI at `/` talks to the same `/api/*` endpoints any
other client would use.

Feature switches, each keyed off configuration rather than a flag of its own:
  DATABASE_URL (or AUTH_ENABLED=true)  -> accounts + per-user history
  SES_SMTP_HOST + MAIL_FROM            -> email verification + password reset
  REDIS_URL                            -> shared cache and rate-limit counters
Leave them all unset and the app runs open: no login, no history.
"""

from __future__ import annotations

import html
import threading
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from . import db, mailer, security
from .config import get_settings
from .index import load_index_manifest
from .models import Response as AnswerResponse
from .passwords import check_strength

WEB = Path(__file__).parent / "web"

app = FastAPI(title="Back Pain Help", version="0.1.0",
              description="Guideline-grounded answers about back pain, with citations.")
api = APIRouter(prefix="/api")

# --- the pipeline, built lazily and once ------------------------------------------

_assistant = None
_assistant_error: str | None = None
_assistant_lock = threading.Lock()


def get_assistant():
    global _assistant, _assistant_error
    if _assistant is None:
        with _assistant_lock:
            if _assistant is None:
                from .assistant import Assistant

                try:
                    _assistant = Assistant.build()
                    _assistant_error = None
                except Exception as e:  # noqa: BLE001
                    _assistant_error = str(e)
                    raise HTTPException(503, f"assistant unavailable: {e}") from e
    return _assistant


# --- identity -----------------------------------------------------------------------


def current_user(request: Request) -> db.User | None:
    if not get_settings().auth_required:
        return None
    token = security.read_session(request.cookies.get(security.COOKIE_NAME))
    if token is None:
        return None
    try:
        user = db.get_user(token.user_id)
    except db.StorageError as e:
        raise HTTPException(503, "account storage unavailable") from e
    # A session older than the last password change is no longer honoured:
    # that is what someone resetting because they fear compromise is asking for.
    if user is None or token.issued_at_ms < user.password_changed_at:
        return None
    return user


def require_auth(user: db.User | None = Depends(current_user)) -> None:
    if get_settings().auth_required and user is None:
        raise HTTPException(401, "sign in required")


def require_user(user: db.User | None = Depends(current_user)) -> db.User:
    if user is None:
        raise HTTPException(404 if not get_settings().auth_required else 401,
                            "accounts are not enabled" if not get_settings().auth_required
                            else "sign in required")
    return user


def require_verified(user: db.User | None = Depends(current_user)) -> None:
    if user is not None and get_settings().verification_required and not user.verified:
        raise HTTPException(403, "confirm your email address before asking questions")


def _ip(request: Request) -> str:
    return security.client_ip(request.headers, request.client.host if request.client else None)


def _quota_headers(v: security.RateVerdict) -> dict[str, str]:
    if v.limit <= 0:
        return {}
    return {"X-RateLimit-Limit": str(v.limit), "X-RateLimit-Remaining": str(v.remaining),
            "X-RateLimit-Reset": str(v.reset_in_s)}


def _subject(request: Request, user: db.User | None) -> str:
    # By account where there is one: an IP is shared by everyone behind a NAT.
    return f"user:{user.id}" if user else f"ip:{_ip(request)}"


# --- health & UI ----------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    m = load_index_manifest()
    return {"ok": True, "index": bool(m), "chunks": m["n_chunks"] if m else 0,
            "auth": get_settings().auth_required}


@app.get("/", include_in_schema=False)
def index(user: db.User | None = Depends(current_user)) -> Response:
    if get_settings().auth_required and user is None:
        return RedirectResponse("/login", status_code=303)
    return FileResponse(WEB / "index.html", media_type="text/html")


# --- auth pages -----------------------------------------------------------------------

_AUTH_CSS = """
:root{--bg:#f7f6f3;--card:#fff;--fg:#1d1d1b;--muted:#6b6a66;--line:#e3e1dc;--accent:#2f6f5e;--err:#b3261e;--ok:#2f6f5e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--fg:#ecebe7;--muted:#a3a19b;--line:#33322f;--accent:#7cc4ad;--err:#f2b8b5;--ok:#7cc4ad}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;display:grid;place-items:center;min-height:100vh;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:28px;width:100%;max-width:380px}
h1{font-size:1.25rem;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 20px;font-size:.92rem}
label{display:block;font-size:.85rem;color:var(--muted);margin:12px 0 4px}
input{width:100%;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--fg);font:inherit}
button{margin-top:18px;width:100%;padding:11px;border:0;border-radius:8px;background:var(--accent);color:var(--card);font:inherit;font-weight:600;cursor:pointer}
.msg{padding:10px 12px;border-radius:8px;font-size:.9rem;margin-bottom:12px;border:1px solid var(--line)}
.msg.err{color:var(--err)}.msg.ok{color:var(--ok)}
.links{margin-top:16px;font-size:.88rem;color:var(--muted);display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
a{color:var(--accent)}
"""


def _page(title: str, sub: str, body: str, msg: str = "", ok: bool = False, code: int = 200) -> HTMLResponse:
    note = f'<div class="msg {"ok" if ok else "err"}">{html.escape(msg)}</div>' if msg else ""
    storage_note = ""
    s = get_settings()
    if s.auth_required and not s.database_url:
        storage_note = ('<div class="msg">Accounts are stored in a local SQLite file. On a '
                        'serverless host that file is discarded on every cold start.</div>')
    page = (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)} · Back Pain Help</title><style>{_AUTH_CSS}</style></head>"
            f"<body><main class=card><h1>{html.escape(title)}</h1><p class=sub>{html.escape(sub)}</p>"
            f"{storage_note}{note}{body}</main></body></html>")
    return HTMLResponse(page, status_code=code)


def _login_form(email: str = "") -> str:
    return (f'<form method=post action=/login><label>Email</label>'
            f'<input name=email type=email required autocomplete=email value="{html.escape(email)}">'
            '<label>Password</label><input name=password type=password required '
            'autocomplete=current-password><button>Sign in</button></form>'
            '<div class=links><a href=/signup>Create an account</a><a href=/forgot>Forgot password?</a></div>')


def _signup_form(email: str = "") -> str:
    return (f'<form method=post action=/signup><label>Email</label>'
            f'<input name=email type=email required autocomplete=email value="{html.escape(email)}">'
            '<label>Password (8+ characters)</label><input name=password type=password required '
            'minlength=8 autocomplete=new-password><button>Create account</button></form>'
            '<div class=links><a href=/login>I already have an account</a></div>')


def _accounts_or_404() -> None:
    if not get_settings().auth_required:
        raise HTTPException(404, "accounts are not enabled on this instance")


def _start_session(user_id: str, to: str = "/") -> Response:
    resp = RedirectResponse(to, status_code=303)
    resp.set_cookie(security.COOKIE_NAME, security.issue_session(user_id),
                    max_age=get_settings().session_ttl_s, httponly=True, samesite="lax",
                    secure=get_settings().public_base_url.startswith("https"))
    return resp


def _base_url(request: Request) -> str:
    # Host is attacker-supplied in general; PUBLIC_BASE_URL pins it for links.
    return (get_settings().public_base_url or str(request.base_url)).rstrip("/")


def _auth_rate_limited(request: Request) -> bool:
    return not security.check_rate_limit(f"ip:{_ip(request)}", scope="auth").allowed


@app.get("/login", include_in_schema=False)
def login_page(user: db.User | None = Depends(current_user)) -> Response:
    _accounts_or_404()
    if user:
        return RedirectResponse("/", status_code=303)
    return _page("Sign in", "Back Pain Help answers from published guidelines.", _login_form())


@app.post("/login", include_in_schema=False)
def login(request: Request, email: str = Form(""), password: str = Form("")) -> Response:
    _accounts_or_404()
    if _auth_rate_limited(request):
        return _page("Sign in", "", _login_form(email), "Too many attempts. Try again later.", code=429)
    try:
        user = db.authenticate(email, password)
    except db.StorageError:
        return _page("Sign in", "", _login_form(email), "Account storage is unavailable.", code=503)
    if user is None:
        return _page("Sign in", "", _login_form(email), "Wrong email or password.", code=401)
    return _start_session(user.id)


@app.get("/signup", include_in_schema=False)
def signup_page(user: db.User | None = Depends(current_user)) -> Response:
    _accounts_or_404()
    if user:
        return RedirectResponse("/", status_code=303)
    return _page("Create an account", "Free. Questions are limited per hour.", _signup_form())


def _send_verification(request: Request, user: db.User) -> bool:
    s = get_settings()
    token = security.issue(security.PURPOSE_VERIFY, user.id, s.verify_token_ttl_s)
    try:
        mailer.send_verification(user.email, f"{_base_url(request)}/verify?token={quote(token)}")
        return True
    except mailer.MailError:
        return False


@app.post("/signup", include_in_schema=False)
def signup(request: Request, email: str = Form(""), password: str = Form("")) -> Response:
    _accounts_or_404()
    if _auth_rate_limited(request):
        return _page("Create an account", "", _signup_form(email), "Too many attempts. Try again later.", code=429)
    if "@" not in email or len(email) > 254:
        return _page("Create an account", "", _signup_form(email), "Enter a valid email address.", code=400)
    weak = check_strength(password)
    if weak:
        return _page("Create an account", "", _signup_form(email), weak, code=400)
    s = get_settings()
    try:
        user = db.create_user(email, password, verified=not s.verification_required)
    except db.EmailTaken:
        return _page("Create an account", "", _signup_form(email),
                     "An account with that email already exists.", code=409)
    except db.StorageError:
        return _page("Create an account", "", _signup_form(email), "Account storage is unavailable.", code=503)
    if s.verification_required:
        _send_verification(request, user)
    return _start_session(user.id)


@app.get("/verify", include_in_schema=False)
def verify(token: str = "") -> Response:
    _accounts_or_404()
    t = security.read(security.PURPOSE_VERIFY, token)
    if t is None:
        return _page("Link expired", "", '<div class=links><a href=/>Back to the app</a></div>',
                     "That confirmation link is invalid or has expired. Sign in to get a new one.", code=400)
    db.mark_verified(t.user_id)
    return _page("Email confirmed", "You can now ask questions.",
                 '<div class=links><a href=/>Open Back Pain Help</a></div>', ok=True)


@api.post("/resend-verification")
def resend_verification(request: Request, user: db.User = Depends(require_user)) -> dict:
    if user.verified or not get_settings().verification_required:
        return {"sent": False, "reason": "already verified"}
    if not security.check_rate_limit(f"user:{user.id}", scope="auth").allowed:
        raise HTTPException(429, "too many requests")
    if not _send_verification(request, user):
        raise HTTPException(502, "we could not send the email -- try again later")
    return {"sent": True}


_FORGOT_FORM = ('<form method=post action=/forgot><label>Email</label>'
                '<input name=email type=email required autocomplete=email>'
                '<button>Send reset link</button></form><div class=links><a href=/login>Back to sign in</a></div>')


@app.get("/forgot", include_in_schema=False)
def forgot_page() -> Response:
    _accounts_or_404()
    if not get_settings().mail_enabled:
        return _page("Reset password", "", "", "Password reset needs email, which is not configured "
                     "on this instance. Ask the administrator (`rag users ...`).")
    return _page("Reset password", "We'll email you a link.", _FORGOT_FORM)


@app.post("/forgot", include_in_schema=False)
def forgot(request: Request, email: str = Form("")) -> Response:
    _accounts_or_404()
    s = get_settings()
    # Identical answer whether or not the address exists: not an enumeration endpoint.
    done = _page("Check your email", "", '<div class=links><a href=/login>Back to sign in</a></div>',
                 "If an account exists for that address, a reset link is on its way.", ok=True)
    if not s.mail_enabled or _auth_rate_limited(request):
        return done
    try:
        user = db.get_user_by_email(email)
    except db.StorageError:
        return done
    if user:
        token = security.issue(security.PURPOSE_RESET, user.id, s.reset_token_ttl_s)
        try:
            mailer.send_reset(user.email, f"{_base_url(request)}/reset?token={quote(token)}")
        except mailer.MailError:
            pass
    return done


def _reset_target(token: str) -> db.User | None:
    t = security.read(security.PURPOSE_RESET, token)
    if t is None:
        return None
    user = db.get_user(t.user_id)
    # Single use: using the link stamps password_changed_at, after which a token
    # issued earlier no longer resolves. A revocation list we don't have to keep.
    if user is None or t.issued_at_ms < user.password_changed_at:
        return None
    return user


def _reset_form(token: str) -> str:
    return (f'<form method=post action=/reset><input type=hidden name=token value="{html.escape(token)}">'
            '<label>New password (8+ characters)</label><input name=password type=password required '
            'minlength=8 autocomplete=new-password><button>Set password</button></form>')


@app.get("/reset", include_in_schema=False)
def reset_page(token: str = "") -> Response:
    _accounts_or_404()
    if _reset_target(token) is None:
        return _page("Link expired", "", '<div class=links><a href=/forgot>Request a new link</a></div>',
                     "That reset link is invalid, expired, or already used.", code=400)
    return _page("Choose a new password", "This signs you out everywhere else.", _reset_form(token))


@app.post("/reset", include_in_schema=False)
def reset(token: str = Form(""), password: str = Form("")) -> Response:
    _accounts_or_404()
    user = _reset_target(token)
    if user is None:
        return _page("Link expired", "", '<div class=links><a href=/forgot>Request a new link</a></div>',
                     "That reset link is invalid, expired, or already used.", code=400)
    weak = check_strength(password)
    if weak:
        return _page("Choose a new password", "", _reset_form(token), weak, code=400)
    db.set_password(user.id, password)
    db.mark_verified(user.id)  # receiving the link proves the address
    return _start_session(user.id)


@app.post("/logout", include_in_schema=False)
def logout() -> Response:
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie(security.COOKIE_NAME)
    return resp


# --- JSON API ------------------------------------------------------------------------


@api.get("/info", dependencies=[Depends(require_auth)])
def info(request: Request, user: db.User | None = Depends(current_user)) -> dict:
    s = get_settings()
    quota = security.check_rate_limit(_subject(request, user), consume=False)
    m = load_index_manifest()
    return {
        "auth": s.auth_required,
        "user": ({"email": user.email, "verified": user.verified} if user else None),
        "verification_required": s.verification_required,
        "quota": {"limit": quota.limit, "remaining": quota.remaining, "reset_in_s": quota.reset_in_s},
        "corpus": ({"documents": m["documents"], "n_chunks": m["n_chunks"], "built_at": m["built_at"],
                    "embed_model": m["embed_model"]} if m else None),
        "thresholds": {"confidence": s.confidence_threshold, "corrective": s.corrective_threshold},
    }


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)


@api.post("/ask", response_model=AnswerResponse,
          dependencies=[Depends(require_auth), Depends(require_verified)])
def ask(req: AskRequest, request: Request, user: db.User | None = Depends(current_user)) -> JSONResponse:
    # Verification is checked before the limiter, so a blocked question never spends a slot.
    quota = security.check_rate_limit(_subject(request, user))
    headers = _quota_headers(quota)
    if not quota.allowed:
        return JSONResponse(
            {"detail": f"You've reached the limit of {quota.limit} questions per hour.",
             "retry_after_s": quota.reset_in_s},
            status_code=429, headers=headers | {"Retry-After": str(quota.reset_in_s)})
    response = get_assistant().ask(req.query)
    if user is not None:
        try:
            db.save_turn(user.id, response.query, response.answer, response.answered,
                         response.refusal_reason.value if response.refusal_reason else None,
                         response.top_score, [c.model_dump() for c in response.citations])
        except db.StorageError:
            response.trace["history"] = "not saved: storage unavailable"
    return JSONResponse(response.model_dump(mode="json"), headers=headers)


def _turn_json(t: db.Turn, full: bool = False) -> dict:
    d = {"id": t.id, "query": t.query, "answered": t.answered, "created_at": t.created_at}
    if full:
        d |= {"answer": t.answer, "refusal_reason": t.refusal_reason, "top_score": t.top_score,
              "citations": t.citations}
    return d


@api.get("/history")
def list_history(before: int | None = None, limit: int = 20,
                 user: db.User = Depends(require_user)) -> dict:
    turns = db.list_turns(user.id, limit=max(1, min(limit, 100)), before=before)
    return {"items": [_turn_json(t) for t in turns],
            "next_before": turns[-1].created_at if len(turns) == limit else None}


@api.get("/history/{turn_id}")
def get_history_item(turn_id: str, user: db.User = Depends(require_user)) -> dict:
    t = db.get_turn(user.id, turn_id)
    if t is None:
        raise HTTPException(404, "not found")
    return _turn_json(t, full=True)


@api.delete("/history/{turn_id}")
def delete_history_item(turn_id: str, user: db.User = Depends(require_user)) -> dict:
    if not db.delete_turn(user.id, turn_id):
        raise HTTPException(404, "not found")
    return {"deleted": 1}


@api.delete("/history")
def clear_history(user: db.User = Depends(require_user)) -> dict:
    return {"deleted": db.clear_turns(user.id)}


@app.exception_handler(db.StorageError)
def _storage_error(_: Request, e: db.StorageError) -> JSONResponse:
    return JSONResponse({"detail": "account storage unavailable"}, status_code=503)


app.include_router(api)


def main() -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="Serve the Back Pain Help UI and API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    a = p.parse_args()
    uvicorn.run("backpain_rag.api:app", host=a.host, port=a.port, reload=a.reload)


if __name__ == "__main__":
    main()
