"""Sessions, email-link tokens, and the spend cap.

These protect different things and fail in opposite directions:

* The **session** is a security boundary and fails closed: a request that
  cannot prove which user it belongs to is refused.
* The **rate limit** is a cost control and fails open: if Redis is down we
  serve the request (falling back to a per-process counter) rather than let a
  limiter take the app down. The real backstop against a runaway bill is a hard
  monthly limit on a project-scoped OpenAI key.

The cookie is a signed statement of identity ("this is user X") carrying no
privileges, so revoking an account is a database delete, not a key rotation.
Nothing is stored server-side for tokens.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sys
import time
from dataclasses import dataclass

from .cache import get_cache
from .config import get_settings

COOKIE_NAME = "bp_session"
_USER_ID_RE = re.compile(r"^[0-9a-f]{32}$")

PURPOSE_SESSION = "session"
PURPOSE_VERIFY = "verify-email"
PURPOSE_RESET = "reset-password"

_ephemeral_secret: str | None = None


def _secret() -> str:
    """Without SESSION_SECRET, sign with a random per-process key. That fails
    *visibly* -- everyone is signed out at each restart -- which beats signing
    with something guessable."""
    global _ephemeral_secret
    configured = get_settings().session_secret
    if configured:
        return configured
    if _ephemeral_secret is None:
        _ephemeral_secret = secrets.token_hex(32)
        print("warning: SESSION_SECRET is not set; sessions will not survive a restart",
              file=sys.stderr)
    return _ephemeral_secret


def reset_secret() -> None:
    global _ephemeral_secret
    _ephemeral_secret = None


def _sign(*parts: object) -> str:
    return hmac.new(_secret().encode(), ".".join(map(str, parts)).encode(),
                    hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class Token:
    user_id: str
    # Milliseconds: a password reset stamps the row and then issues a new
    # session microseconds later, and the two must be ordered correctly.
    issued_at_ms: int


def issue(purpose: str, user_id: str, ttl_s: int, now: float | None = None) -> str:
    """The purpose is inside the signature, so a verification link -- handed to
    an address nobody has proven yet -- can never be used as a reset link."""
    now = time.time() if now is None else now
    issued_ms = int(now * 1000)
    expires = int(now) + ttl_s
    return f"{user_id}.{issued_ms}.{expires}.{_sign(purpose, user_id, issued_ms, expires)}"


def read(purpose: str, token: str | None, now: float | None = None) -> Token | None:
    """Every rejection (malformed, forged, wrong purpose, expired) is None."""
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 4:
        return None
    user_id, raw_issued, raw_exp, sig = parts
    if not _USER_ID_RE.match(user_id):
        return None
    try:
        issued_ms, expires = int(raw_issued), int(raw_exp)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(purpose, user_id, issued_ms, expires)):
        return None
    if (time.time() if now is None else now) >= expires:
        return None
    return Token(user_id, issued_ms)


def issue_session(user_id: str) -> str:
    return issue(PURPOSE_SESSION, user_id, get_settings().session_ttl_s)


def read_session(token: str | None) -> Token | None:
    return read(PURPOSE_SESSION, token)


# --- rate limiting -------------------------------------------------------------


@dataclass
class RateVerdict:
    allowed: bool
    limit: int
    remaining: int
    reset_in_s: int


def client_ip(headers, fallback: str | None) -> str:
    """First x-forwarded-for entry is the original client; the last is the proxy."""
    if get_settings().trust_proxy_header:
        fwd = headers.get("x-forwarded-for") or headers.get("x-real-ip")
        if fwd:
            return fwd.split(",")[0].strip()
    return fallback or "unknown"


def check_rate_limit(subject: str, scope: str = "ask", consume: bool = True) -> RateVerdict:
    """Count one request against `subject` (user id, or IP before sign-in).

    Scopes have separate counters: questions are metered because they cost
    money, sign-in because guessing passwords is free -- sharing a bucket would
    let failed logins eat the question quota. The subject is hashed so the
    counter store is never a log of who asked medical questions.
    """
    s = get_settings()
    limit = s.rate_limit_per_window if scope == "ask" else s.auth_rate_limit_per_window
    window = s.rate_limit_window_s
    if limit <= 0:
        return RateVerdict(True, 0, -1, 0)
    now = int(time.time())
    bucket = now // window
    reset_in = (bucket + 1) * window - now
    digest = hashlib.sha256(f"{scope}|{subject}".encode()).hexdigest()[:24]
    key = f"rl:{scope}:{bucket}:{digest}"
    cache = get_cache()
    if consume:
        n = cache.incr(key, window + 5)
    else:
        n = int(cache.get_json(key) or 0)
    return RateVerdict(n <= limit, limit, max(0, limit - n), reset_in)
