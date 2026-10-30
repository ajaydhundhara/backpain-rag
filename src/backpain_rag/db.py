"""Accounts and question history. Postgres when DATABASE_URL is set, SQLite otherwise.

This store is a security boundary, so it fails CLOSED: any database error
surfaces as `StorageError` and the API answers 503 rather than guessing who a
request belongs to. (The cache is the opposite: it fails open. See cache.py.)

One seam for both backends: statements are written once with `?` placeholders
and translated for psycopg. No ORM -- two tables, every statement visible here.
Schema is `CREATE TABLE IF NOT EXISTS`, applied on first use; later columns go
in MIGRATIONS, which runs in order on connect. Never edit a migration that has
shipped -- a deployed database has already run it.

`User` has no password_hash field on purpose: the hash is read inside
`authenticate` and never leaves it, so no route can hand one to a browser.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from .config import get_settings
from .passwords import hash_password, needs_rehash, verify_password


class StorageError(RuntimeError):
    pass


class EmailTaken(StorageError):
    pass


SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at BIGINT NOT NULL,
        verified_at BIGINT,
        password_changed_at BIGINT NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS history (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        query TEXT NOT NULL,
        answer TEXT NOT NULL,
        answered BOOLEAN NOT NULL,
        refusal_reason TEXT,
        top_score DOUBLE PRECISION,
        citations TEXT NOT NULL,
        created_at BIGINT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS history_user_time ON history (user_id, created_at DESC)",
]

#: Append-only. Each runs on every connect, so make them idempotent.
MIGRATIONS: list[str] = []


@dataclass(frozen=True)
class User:
    id: str
    email: str
    created_at: int
    verified_at: int | None
    password_changed_at: int  # milliseconds

    @property
    def verified(self) -> bool:
        return self.verified_at is not None


@dataclass(frozen=True)
class Turn:
    id: str
    query: str
    answer: str
    answered: bool
    refusal_reason: str | None
    top_score: float | None
    citations: list[dict]
    created_at: int


def _now_ms() -> int:
    return int(time.time() * 1000)


# --- engine -----------------------------------------------------------------------


class _Engine:
    def __init__(self) -> None:
        s = get_settings()
        self.postgres = bool(s.database_url)
        self._ready = False
        self._lock = threading.Lock()
        self._pool = None
        if self.postgres:
            from psycopg_pool import ConnectionPool

            # min_size=0: a cold start that never serves a request opens nothing.
            # prepare_threshold=None: server-side prepared statements break under
            # PgBouncer transaction pooling (Neon/Supabase poolers).
            self._pool = ConnectionPool(
                s.database_url, min_size=0, max_size=s.db_max_connections,
                timeout=s.db_timeout_s, open=False,
                kwargs={"prepare_threshold": None, "autocommit": False},
            )
        else:
            self._path = s.sqlite_file
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def _sql(self, stmt: str) -> str:
        return stmt.replace("?", "%s") if self.postgres else stmt

    @contextmanager
    def conn(self) -> Iterator[Any]:
        try:
            if self.postgres:
                if self._pool.closed:
                    self._pool.open(wait=False)
                with self._pool.connection() as c:
                    self._ensure(c)
                    yield c
            else:
                c = sqlite3.connect(self._path, timeout=get_settings().db_timeout_s)
                try:
                    c.execute("PRAGMA foreign_keys = ON")
                    self._ensure(c)
                    yield c
                    c.commit()
                finally:
                    c.close()
        except StorageError:
            raise
        except Exception as e:  # noqa: BLE001 -- fail closed, with a named error
            raise StorageError(f"database unavailable: {e}") from e

    def _ensure(self, c) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            for stmt in SCHEMA + MIGRATIONS:
                c.execute(stmt)
            if self.postgres:
                c.commit()
            self._ready = True

    def execute(self, stmt: str, params: tuple = ()) -> int:
        with self.conn() as c:
            cur = c.execute(self._sql(stmt), params)
            n = cur.rowcount
            if self.postgres:
                c.commit()
            return n

    def fetchall(self, stmt: str, params: tuple = ()) -> list[tuple]:
        with self.conn() as c:
            return list(c.execute(self._sql(stmt), params).fetchall())

    def fetchone(self, stmt: str, params: tuple = ()) -> tuple | None:
        rows = self.fetchall(stmt, params)
        return rows[0] if rows else None


_engine: _Engine | None = None


def engine() -> _Engine:
    global _engine
    if _engine is None:
        _engine = _Engine()
    return _engine


def reset_engine() -> None:
    """Forget the engine (tests, or after changing settings)."""
    global _engine
    _engine = None


def _is_unique_violation(e: Exception) -> bool:
    msg = str(e).lower()
    return "unique" in msg or "duplicate key" in msg


# --- users ------------------------------------------------------------------------

_USER_COLS = "id, email, created_at, verified_at, password_changed_at"


def _user(row: tuple | None) -> User | None:
    return None if row is None else User(*row)


def normalise_email(email: str) -> str:
    return email.strip().lower()


def create_user(email: str, password: str, verified: bool = False) -> User:
    uid = uuid.uuid4().hex
    now = _now_ms()
    try:
        engine().execute(
            "INSERT INTO users (id, email, password_hash, created_at, verified_at, "
            "password_changed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (uid, normalise_email(email), hash_password(password), now,
             now if verified else None, 0),
        )
    except StorageError as e:
        if _is_unique_violation(e):
            raise EmailTaken("an account with that email already exists") from e
        raise
    return get_user(uid)  # type: ignore[return-value]


def get_user(user_id: str) -> User | None:
    return _user(engine().fetchone(f"SELECT {_USER_COLS} FROM users WHERE id = ?", (user_id,)))


def get_user_by_email(email: str) -> User | None:
    return _user(engine().fetchone(f"SELECT {_USER_COLS} FROM users WHERE email = ?",
                                   (normalise_email(email),)))


_DUMMY: str | None = None


def _dummy_hash() -> str:
    global _DUMMY
    if _DUMMY is None:
        _DUMMY = hash_password("not-a-real-password")
    return _DUMMY


def authenticate(email: str, password: str) -> User | None:
    row = engine().fetchone(f"SELECT {_USER_COLS}, password_hash FROM users WHERE email = ?",
                            (normalise_email(email),))
    if row is None:
        # Spend the same work as a real check so timing does not reveal accounts.
        verify_password(password, _dummy_hash())
        return None
    *cols, stored = row
    if not verify_password(password, stored):
        return None
    if needs_rehash(stored):
        engine().execute("UPDATE users SET password_hash = ? WHERE id = ?",
                         (hash_password(password), cols[0]))
    return User(*cols)


def mark_verified(user_id: str) -> None:
    engine().execute("UPDATE users SET verified_at = ? WHERE id = ? AND verified_at IS NULL",
                     (_now_ms(), user_id))


def set_password(user_id: str, password: str) -> int:
    """Returns the password_changed_at stamp (ms). Stamping it invalidates every
    session and reset link issued before this moment."""
    now = _now_ms()
    engine().execute("UPDATE users SET password_hash = ?, password_changed_at = ? WHERE id = ?",
                     (hash_password(password), now, user_id))
    return now


def delete_user(user_id: str) -> None:
    engine().execute("DELETE FROM history WHERE user_id = ?", (user_id,))
    engine().execute("DELETE FROM users WHERE id = ?", (user_id,))


def list_users() -> list[tuple[User, int]]:
    rows = engine().fetchall(
        f"SELECT {', '.join('u.' + c.strip() for c in _USER_COLS.split(','))}, "
        "COUNT(h.id) FROM users u LEFT JOIN history h ON h.user_id = u.id "
        "GROUP BY u.id ORDER BY u.created_at"
    )
    return [(User(*r[:5]), int(r[5])) for r in rows]


# --- history ------------------------------------------------------------------------

_TURN_COLS = "id, query, answer, answered, refusal_reason, top_score, citations, created_at"


def _turn(row: tuple) -> Turn:
    tid, q, a, answered, reason, top, cites, created = row
    return Turn(tid, q, a, bool(answered), reason, top, json.loads(cites), created)


def save_turn(user_id: str, query: str, answer: str, answered: bool,
              refusal_reason: str | None, top_score: float | None, citations: list[dict]) -> str:
    """The trace is NOT stored -- it is debugging data and the bulk of a
    response. Citations ARE: an answer without its sources is worth nothing in
    a system whose whole claim is that every statement is sourced."""
    tid = uuid.uuid4().hex
    engine().execute(
        f"INSERT INTO history ({_TURN_COLS.replace('id, ', 'id, user_id, ', 1)}) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (tid, user_id, query, answer, answered, refusal_reason, top_score,
         json.dumps(citations), _now_ms()),
    )
    cap = get_settings().history_max_items
    if cap > 0:
        engine().execute(
            "DELETE FROM history WHERE user_id = ? AND id NOT IN ("
            "SELECT id FROM history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?)",
            (user_id, user_id, cap),
        )
    return tid


def list_turns(user_id: str, limit: int = 20, before: int | None = None) -> list[Turn]:
    if before is None:
        rows = engine().fetchall(
            f"SELECT {_TURN_COLS} FROM history WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT ?", (user_id, limit))
    else:
        rows = engine().fetchall(
            f"SELECT {_TURN_COLS} FROM history WHERE user_id = ? AND created_at < ? "
            "ORDER BY created_at DESC LIMIT ?", (user_id, before, limit))
    return [_turn(r) for r in rows]


def get_turn(user_id: str, turn_id: str) -> Turn | None:
    # user_id in the WHERE clause: one account can never read another's turn,
    # even with a guessed id.
    row = engine().fetchone(f"SELECT {_TURN_COLS} FROM history WHERE id = ? AND user_id = ?",
                            (turn_id, user_id))
    return None if row is None else _turn(row)


def delete_turn(user_id: str, turn_id: str) -> bool:
    return engine().execute("DELETE FROM history WHERE id = ? AND user_id = ?",
                            (turn_id, user_id)) > 0


def clear_turns(user_id: str) -> int:
    return engine().execute("DELETE FROM history WHERE user_id = ?", (user_id,))
