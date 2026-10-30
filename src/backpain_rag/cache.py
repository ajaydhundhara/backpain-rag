"""Read-through cache. Redis when configured, an in-process dict otherwise.

A cache must never be the reason a request fails or slows down, so every
operation here fails *open*: an unreachable Redis means "miss" on read and
"skip" on write, with a short timeout so a hung connection cannot add latency.
This is the opposite failure mode to the accounts database (db.py), which
fails closed because it is a security boundary and this is an optimisation.

Nothing user-identifying goes in here: keys are a pipeline fingerprint plus a
hash of the normalised question, so two people asking the same guideline
question share one answer and the cache is never a log of who asked what.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from functools import lru_cache
from typing import Any

from .config import get_settings


def key_for(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:32]


class MemoryCache:
    """Per-process TTL cache. Bounded so a long-running server cannot grow forever."""

    def __init__(self, max_items: int = 5000) -> None:
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()
        self.max_items = max_items

    def get_json(self, key: str) -> Any | None:
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            expires, raw = hit
            if expires < time.time():
                self._data.pop(key, None)
                return None
        return json.loads(raw)

    def set_json(self, key: str, value: Any, ttl_s: int) -> None:
        with self._lock:
            if len(self._data) >= self.max_items:
                # Evict the oldest-expiring tenth. Crude, but bounded and cheap.
                for k, _ in sorted(self._data.items(), key=lambda kv: kv[1][0])[
                    : self.max_items // 10
                ]:
                    self._data.pop(k, None)
            self._data[key] = (time.time() + ttl_s, json.dumps(value))

    def incr(self, key: str, ttl_s: int) -> int:
        with self._lock:
            expires, raw = self._data.get(key, (0.0, "0"))
            if expires < time.time():
                expires, raw = time.time() + ttl_s, "0"
            n = int(raw) + 1
            self._data[key] = (expires, str(n))
            return n

    def delete_prefix(self, prefix: str) -> None:
        with self._lock:
            for k in [k for k in self._data if k.startswith(prefix)]:
                self._data.pop(k, None)


class RedisCache:
    """Redis-backed, with a MemoryCache fallback for when Redis is unreachable."""

    def __init__(self, url: str, timeout_ms: int) -> None:
        import redis

        t = timeout_ms / 1000
        self._r = redis.Redis.from_url(url, socket_timeout=t, socket_connect_timeout=t)
        self._fallback = MemoryCache()
        self._warned = False

    def _warn(self, e: Exception) -> None:
        if not self._warned:
            self._warned = True
            print(f"warning: redis unavailable, using in-process cache ({e})", file=sys.stderr)

    def get_json(self, key: str) -> Any | None:
        try:
            raw = self._r.get(key)
        except Exception as e:  # noqa: BLE001 -- any failure is a miss
            self._warn(e)
            return self._fallback.get_json(key)
        return None if raw is None else json.loads(raw)

    def set_json(self, key: str, value: Any, ttl_s: int) -> None:
        try:
            self._r.set(key, json.dumps(value), ex=ttl_s)
        except Exception as e:  # noqa: BLE001
            self._warn(e)
            self._fallback.set_json(key, value, ttl_s)

    def incr(self, key: str, ttl_s: int) -> int:
        try:
            pipe = self._r.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl_s, nx=True)
            n, _ = pipe.execute()
            return int(n)
        except Exception as e:  # noqa: BLE001
            self._warn(e)
            return self._fallback.incr(key, ttl_s)

    def delete_prefix(self, prefix: str) -> None:
        try:
            for k in self._r.scan_iter(match=f"{prefix}*", count=200):
                self._r.delete(k)
        except Exception as e:  # noqa: BLE001
            self._warn(e)
        self._fallback.delete_prefix(prefix)


@lru_cache(maxsize=1)
def get_cache() -> MemoryCache | RedisCache:
    s = get_settings()
    if s.redis_url:
        try:
            return RedisCache(s.redis_url, s.redis_timeout_ms)
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not configure redis ({e}); using in-process cache",
                  file=sys.stderr)
    return MemoryCache()
