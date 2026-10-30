"""Password hashing with stdlib scrypt -- memory-hard, and no dependency to patch.

Each stored hash carries the cost parameters it was made with, so raising
SCRYPT_N later re-hashes people as they sign in instead of locking them out.
Format: scrypt$n$r$p$salt_hex$hash_hex
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from .config import get_settings

MIN_LENGTH = 8


def hash_password(password: str) -> str:
    s = get_settings()
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=s.scrypt_n, r=s.scrypt_r,
                        p=s.scrypt_p, maxmem=256 * 1024 * 1024, dklen=32)
    return f"scrypt${s.scrypt_n}${s.scrypt_r}${s.scrypt_p}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_hex, hash_hex = stored.split("$")
        if algo != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=int(n),
                            r=int(r), p=int(p), maxmem=256 * 1024 * 1024,
                            dklen=len(hash_hex) // 2)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk.hex(), hash_hex)


def needs_rehash(stored: str) -> bool:
    s = get_settings()
    try:
        _, n, r, p, *_ = stored.split("$")
        return (int(n), int(r), int(p)) != (s.scrypt_n, s.scrypt_r, s.scrypt_p)
    except ValueError:
        return True


def check_strength(password: str) -> str | None:
    """A reason the password is unacceptable, or None."""
    if len(password) < MIN_LENGTH:
        return f"Password must be at least {MIN_LENGTH} characters."
    if len(password) > 256:
        return "Password is too long."
    return None
