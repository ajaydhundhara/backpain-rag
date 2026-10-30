import time

import pytest
from fastapi.testclient import TestClient

from backpain_rag import api, config, db, security
from backpain_rag.models import Response
from backpain_rag.passwords import hash_password, needs_rehash, verify_password


def test_password_roundtrip_and_rehash(monkeypatch):
    h = hash_password("correct horse")
    assert verify_password("correct horse", h) and not verify_password("wrong", h)
    monkeypatch.setenv("SCRYPT_N", "2048")
    config.get_settings.cache_clear()
    assert needs_rehash(h)


def test_tokens_are_purpose_bound_and_expire():
    uid = "a" * 32
    t = security.issue(security.PURPOSE_VERIFY, uid, 60)
    assert security.read(security.PURPOSE_VERIFY, t).user_id == uid
    assert security.read(security.PURPOSE_RESET, t) is None  # wrong purpose
    assert security.read(security.PURPOSE_VERIFY, t, now=time.time() + 120) is None
    assert security.read(security.PURPOSE_VERIFY, t[:-1] + "0") is None


def test_users_and_history_are_scoped():
    a = db.create_user("A@example.com", "password1", verified=True)
    b = db.create_user("b@example.com", "password1", verified=True)
    with pytest.raises(db.EmailTaken):
        db.create_user("a@example.com", "x" * 8)
    assert db.authenticate("a@example.com", "password1").id == a.id
    assert db.authenticate("a@example.com", "nope") is None
    tid = db.save_turn(a.id, "q", "ans", True, None, 9.0, [{"marker": "C1"}])
    assert db.get_turn(b.id, tid) is None  # another account cannot read it
    assert db.get_turn(a.id, tid).citations == [{"marker": "C1"}]
    assert not db.delete_turn(b.id, tid) and db.delete_turn(a.id, tid)


class _FakeAssistant:
    def ask(self, query, screen=True, use_cache=True):
        return Response(query=query, answered=True, answer="ok [C1]", disclaimer="d")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "2")
    config.get_settings.cache_clear()
    monkeypatch.setattr(api, "_assistant", _FakeAssistant())
    return TestClient(api.app)


def test_open_instance_needs_no_login(monkeypatch):
    monkeypatch.setattr(api, "_assistant", _FakeAssistant())
    c = TestClient(api.app)
    assert c.get("/").status_code == 200
    assert c.post("/api/ask", json={"query": "what is sciatica"}).json()["answered"]


def test_signup_ask_history_and_quota(client):
    assert client.get("/", follow_redirects=False).status_code == 303
    assert client.post("/api/ask", json={"query": "what is sciatica"}).status_code == 401
    r = client.post("/signup", data={"email": "u@example.com", "password": "password1"},
                    follow_redirects=False)
    assert r.status_code == 303 and security.COOKIE_NAME in r.cookies
    r1 = client.post("/api/ask", json={"query": "what is sciatica"})
    assert r1.status_code == 200 and r1.headers["X-RateLimit-Remaining"] == "1"
    client.post("/api/ask", json={"query": "what is lumbago"})
    r3 = client.post("/api/ask", json={"query": "what is a disc"})
    assert r3.status_code == 429 and "Retry-After" in r3.headers
    items = client.get("/api/history").json()["items"]
    assert [i["query"] for i in items] == ["what is lumbago", "what is sciatica"]
    assert client.delete("/api/history").json()["deleted"] == 2


def test_password_change_invalidates_old_sessions(client):
    client.post("/signup", data={"email": "v@example.com", "password": "password1"})
    old = client.cookies.get(security.COOKIE_NAME)
    user = db.get_user_by_email("v@example.com")
    time.sleep(0.002)
    db.set_password(user.id, "password2")
    c2 = TestClient(api.app, cookies={security.COOKIE_NAME: old})
    assert c2.get("/api/info").status_code == 401


def test_reset_link_is_single_use(client):
    u = db.create_user("w@example.com", "password1", verified=True)
    token = security.issue(security.PURPOSE_RESET, u.id, 3600)
    time.sleep(0.002)
    assert client.post("/reset", data={"token": token, "password": "password9"},
                       follow_redirects=False).status_code == 303
    assert client.post("/reset", data={"token": token, "password": "password8"}).status_code == 400
