"""Central configuration. Every tunable the pipeline and eval harness touch lives here.

Thresholds measured by `rag eval calibrate --write` are loaded from
data/calibration.json on top of these defaults, so a recalibration never needs
a code change.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- OpenAI -------------------------------------------------------------
    openai_api_key: str = ""
    #: Writes the user-facing answer.
    openai_model: str = "gpt-4o-mini"
    #: Intent classifier, reranker, HyDE, support/framing judges. High volume.
    openai_guard_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536

    # --- Index ----------------------------------------------------------------
    index_version: str = "v1"
    chunk_target_tokens: int = 320
    chunk_overlap_tokens: int = 48
    chunk_min_tokens: int = 60
    chunk_drop_below_tokens: int = 10

    # --- Retrieval + gates ----------------------------------------------------
    retrieve_k: int = 20
    rerank_top_n: int = 6
    #: Gate 2: top-1 rerank score needed to answer.
    confidence_threshold: float = 7.5
    #: Gate 2 corrective band: [corrective_threshold, confidence_threshold).
    corrective_enabled: bool = True
    corrective_threshold: float = 6.0
    corrective_k: int = 40
    corrective_hyde_query_weight: float = 0.2

    hyde_enabled: bool = True
    hyde_n: int = 1
    hyde_query_weight: float = 0.5
    hyde_model: str | None = None
    hyde_version: str = "v1"

    #: Hand-bumped whenever gate *logic* or a prompt changes. The fingerprint
    #: can see configuration but not code, so this is its stand-in for code.
    guardrails_version: str = "v2"

    # --- Cache (Redis, optional) ----------------------------------------------
    redis_url: str = ""
    redis_timeout_ms: int = 400
    cache_embed_ttl_s: int = 30 * 24 * 3600
    cache_response_ttl_s: int = 24 * 3600
    cache_hyde_ttl_s: int = 7 * 24 * 3600

    # --- Accounts (Postgres or SQLite) ----------------------------------------
    database_url: str = ""
    sqlite_path: str = ""
    db_max_connections: int = 4
    db_timeout_s: float = 10.0
    auth_enabled: bool | None = None
    session_secret: str = ""
    session_ttl_s: int = 7 * 24 * 3600
    scrypt_n: int = 2**14
    scrypt_r: int = 8
    scrypt_p: int = 1
    history_max_items: int = 200

    # --- Rate limiting --------------------------------------------------------
    rate_limit_per_window: int = 5
    rate_limit_window_s: int = 3600
    auth_rate_limit_per_window: int = 20
    trust_proxy_header: bool = True

    # --- Email (SES over SMTP, optional) --------------------------------------
    ses_smtp_host: str = ""
    ses_smtp_port: int = 587
    ses_smtp_user: str = ""
    ses_smtp_password: str = ""
    mail_from: str = ""
    mail_from_name: str = "Back Pain Help"
    mail_timeout_s: float = 10.0
    public_base_url: str = ""
    verify_token_ttl_s: int = 3 * 24 * 3600
    reset_token_ttl_s: int = 3600

    # --- Derived --------------------------------------------------------------

    @property
    def mail_enabled(self) -> bool:
        """An instance that cannot send must not demand a link it will never
        deliver, so verification keys off this too."""
        return bool(self.ses_smtp_host and self.mail_from)

    @property
    def verification_required(self) -> bool:
        return self.mail_enabled

    @property
    def auth_required(self) -> bool:
        if self.auth_enabled is not None:
            return self.auth_enabled
        return bool(self.database_url)

    @property
    def pipeline_fingerprint(self) -> str:
        """Everything that can change an answer for a fixed query.

        Cached responses are namespaced by this, so switching model, rebuilding
        the index or recalibrating gate 2 makes old entries unreachable rather
        than stale.
        """
        material = "|".join(
            str(x)
            for x in (
                self.openai_model, self.openai_guard_model, self.openai_embed_model,
                self.embed_dim, self.index_version, self.retrieve_k, self.rerank_top_n,
                self.confidence_threshold, self.guardrails_version,
                self.hyde_enabled, self.hyde_n, self.hyde_query_weight, self.hyde_model,
                self.hyde_version, self.corrective_enabled, self.corrective_threshold,
                self.corrective_k, self.corrective_hyde_query_weight,
                self.chunk_target_tokens, self.chunk_overlap_tokens,
            )
        )
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    # --- Paths ----------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def eval_path(self) -> Path:
        return self.data_dir / "eval" / "questions.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "corpus_manifest.yaml"

    @property
    def index_manifest_path(self) -> Path:
        return self.data_dir / "index_manifest.json"

    @property
    def calibration_path(self) -> Path:
        return self.data_dir / "calibration.json"

    @property
    def sqlite_file(self) -> Path:
        return Path(self.sqlite_path) if self.sqlite_path else self.data_dir / "app.db"

    def load_calibration(self) -> None:
        if not self.calibration_path.exists():
            return
        data = json.loads(self.calibration_path.read_text())
        if "confidence_threshold" in data:
            self.confidence_threshold = float(data["confidence_threshold"])
        if "corrective_threshold" in data:
            self.corrective_threshold = float(data["corrective_threshold"])


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.load_calibration()
    return s
