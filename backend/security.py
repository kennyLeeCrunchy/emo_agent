"""Security controls for exposing the local backend through a trusted proxy."""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque

from fastapi import Request


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SecuritySettings:
    public_mode: bool = False
    proxy_secret: str | None = None
    max_upload_bytes: int = 20 * 1024 * 1024
    requests_per_minute: int = 60
    expensive_requests_per_minute: int = 6
    max_concurrent_analysis: int = 1
    llm_upstream_calls_per_day: int = 100
    llm_upstream_calls_per_user_per_day: int = 20
    llm_max_tool_rounds: int = 2
    llm_web_search_cost: int = 4
    llm_budget_db: str = "backend/data/emo_agent_llm_budget.sqlite3"

    @classmethod
    def from_env(cls) -> "SecuritySettings":
        return cls(
            public_mode=_env_bool("EMO_AGENT_PUBLIC_MODE"),
            proxy_secret=os.environ.get("EMO_AGENT_PROXY_SECRET") or None,
            max_upload_bytes=int(os.environ.get("EMO_AGENT_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))),
            requests_per_minute=int(os.environ.get("EMO_AGENT_REQUESTS_PER_MINUTE", "60")),
            expensive_requests_per_minute=int(os.environ.get("EMO_AGENT_EXPENSIVE_REQUESTS_PER_MINUTE", "6")),
            max_concurrent_analysis=int(os.environ.get("EMO_AGENT_MAX_CONCURRENT_ANALYSIS", "1")),
            llm_upstream_calls_per_day=int(os.environ.get("EMO_AGENT_LLM_UPSTREAM_CALLS_PER_DAY", "100")),
            llm_upstream_calls_per_user_per_day=int(os.environ.get("EMO_AGENT_LLM_UPSTREAM_CALLS_PER_USER_PER_DAY", "20")),
            llm_max_tool_rounds=int(os.environ.get("EMO_AGENT_LLM_MAX_TOOL_ROUNDS", "2")),
            llm_web_search_cost=int(os.environ.get("EMO_AGENT_LLM_WEB_SEARCH_COST", "4")),
            llm_budget_db=os.environ.get("EMO_AGENT_LLM_BUDGET_DB", "backend/data/emo_agent_llm_budget.sqlite3"),
        )

    def validate(self) -> None:
        if self.public_mode and (not self.proxy_secret or len(self.proxy_secret) < 32):
            raise RuntimeError("EMO_AGENT_PROXY_SECRET must contain at least 32 characters in public mode")
        for name, value in (
            ("EMO_AGENT_MAX_UPLOAD_BYTES", self.max_upload_bytes),
            ("EMO_AGENT_REQUESTS_PER_MINUTE", self.requests_per_minute),
            ("EMO_AGENT_EXPENSIVE_REQUESTS_PER_MINUTE", self.expensive_requests_per_minute),
            ("EMO_AGENT_MAX_CONCURRENT_ANALYSIS", self.max_concurrent_analysis),
            ("EMO_AGENT_LLM_UPSTREAM_CALLS_PER_DAY", self.llm_upstream_calls_per_day),
            ("EMO_AGENT_LLM_UPSTREAM_CALLS_PER_USER_PER_DAY", self.llm_upstream_calls_per_user_per_day),
            ("EMO_AGENT_LLM_MAX_TOOL_ROUNDS", self.llm_max_tool_rounds),
            ("EMO_AGENT_LLM_WEB_SEARCH_COST", self.llm_web_search_cost),
        ):
            if value < 1:
                raise RuntimeError(f"{name} must be positive")


class PersistentDailyQuota:
    """A restart-safe, UTC daily counter for upstream paid-model calls."""

    def __init__(self, database_path: str) -> None:
        self.database_path = Path(database_path)
        self._lock = threading.Lock()

    def allow_many(self, limits: dict[str, int], cost: int = 1) -> bool:
        if cost < 1:
            raise ValueError("quota cost must be positive")
        day = datetime.now(timezone.utc).date().isoformat()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_llm_quota (
                    day TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    calls INTEGER NOT NULL,
                    PRIMARY KEY (day, scope)
                )
                """
            )
            connection.execute("BEGIN IMMEDIATE")
            for scope, limit in limits.items():
                row = connection.execute(
                    "SELECT calls FROM daily_llm_quota WHERE day = ? AND scope = ?", (day, scope)
                ).fetchone()
                if (row[0] if row else 0) + cost > limit:
                    connection.rollback()
                    return False
            for scope in limits:
                connection.execute(
                    """
                    INSERT INTO daily_llm_quota(day, scope, calls) VALUES (?, ?, ?)
                    ON CONFLICT(day, scope) DO UPDATE SET calls = calls + excluded.calls
                    """,
                    (day, scope, cost),
                )
            connection.commit()
        return True


def proxy_is_authorized(request: Request, settings: SecuritySettings) -> bool:
    if not settings.public_mode:
        return True
    supplied = request.headers.get("X-Emo-Proxy-Secret", "")
    return bool(settings.proxy_secret) and hmac.compare_digest(supplied, settings.proxy_secret)


def trusted_user_id(request: Request, settings: SecuritySettings, claimed: object = None) -> str:
    """Return a non-enumerable identity; never trust browser claims in public mode."""
    if settings.public_mode:
        raw = request.headers.get("X-Emo-User-Id", "").strip()
        if not raw:
            raise ValueError("trusted user identity is missing")
        # Hash again so an email address or other upstream identifier is never stored directly.
        digest = hmac.new(
            settings.proxy_secret.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return f"site-{digest[:32]}"
    clean = str(claimed or request.headers.get("X-User-Id") or "local-user").strip()[:128]
    return clean or "local-user"


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._events: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_seconds: float = 60.0) -> bool:
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                return False
            events.append(now)
            return True
