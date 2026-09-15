"""Regression tests for the public Sites-to-local-backend trust boundary."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.agent_service import DeepSeekConfig, EmotionCompanionAgent
from backend.fastapi_app import create_app
from backend.memory_service import MemoryService
from backend.security import SecuritySettings, trusted_user_id


SECRET = "s" * 64


def public_settings(**overrides: int) -> SecuritySettings:
    values = {
        "public_mode": True,
        "proxy_secret": SECRET,
        "max_upload_bytes": 32,
        "requests_per_minute": 10,
        "expensive_requests_per_minute": 2,
        "max_concurrent_analysis": 1,
    }
    values.update(overrides)
    return SecuritySettings(**values)


def make_client(settings: SecuritySettings | None = None) -> tuple[TestClient, MemoryService]:
    memory = MemoryService()
    app = create_app(
        analyzer=object(),
        agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key=None)),
        memory=memory,
        frontend_dist=None,
        security=settings or public_settings(),
    )
    return TestClient(app), memory


def headers(identity: str = "anonymous-browser-a") -> dict[str, str]:
    return {"X-Emo-Proxy-Secret": SECRET, "X-Emo-User-Id": identity}


def test_public_mode_requires_a_strong_proxy_secret() -> None:
    settings = SecuritySettings(public_mode=True, proxy_secret="short")
    try:
        settings.validate()
    except RuntimeError as exc:
        assert "at least 32" in str(exc)
    else:
        raise AssertionError("weak public proxy secret was accepted")


def test_public_api_rejects_missing_or_wrong_proxy_secret() -> None:
    client, _ = make_client()

    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/sessions", headers={"X-Emo-Proxy-Secret": "wrong"}).status_code == 401
    assert client.get("/api/health").status_code == 200


def test_public_api_rejects_proxy_request_without_trusted_identity() -> None:
    client, _ = make_client()

    response = client.get("/api/sessions", headers={"X-Emo-Proxy-Secret": SECRET})

    assert response.status_code == 401
    assert response.json()["error"] == "trusted_identity_missing"


def test_public_identity_ignores_spoofed_query_user_id() -> None:
    client, memory = make_client()
    response = client.get("/api/sessions?user_id=victim", headers=headers("browser-a"))
    assert response.status_code == 200
    assert response.json()["sessions"] == []

    # Creating data through the trusted identity must not expose it to another identity,
    # even if that visitor claims the same browser-side user_id.
    trusted_a = "site-" + __import__("hmac").new(
        SECRET.encode(), b"browser-a", __import__("hashlib").sha256
    ).hexdigest()[:32]
    memory.ensure_session(trusted_a, "session-a")
    own = client.get("/api/sessions?user_id=victim", headers=headers("browser-a"))
    other = client.get("/api/sessions?user_id=victim", headers=headers("browser-b"))

    assert [item["session_id"] for item in own.json()["sessions"]] == ["session-a"]
    assert other.json()["sessions"] == []


def test_public_upload_limit_is_enforced_before_inference() -> None:
    client, _ = make_client(public_settings(max_upload_bytes=4))

    response = client.post(
        "/api/analyze-audio",
        content=b"12345",
        headers={**headers(), "Content-Type": "audio/wav"},
    )

    assert response.status_code == 413
    assert response.json()["error"] == "upload_too_large"


def test_expensive_endpoint_rate_limit_returns_429() -> None:
    client, _ = make_client(public_settings(expensive_requests_per_minute=1))
    payload = {"message": "hello"}

    first = client.post("/api/agent/chat", headers=headers(), json=payload)
    second = client.post("/api/agent/chat", headers=headers(), json=payload)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "60"


def test_daily_llm_quota_blocks_before_a_second_paid_request(tmp_path) -> None:
    def fake_deepseek(*_args, **_kwargs):
        return {"choices": [{"message": {"content": "可以。"}}]}

    settings = public_settings(
        llm_upstream_calls_per_day=2,
        llm_upstream_calls_per_user_per_day=2,
        llm_max_tool_rounds=2,
        llm_budget_db=str(tmp_path / "quota.sqlite3"),
    )
    agent = EmotionCompanionAgent(
        config=DeepSeekConfig(api_key="test-key"), post_json=fake_deepseek
    )
    app = create_app(agent=agent, memory=MemoryService(), frontend_dist=None, security=settings)
    client = TestClient(app)

    assert client.post("/api/agent/chat", headers=headers(), json={"message": "你好"}).status_code == 200
    blocked = client.post("/api/agent/chat", headers=headers(), json={"message": "再说一句"})

    assert blocked.status_code == 429
    assert blocked.json()["error"] == "llm_daily_quota_exceeded"


def test_chat_message_length_is_limited_before_llm_budget() -> None:
    client, _ = make_client()

    response = client.post("/api/agent/chat", headers=headers(), json={"message": "x" * 2001})

    assert response.status_code == 422
    assert response.json()["error"] == "message_too_long"
