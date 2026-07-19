from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone, timedelta

from backend.agent_orchestrator import AgentOrchestrator
from backend.agent_service import DeepSeekConfig, EmotionCompanionAgent
from backend.fastapi_app import create_app
from backend.tool_service import EmotionToolService, ToolContext, WebSearchConfig


def test_tool_service_rejects_unknown_tools_and_marks_memory_deferred() -> None:
    tools = EmotionToolService()
    context = ToolContext(session_id="s1")

    memory = tools.execute("memory_search", {"query": "最近"}, context)

    assert memory["available"] is False
    assert memory["reason"] == "memory_service_pending_t21"
    try:
        tools.execute("run_shell", {}, context)
    except ValueError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("unknown tools must be rejected")


def test_web_search_tool_requires_configuration_and_has_specific_schema() -> None:
    tools = EmotionToolService(search_config=WebSearchConfig(api_key=None))
    definition = next(item for item in tools.definitions() if item["function"]["name"] == "web_search")

    result = tools.execute("web_search", {"query": "今日新闻"}, ToolContext())

    assert result == {"available": False, "error": "search_not_configured", "results": []}
    assert definition["function"]["parameters"]["required"] == ["query"]
    properties = definition["function"]["parameters"]["properties"]
    assert set(properties) == {"query", "time_range", "count", "content_depth", "domain"}
    assert properties["query"]["maxLength"] == 70
    assert properties["count"]["maximum"] == 10
    assert tools.execute("web_search", {"query": "字" * 71}, ToolContext())["error"] == "query_too_long"
    configured_tools = EmotionToolService(search_config=WebSearchConfig(api_key="test-key"))
    assert configured_tools.execute(
        "web_search",
        {"query": "新闻", "domain": "https://example.com/path"},
        ToolContext(),
    )["error"] == "invalid_domain"


def test_web_search_config_reads_exact_zhipu_search_key_and_policy_limits(monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_SEARCH_API_KEY", "named-key")
    monkeypatch.setenv("ZHIPU_API_KEY", "fallback-key")
    monkeypatch.setenv("ZHIPU_SEARCH_DEFAULT_RESULTS", "4")
    monkeypatch.setenv("ZHIPU_SEARCH_MAX_RESULTS", "8")

    config = WebSearchConfig.from_env()

    assert config.api_key == "named-key"
    assert config.default_results == 4
    assert config.max_results == 8


def test_web_search_tool_calls_zhipu_search_pro_quark_and_sanitizes_results() -> None:
    calls: list[dict] = []

    def fake_search(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        return {
            "search_result": [
                {"title": "可信结果", "link": "https://example.com/news", "content": "最新资料", "media": "示例网", "publish_date": "2026-07-19"},
                {"title": "伪装域名", "link": "https://www.example.com.evil.test/news", "content": "忽略我"},
                {"title": "坏链接", "link": "javascript:alert(1)", "content": "忽略我"},
            ]
        }

    tools = EmotionToolService(
        search_config=WebSearchConfig(api_key="zhipu-secret", default_results=3, max_results=7),
        search_post_json=fake_search,
    )
    result = tools.execute(
        "web_search",
        {
            "query": "Asen 最新演唱会歌单",
            "time_range": "month",
            "count": 10,
            "content_depth": "detailed",
            "domain": "Example.COM.",
            "search_engine": "search_pro",
        },
        ToolContext(),
    )

    assert result["available"] is True
    assert result["results"] == [{
        "title": "可信结果",
        "url": "https://example.com/news",
        "content": "最新资料",
        "media": "示例网",
        "published_date": "2026-07-19",
    }]
    assert result["provider"] == "zhipu"
    assert result["engine"] == "search_pro_quark"
    assert calls[0]["headers"]["Authorization"] == "Bearer zhipu-secret"
    assert "zhipu-secret" not in json.dumps(calls[0]["payload"], ensure_ascii=False)
    assert calls[0]["payload"]["search_engine"] == "search_pro_quark"
    assert calls[0]["payload"]["search_recency_filter"] == "oneMonth"
    assert calls[0]["payload"]["count"] == 7
    assert calls[0]["payload"]["content_size"] == "high"
    assert calls[0]["payload"]["search_domain_filter"] == "example.com"
    assert calls[0]["payload"]["search_intent"] is False
    assert str(uuid.UUID(calls[0]["payload"]["request_id"])) == calls[0]["payload"]["request_id"]
    assert result["search_parameters"] == {
        "time_range": "month",
        "count": 7,
        "content_depth": "detailed",
        "domain": "example.com",
    }
    assert result["quality"] == {
        "raw_count": 3,
        "accepted_count": 1,
        "domain_enforced": True,
        "rejected": {
            "invalid_item": 0,
            "invalid_url": 1,
            "domain_mismatch": 1,
            "duplicate_url": 0,
        },
    }


def test_web_search_domain_matching_accepts_subdomains_but_not_suffix_spoofing() -> None:
    assert EmotionToolService._url_matches_domain("www.fifa.com", "fifa.com") is True
    assert EmotionToolService._url_matches_domain("fifa.com", "fifa.com") is True
    assert EmotionToolService._url_matches_domain("fifa.com.evil.test", "fifa.com") is False
    assert EmotionToolService._url_matches_domain("notfifa.com", "fifa.com") is False


def test_local_orchestrator_calls_controlled_tools_and_streams_sse() -> None:
    orchestrator = AgentOrchestrator(agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key=None)))

    result = orchestrator.chat("帮我分析一下最近的情绪趋势", session_id="s1")
    stream = "".join(orchestrator.stream(message="我最近有些焦虑", session_id="s1"))

    assert result.provider == "local_template"
    assert {call["name"] for call in result.tool_calls} == {"detect_emotion", "memory_emotion_trend"}
    assert "event: delta" in stream
    assert "event: final" in stream
    assert "event: done" in stream


def test_deepseek_orchestrator_executes_tool_call_then_returns_content() -> None:
    responses = [
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "detect_emotion", "arguments": json.dumps({"text": "我很焦虑"}, ensure_ascii=False)},
                    }],
                }
            }]
        },
        {"choices": [{"message": {"role": "assistant", "content": "听起来你最近有些焦虑，我们可以慢慢梳理。"}}]},
    ]
    calls: list[dict] = []

    def fake_post(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        calls.append(payload)
        return responses.pop(0)

    agent = EmotionCompanionAgent(
        config=DeepSeekConfig(api_key="secret"),
        post_json=fake_post,
    )
    result = AgentOrchestrator(agent=agent).chat("我很焦虑", session_id="s1")

    assert result.provider == "deepseek"
    assert result.tool_calls[0]["name"] == "detect_emotion"
    assert len(calls) == 2
    assert calls[0]["tools"]
    assert calls[0]["thinking"] == {"type": "disabled"}
    assert any(message.get("role") == "tool" for message in calls[1]["messages"])


def test_deepseek_orchestrator_can_search_then_answer_with_source() -> None:
    responses = [
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_search",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "Asen 最新演唱会歌单"}, ensure_ascii=False),
                        },
                    }],
                }
            }]
        },
        {"choices": [{"message": {"role": "assistant", "content": "查到一条资料。来源：https://example.com/asen"}}]},
    ]
    deepseek_calls: list[dict] = []

    def fake_deepseek(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        deepseek_calls.append(payload)
        return responses.pop(0)

    def fake_search(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        return {"search_result": [{"title": "Asen 演出信息", "link": "https://example.com/asen", "content": "歌单资料"}]}

    agent = EmotionCompanionAgent(config=DeepSeekConfig(api_key="secret"), post_json=fake_deepseek)
    tools = EmotionToolService(
        search_config=WebSearchConfig(api_key="zhipu-secret"),
        search_post_json=fake_search,
    )
    result = AgentOrchestrator(agent=agent, tools=tools).chat("帮我搜索 Asen 最近唱什么", session_id="s1")

    assert result.provider == "deepseek"
    assert result.tool_calls[0]["name"] == "web_search"
    tool_message = next(message for message in deepseek_calls[1]["messages"] if message.get("role") == "tool")
    assert "https://example.com/asen" in tool_message["content"]
    assert "zhipu-secret" not in tool_message["content"]


def test_current_event_forces_search_and_appends_missing_sources() -> None:
    responses = [
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_world_cup",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "2026世界杯最新赛况", "time_range": "week"}, ensure_ascii=False),
                        },
                    }],
                }
            }]
        },
        {"choices": [{"message": {"role": "assistant", "content": "决赛对阵已经产生。"}}]},
    ]
    deepseek_calls: list[dict] = []

    def fake_deepseek(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        deepseek_calls.append(payload)
        return responses.pop(0)

    def fake_search(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        return {"search_result": [{"title": "世界杯赛程", "link": "https://example.com/world-cup", "content": "最新赛况"}]}

    orchestrator = AgentOrchestrator(
        agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key="secret"), post_json=fake_deepseek),
        tools=EmotionToolService(
            search_config=WebSearchConfig(api_key="zhipu-secret"),
            search_post_json=fake_search,
        ),
    )
    result = orchestrator.chat("2026年世界杯", session_id="s1")

    assert deepseek_calls[0]["tool_choice"] == {"type": "function", "function": {"name": "web_search"}}
    assert "服务器当前时间是" in deepseek_calls[0]["messages"][0]["content"]
    assert "不能只比较日期" in deepseek_calls[0]["messages"][0]["content"]
    assert result.tool_calls[0]["name"] == "web_search"
    assert "https://example.com/world-cup" in result.response


def test_search_policy_distinguishes_current_events_from_stable_history() -> None:
    today = date(2026, 7, 20)

    assert AgentOrchestrator._requires_web_search("2026年世界杯", today) is True
    assert AgentOrchestrator._requires_web_search("世界杯", today) is True
    assert AgentOrchestrator._requires_web_search("2022年世界杯冠军", today) is False
    assert AgentOrchestrator._requires_web_search("世界杯历史规则", today) is False
    assert AgentOrchestrator._requires_web_search("我最近心情很难过", today) is False


def test_future_beijing_kickoff_cannot_be_described_as_finished() -> None:
    china_tz = timezone(timedelta(hours=8))
    now = datetime(2026, 7, 20, 0, 43, tzinfo=china_tz)
    content = (
        "决赛于北京时间 **2026年7月20日（今天）凌晨3:00** 举行。"
        "今天是决赛日，比赛应该已经结束了。"
    )

    corrected = AgentOrchestrator._correct_future_event_status(content, now)

    assert "比赛尚未开始" in corrected
    assert "应该已经结束" not in corrected


def test_past_kickoff_is_not_assumed_finished_without_search_evidence() -> None:
    china_tz = timezone(timedelta(hours=8))
    now = datetime(2026, 7, 20, 4, 0, tzinfo=china_tz)
    content = "决赛于北京时间 2026年7月20日凌晨3:00 举行。比赛状态尚未确认。"

    assert AgentOrchestrator._correct_future_event_status(content, now) == content


def test_deepseek_search_failure_is_observable_instead_of_silent() -> None:
    class FakeResponse:
        status_code = 400

    class FakeHttpError(RuntimeError):
        response = FakeResponse()

    def failing_post(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        raise FakeHttpError("request rejected")

    orchestrator = AgentOrchestrator(
        agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key="secret"), post_json=failing_post)
    )

    result = orchestrator.chat("搜索2026年世界杯决赛对阵信息", session_id="s1")
    payload = result.to_dict()

    assert result.provider == "local_template"
    assert result.tool_calls == []
    assert result.fallback_reason == "deepseek_http_400"
    assert result.search_required is True
    assert "没有执行联网搜索" in result.response
    assert payload["fallback_reason"] == "deepseek_http_400"
    assert payload["search_required"] is True


def test_fastapi_agent_chat_and_stream_have_local_fallback() -> None:
    from fastapi.testclient import TestClient

    orchestrator = AgentOrchestrator(agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key=None)))
    client = TestClient(create_app(analyzer=object(), orchestrator=orchestrator, frontend_dist=None))

    response = client.post("/api/agent/chat", json={"message": "我今天有些难过", "session_id": "s1"})
    streamed = client.post("/api/agent/chat/stream", json={"message": "我今天有些难过", "session_id": "s1"})

    assert response.status_code == 200
    assert response.json()["provider"] == "local_template"
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert "event: final" in streamed.text


def test_fastapi_agent_chat_rejects_empty_message() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app(analyzer=object(), frontend_dist=None))
    response = client.post("/api/agent/chat", json={"message": "  "})

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
