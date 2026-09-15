from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

from backend.agent_orchestrator import AgentOrchestrator, ContextCompactionUnavailable
from backend.agent_service import DeepSeekConfig, EmotionCompanionAgent
from backend.fastapi_app import create_app
from backend.tool_service import EmotionToolService, ToolContext


def test_tool_service_only_exposes_local_controlled_tools() -> None:
    tools = EmotionToolService()
    assert "web_search" not in tools.names
    assert tools.execute("memory_search", {"query": "最近"}, ToolContext())["available"] is True
    try:
        tools.execute("web_search", {}, ToolContext())
    except ValueError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("native web search must not become a browser-callable local tool")


def test_local_orchestrator_calls_controlled_tools_and_streams_sse() -> None:
    orchestrator = AgentOrchestrator(agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key=None)))
    result = orchestrator.chat("帮我分析一下最近的情绪趋势", session_id="s1")
    stream = "".join(orchestrator.stream(message="我最近有些焦虑", session_id="s1"))
    assert result.provider == "local_template"
    assert {call["name"] for call in result.tool_calls} == {"detect_emotion", "memory_emotion_trend"}
    assert "event: delta" in stream
    assert "event: final" in stream


def test_deepseek_orchestrator_executes_local_tool_then_returns_content() -> None:
    responses = [
        {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "detect_emotion", "arguments": json.dumps({"text": "我很焦虑"}, ensure_ascii=False)},
        }]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "听起来你最近有些焦虑，我们可以慢慢梳理。"}}]},
    ]
    calls: list[dict] = []

    def fake_post(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        return responses.pop(0)

    agent = EmotionCompanionAgent(config=DeepSeekConfig(api_key="test-key"), post_json=fake_post)
    result = AgentOrchestrator(agent=agent).chat("我很焦虑", session_id="s1")
    assert result.provider == "deepseek"
    assert result.tool_calls[0]["name"] == "detect_emotion"
    assert len(calls) == 2
    assert calls[0]["url"].endswith("/chat/completions")
    assert calls[0]["payload"]["thinking"] == {"type": "disabled"}
    assert all(item["function"]["name"] != "web_search" for item in calls[0]["payload"]["tools"])


def test_current_event_uses_deepseek_native_responses_web_search() -> None:
    calls: list[dict] = []

    def fake_post(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        return {"output_text": "我已核对公开资料：比赛状态仍需以官方最终公告为准。"}

    agent = EmotionCompanionAgent(config=DeepSeekConfig(api_key="test-key"), post_json=fake_post)
    result = AgentOrchestrator(agent=agent).chat("搜索2026年世界杯最新赛况", session_id="s1", user_id="site-opaque-user")
    assert result.provider == "deepseek"
    assert result.search_required is True
    assert result.tool_calls[0]["name"] == "web_search"
    assert result.tool_calls[0]["result"] == {
        "available": True, "provider": "deepseek_native", "native": True, "results": []
    }
    assert calls[0]["url"].endswith("/responses")
    assert calls[0]["payload"]["tools"] == [{"type": "web_search"}]
    assert calls[0]["payload"]["tool_choice"] == {"type": "web_search"}
    assert calls[0]["payload"]["user"] == "site-opaque-user"
    assert calls[0]["payload"]["max_output_tokens"] == 1000
    assert "服务器当前时间是" in calls[0]["payload"]["instructions"]
    assert calls[0]["payload"]["input"] == [
        {"role": "user", "content": "搜索2026年世界杯最新赛况"}
    ]


def test_ordinary_factual_question_offers_native_search_without_explicit_command() -> None:
    calls: list[dict] = []

    def fake_post(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        calls.append({"url": url, "payload": payload})
        return {
            "output": [
                {"type": "web_search_call"},
                {"type": "message", "content": [{"type": "output_text", "text": "这是公开资料的整理。"}]},
            ]
        }

    agent = EmotionCompanionAgent(config=DeepSeekConfig(api_key="test-key"), post_json=fake_post)
    result = AgentOrchestrator(agent=agent).chat("Asen 最近的演出安排是什么？", session_id="s1")

    assert calls[0]["url"].endswith("/responses")
    assert calls[0]["payload"]["tool_choice"] == "auto"
    assert result.tool_calls[0]["name"] == "web_search"
    assert result.search_required is False


def test_personal_emotion_question_does_not_offer_web_search() -> None:
    calls: list[dict] = []

    def fake_post(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        calls.append({"url": url, "payload": payload})
        return {"choices": [{"message": {"role": "assistant", "content": "我在听。"}}]}

    agent = EmotionCompanionAgent(config=DeepSeekConfig(api_key="test-key"), post_json=fake_post)
    AgentOrchestrator(agent=agent).chat("我最近为什么总是很焦虑？", session_id="s1")

    assert calls[0]["url"].endswith("/chat/completions")


def test_responses_output_parser_accepts_message_content_shape() -> None:
    raw = {"output": [{"type": "web_search_call"}, {"type": "message", "content": [
        {"type": "output_text", "text": "第一段"}, {"type": "output_text", "text": "第二段"}
    ]}]}
    assert AgentOrchestrator._responses_output_text(raw) == "第一段\n第二段"


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
    content = "决赛于北京时间 **2026年7月20日（今天）凌晨3:00** 举行。比赛应该已经结束了。"
    assert "比赛尚未开始" in AgentOrchestrator._correct_future_event_status(content, now)


def test_deepseek_native_search_failure_is_observable() -> None:
    class FakeResponse:
        status_code = 400

    class FakeHttpError(RuntimeError):
        response = FakeResponse()

    def failing_post(*_args, **_kwargs):
        raise FakeHttpError("request rejected")

    orchestrator = AgentOrchestrator(
        agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key="test-key"), post_json=failing_post)
    )
    result = orchestrator.chat("搜索2026年世界杯决赛对阵信息", session_id="s1")
    assert result.provider == "local_template"
    assert result.fallback_reason == "deepseek_http_400"
    assert result.search_required is True
    assert "没有执行联网搜索" in result.response


def test_fastapi_agent_chat_and_stream_have_local_fallback() -> None:
    from fastapi.testclient import TestClient

    orchestrator = AgentOrchestrator(agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key=None)))
    client = TestClient(
        create_app(
            analyzer=object(),
            agent=orchestrator.agent,
            orchestrator=orchestrator,
            frontend_dist=None,
        )
    )
    response = client.post("/api/agent/chat", json={"message": "我今天有些难过", "session_id": "s1"})
    streamed = client.post("/api/agent/chat/stream", json={"message": "我今天有些难过", "session_id": "s1"})
    assert response.status_code == 200
    assert response.json()["provider"] == "local_template"
    assert streamed.status_code == 200
    assert "event: final" in streamed.text


def test_long_chat_is_compacted_and_summary_is_injected() -> None:
    from backend.memory_service import MemoryService

    calls: list[dict] = []

    def fake_post(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        calls.append(payload)
        if payload.get("response_format") == {"type": "json_object"}:
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "conversation_summary": "用户一直在讨论工作压力。",
                "current_topics": ["工作"],
                "important_facts": [{"content": "近期工作较忙", "source_message_ids": [1]}],
                "decisions_and_commitments": [],
                "unresolved_questions": ["如何休息"],
                "emotional_context": {"recent_emotions": ["疲惫"]},
                "summary_until_message_id": 0,
            }, ensure_ascii=False)}}]}
        return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "我记得，我们继续聊。"}}]}

    memory = MemoryService()
    memory.ensure_session("u1", "s1")
    for index in range(22):
        memory.append_message("u1", "s1", "user" if index % 2 == 0 else "assistant", f"历史消息{index}")
    memory.write_memory("u1", "s1", {
        "kind": "preference", "key": "reply_style", "value": "简短", "confirmed": True
    })
    config = DeepSeekConfig(api_key="test-key", summary_retry_delay_seconds=0)
    result = AgentOrchestrator(
        agent=EmotionCompanionAgent(config=config, post_json=fake_post), memory=memory, sleep=lambda _: None
    ).chat("继续陪我聊聊", user_id="u1", session_id="s1")

    state = memory.get_session_summary("u1", "s1")
    assert result.response == "我记得，我们继续聊。"
    assert state is not None
    assert state["summarized_through_message_id"] == 14
    main_payload = calls[-1]
    assert "rolling_session_summary" in main_payload["messages"][0]["content"]
    assert "confirmed_long_term_preferences" in main_payload["messages"][0]["content"]
    assert len([item for item in main_payload["messages"] if item["role"] in {"user", "assistant"}]) == 9


def test_compaction_retries_three_times_then_enters_cooldown() -> None:
    from backend.memory_service import MemoryService

    attempts = 0

    def failing_post(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("temporary upstream failure")

    memory = MemoryService()
    memory.ensure_session("u1", "s1")
    for index in range(21):
        memory.append_message("u1", "s1", "user" if index % 2 == 0 else "assistant", str(index))
    config = DeepSeekConfig(
        api_key="test-key",
        summary_retry_attempts=3,
        summary_retry_delay_seconds=0,
        summary_cooldown_seconds=600,
    )
    orchestrator = AgentOrchestrator(
        agent=EmotionCompanionAgent(config=config, post_json=failing_post), memory=memory, sleep=lambda _: None
    )

    try:
        orchestrator.chat("继续", user_id="u1", session_id="s1")
    except ContextCompactionUnavailable as exc:
        assert exc.retry_after_seconds == 600
    else:
        raise AssertionError("failed compaction must block the reply")
    assert attempts == 3

    try:
        orchestrator.chat("再次发送", user_id="u1", session_id="s1")
    except ContextCompactionUnavailable as exc:
        assert 1 <= exc.retry_after_seconds <= 600
    else:
        raise AssertionError("cooldown must reject immediately")
    assert attempts == 3

    memory.set_summary_cooldown(
        "u1", "s1", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    try:
        orchestrator.chat("冷却后重新发送", user_id="u1", session_id="s1")
    except ContextCompactionUnavailable:
        pass
    else:
        raise AssertionError("a new failed cycle must still block the reply")
    assert attempts == 6


def test_truncated_chat_response_is_continued_once() -> None:
    responses = [
        {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "先慢慢"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "呼吸一下。"}}]},
    ]
    calls: list[dict] = []

    def fake_post(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        calls.append(payload)
        return responses.pop(0)

    agent = EmotionCompanionAgent(config=DeepSeekConfig(api_key="test-key"), post_json=fake_post)
    result = AgentOrchestrator(agent=agent).chat("我有些紧张", session_id="s1")

    assert result.response == "先慢慢呼吸一下。"
    assert len(calls) == 2
    assert calls[1]["messages"][-1]["content"].startswith("请从刚才截断的位置继续")
    assert "tools" not in calls[1]


def test_truncated_tool_call_is_regenerated_once() -> None:
    truncated_call = {
        "id": "bad", "type": "function",
        "function": {"name": "detect_emotion", "arguments": "{\"text\":"},
    }
    complete_call = {
        "id": "good", "type": "function",
        "function": {"name": "detect_emotion", "arguments": json.dumps({"text": "焦虑"}, ensure_ascii=False)},
    }
    responses = [
        {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "tool_calls": [truncated_call]}}]},
        {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [complete_call]}}]},
        {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "我在听。"}}]},
    ]
    calls: list[dict] = []

    def fake_post(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        calls.append(payload)
        return responses.pop(0)

    result = AgentOrchestrator(agent=EmotionCompanionAgent(
        config=DeepSeekConfig(api_key="test-key"), post_json=fake_post
    )).chat("我有些焦虑", session_id="s1")

    assert result.response == "我在听。"
    assert len(calls) == 3
    assert "更精简" in calls[1]["messages"][-1]["content"]
    assert result.tool_calls[0]["name"] == "detect_emotion"


def test_truncated_summary_json_is_regenerated_once() -> None:
    complete = {
        "conversation_summary": "简短摘要",
        "current_topics": [],
        "important_facts": [],
        "decisions_and_commitments": [],
        "unresolved_questions": [],
        "emotional_context": {},
        "summary_until_message_id": 0,
    }
    responses = [
        {"choices": [{"finish_reason": "length", "message": {"content": "{\"conversation_summary\":"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(complete, ensure_ascii=False)}}]},
    ]
    calls: list[dict] = []

    def fake_post(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        calls.append(payload.copy())
        return responses.pop(0)

    orchestrator = AgentOrchestrator(agent=EmotionCompanionAgent(
        config=DeepSeekConfig(api_key="test-key"), post_json=fake_post
    ))
    result = orchestrator._request_summary({}, [{"id": 7, "role": "user", "content": "很长的旧消息"}])

    assert result["summary_until_message_id"] == 7
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 1500
    assert "更短但完整" in calls[1]["messages"][-1]["content"]


def test_incomplete_native_search_response_is_regenerated_once() -> None:
    responses = [
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output_text": "未完成",
        },
        {"status": "completed", "output_text": "这是完整且精简的搜索回答。"},
    ]
    calls: list[dict] = []

    def fake_post(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        calls.append(payload)
        return responses.pop(0)

    result = AgentOrchestrator(agent=EmotionCompanionAgent(
        config=DeepSeekConfig(api_key="test-key"), post_json=fake_post
    )).chat("搜索今天的天气", session_id="s1")

    assert result.response == "这是完整且精简的搜索回答。"
    assert len(calls) == 2
    assert "更精简" in calls[1]["instructions"]


def test_stream_endpoint_returns_503_when_compaction_is_unavailable() -> None:
    from fastapi.testclient import TestClient
    from backend.memory_service import MemoryService

    memory = MemoryService()
    memory.ensure_session("local-user", "s1")
    for index in range(21):
        memory.append_message("local-user", "s1", "user" if index % 2 == 0 else "assistant", str(index))
    config = DeepSeekConfig(api_key=None, summary_retry_delay_seconds=0)
    orchestrator = AgentOrchestrator(
        agent=EmotionCompanionAgent(config=config), memory=memory, sleep=lambda _: None
    )
    client = TestClient(create_app(
        analyzer=object(), agent=orchestrator.agent, orchestrator=orchestrator, memory=memory, frontend_dist=None
    ))

    response = client.post("/api/agent/chat/stream", json={"message": "继续", "session_id": "s1"})

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "600"
    assert response.json()["error"] == "memory_compaction_unavailable"
