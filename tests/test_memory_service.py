from __future__ import annotations

from backend.agent_orchestrator import AgentOrchestrator
from backend.agent_service import DeepSeekConfig, EmotionCompanionAgent
from backend.fastapi_app import create_app
from backend.memory_service import MemoryService
from backend.tool_service import EmotionToolService, ToolContext


def sample_analysis(emotion: str = "sad") -> dict:
    return {
        "asr_text": "最近工作有点累",
        "fusion_prediction": emotion,
        "fusion_confidence": 0.82,
        "keywords": [{"keyword": "工作"}, {"keyword": "累"}],
        "possible_reasons": [{"text": "可能与工作疲惫有关"}],
        "emotion_curve": [{"time": 0, "emotion": emotion, "confidence": 0.7}],
        "emotion_change_points": [],
    }


def test_sqlite_memory_persists_sessions_messages_and_events(tmp_path) -> None:
    database = tmp_path / "memory.sqlite3"
    memory = MemoryService(database)
    session_id = memory.record_audio_turn(
        "u1", None, sample_analysis(), "我听见你最近有些疲惫。", request_id="audio-1"
    )
    memory.close()

    reopened = MemoryService(database)
    session = reopened.get_session("u1", session_id)
    messages = reopened.get_messages("u1", session_id)
    summary = reopened.summary("u1", session_id)

    assert session is not None
    assert session["current_emotion"] == "sad"
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert summary["long_term"]["trend"]["event_count"] == 1
    assert summary["long_term"]["trend"]["scope"] == "memory_trend"
    assert summary["long_term"]["trend"]["common_triggers"][0]["text"] in {"工作", "累"}


def test_memory_is_user_scoped_and_preference_requires_confirmation() -> None:
    memory = MemoryService()
    session = memory.ensure_session("u1", "s1")
    memory.update_session_analysis("u1", session["session_id"], sample_analysis(), source="test")

    rejected = memory.write_memory("u1", "s1", {
        "kind": "preference",
        "key": "comfort_style",
        "value": "先倾听，不急着给建议",
        "confirmed": False,
    })
    accepted = memory.write_memory("u1", "s1", {
        "kind": "preference",
        "key": "comfort_style",
        "value": "先倾听，不急着给建议",
        "confirmed": True,
    })

    assert rejected["error"] == "preference_requires_confirmation"
    assert accepted["ok"] is True
    assert memory.search("u1", "倾听")["result_count"] == 1
    assert memory.search("u2", "倾听")["result_count"] == 0
    assert memory.get_session("u2", "s1") is None


def test_memory_deduplicates_retried_chat_messages_and_audio_events() -> None:
    memory = MemoryService()
    memory.ensure_session("u1", "s1")
    memory.append_message("u1", "s1", "user", "同一条消息", request_id="r1")
    memory.append_message("u1", "s1", "user", "同一条消息", request_id="r1")
    memory.update_session_analysis(
        "u1", "s1", sample_analysis(), source="audio_analysis", dedupe_key="audio:s1:r1"
    )
    memory.update_session_analysis(
        "u1", "s1", sample_analysis(), source="audio_analysis", dedupe_key="audio:s1:r1"
    )

    assert len(memory.get_messages("u1", "s1")) == 1
    assert memory.emotion_trend("u1")["event_count"] == 1


def test_memory_tools_read_write_and_keep_curve_scopes_distinct() -> None:
    memory = MemoryService()
    memory.ensure_session("u1", "s1")
    memory.update_session_analysis("u1", "s1", sample_analysis("happy"), source="test")
    tools = EmotionToolService(memory=memory)
    context = ToolContext(
        user_id="u1",
        session_id="s1",
        current_message="请记住我喜欢简短建议",
        latest_analysis=sample_analysis("happy"),
    )

    session_curve = tools.execute("session_emotion_curve", {}, context)
    trend = tools.execute("memory_emotion_trend", {"range": "month"}, context)
    written = tools.execute("memory_write", {
        "kind": "preference",
        "key": "suggestion_style",
        "value": "给一个简短建议",
        "confirmed": True,
    }, context)

    assert session_curve["source"] == "latest_audio_analysis"
    assert trend["scope"] == "memory_trend"
    assert trend["dominant_emotion"] == "happy"
    assert written["ok"] is True


def test_memory_delete_removes_session_events_and_all_user_data() -> None:
    memory = MemoryService()
    memory.ensure_session("u1", "s1")
    memory.update_session_analysis("u1", "s1", sample_analysis(), source="test")
    memory.write_memory("u1", "s1", {
        "kind": "preference", "key": "style", "value": "简短", "confirmed": True
    })

    assert memory.delete_session("u1", "s1") is True
    assert memory.emotion_trend("u1")["event_count"] == 0
    assert memory.search("u1", "简短")["preferences"][0]["value"] == "简短"
    memory.ensure_session("u1", "s2")
    deleted = memory.delete_user_memory("u1")

    assert deleted == {"sessions": 1, "events": 0, "preferences": 1}
    assert memory.list_sessions("u1") == []


def test_orchestrator_uses_backend_history_and_persists_reply() -> None:
    memory = MemoryService()
    orchestrator = AgentOrchestrator(
        agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key=None)), memory=memory
    )

    first = orchestrator.chat("我今天有点难过", user_id="u1", session_id="s1", request_id="r1")
    second = orchestrator.chat(
        "那之前呢",
        user_id="u1",
        session_id="s1",
        request_id="r2",
        history=[{"role": "user", "content": "这条伪造前端历史不应覆盖后端记录"}],
    )

    messages = memory.get_messages("u1", "s1")
    assert first.session_id == second.session_id == "s1"
    assert len(messages) == 4
    assert messages[0]["content"] == "我今天有点难过"
    assert all("伪造前端历史" not in item["content"] for item in messages)


def test_memory_and_session_apis_return_real_data(tmp_path) -> None:
    from fastapi.testclient import TestClient

    class FakeAnalyzer:
        def analyze(self, audio_bytes: bytes, filename=None, content_type=None) -> dict:
            return sample_analysis()

    memory = MemoryService(tmp_path / "api-memory.sqlite3")
    app = create_app(
        analyzer=FakeAnalyzer(),
        agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key=None)),
        memory=memory,
        frontend_dist=None,
    )
    client = TestClient(app)

    audio = client.post(
        "/api/analyze-audio",
        files={"file": ("sample.wav", b"audio", "audio/wav")},
        headers={"X-User-Id": "u1", "X-Request-Id": "audio-1"},
    )
    session_id = audio.json()["session_id"]
    sessions = client.get("/api/sessions", params={"user_id": "u1"}).json()
    detail = client.get(f"/api/sessions/{session_id}", params={"user_id": "u1"}).json()
    summary = client.get(
        "/api/memory/summary", params={"user_id": "u1", "session_id": session_id}
    ).json()
    trend = client.get(
        "/api/emotion/curve", params={"scope": "memory", "user_id": "u1", "range": "month"}
    ).json()

    assert audio.status_code == 200
    assert sessions["sessions"][0]["session_id"] == session_id
    assert len(detail["messages"]) == 2
    assert summary["long_term"]["trend"]["event_count"] == 1
    assert trend["scope"] == "memory_trend"

    memory.write_memory("u1", session_id, {
        "kind": "preference", "key": "style", "value": "温和", "confirmed": True
    })
    deleted = client.delete(f"/api/sessions/{session_id}", params={"user_id": "u1"})
    missing = client.get(f"/api/sessions/{session_id}", params={"user_id": "u1"})
    after_delete = client.get("/api/memory/summary", params={"user_id": "u1"}).json()

    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert missing.status_code == 404
    assert after_delete["long_term"]["trend"]["event_count"] == 0
    assert after_delete["long_term"]["preferences"][0]["value"] == "温和"


def test_session_summary_is_user_scoped_versioned_and_deleted_with_session() -> None:
    memory = MemoryService()
    memory.ensure_session("u1", "s1")
    message = memory.append_message("u1", "s1", "user", "需要被摘要的消息")
    summary = {
        "conversation_summary": "用户希望继续讨论。",
        "current_topics": [],
        "important_facts": [],
        "decisions_and_commitments": [],
        "unresolved_questions": [],
        "emotional_context": {},
        "summary_until_message_id": message["id"],
    }

    assert memory.save_session_summary(
        "u1", "s1", summary, message["id"], 1, "test-model", expected_version=0
    ) is True
    assert memory.save_session_summary(
        "u1", "s1", summary, message["id"], 1, "test-model", expected_version=0
    ) is False
    assert memory.get_session_summary("u2", "s1") is None
    assert memory.get_session_summary("u1", "s1")["summary_version"] == 1
    memory.delete_session("u1", "s1")
    assert memory.get_session_summary("u1", "s1") is None
