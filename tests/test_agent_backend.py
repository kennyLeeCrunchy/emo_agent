"""Tests for Module 7 Agent and Module 8 local backend service."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.agent_service import (
    AgentInput,
    DeepSeekConfig,
    EmotionCompanionAgent,
    local_template_response,
)
from backend.backend_service import INDEX_HTML, _parse_multipart_audio, analyze_audio_bytes, build_health_status
from backend.runtime_pipeline import (
    AudioBytesDecoder,
    DEFAULT_AUDIO_PIPELINE_PATH,
    DEFAULT_FUSION_PIPELINE_PATH,
    DEFAULT_LABEL_MAP_PATH,
    DEFAULT_ROBERTA_MODEL,
    DEFAULT_TEXT_PIPELINE_PATH,
    EmotionBackendRuntime,
    PROJECT_ROOT,
    PredictionResult,
    _remove_temp_file,
    predict_with_pipeline,
)
from backend.fastapi_app import create_app
from backend.memory_service import MemoryService
from backend.security import SecuritySettings


def test_runtime_default_paths_are_resolved_from_repository_root() -> None:
    expected_root = Path(__file__).resolve().parents[1]

    assert PROJECT_ROOT == expected_root
    assert DEFAULT_LABEL_MAP_PATH == expected_root / "modules/module1_preprocess/outputs/label_map.json"
    assert DEFAULT_ROBERTA_MODEL == expected_root / "models/pretrained/manual/chinese-roberta-wwm-ext"
    assert DEFAULT_AUDIO_PIPELINE_PATH.is_relative_to(expected_root)
    assert DEFAULT_TEXT_PIPELINE_PATH.is_relative_to(expected_root)
    assert DEFAULT_FUSION_PIPELINE_PATH.is_relative_to(expected_root)
    assert DEFAULT_LABEL_MAP_PATH.is_file()


def sample_agent_input() -> AgentInput:
    return AgentInput(
        asr_text="今天真的很累，感觉没人理解我。",
        audio_prediction="sad",
        text_prediction="sad",
        fusion_prediction="sad",
        audio_confidence=0.74,
        text_confidence=0.71,
        fusion_confidence=0.86,
        emotion_curve=[{"time": 0.0, "sad": 0.3}, {"time": 2.0, "sad": 0.62}],
        keywords=[{"keyword": "很累", "category": "fatigue"}],
        emotion_change_points=[{"time": 2.0, "emotion": "sad", "delta": 0.32}],
        possible_reasons=[{"type": "fusion_conditioned_reason", "text": "可能与疲惫感有关。"}],
    )


def test_deepseek_config_reads_system_env_without_leaking_key(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-key")

    config = DeepSeekConfig.from_env()

    assert config.api_key == "secret-key"
    assert config.base_url == "https://api.deepseek.com"
    assert config.chat_url == "https://api.deepseek.com/chat/completions"
    assert "secret-key" not in repr(config)


def test_local_template_response_returns_safe_five_part_output() -> None:
    response = local_template_response(sample_agent_input(), reason="missing_api_key")

    assert set(response) >= {
        "main_emotion_summary",
        "curve_interpretation",
        "possible_reasons",
        "companion_response",
        "gentle_suggestion",
        "safety_note",
        "uncertainty_notes",
        "provider",
    }
    assert response["provider"] == "local_template"
    assert "sad" in response["main_emotion_summary"]
    assert "医疗诊断" in response["safety_note"]
    forbidden = "你一定"
    assert forbidden not in json.dumps(response, ensure_ascii=False)


def test_backend_health_reports_deepseek_configuration(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    health = build_health_status()

    assert health["status"] == "ok"
    assert health["deepseek"]["configured"] is False
    assert health["deepseek"]["base_url"] == "https://api.deepseek.com"
    assert health["web_search"]["configured"] is False
    assert health["web_search"]["provider"] == "deepseek_native"
    assert health["web_search"]["api"] == "responses"


def test_multipart_audio_parser_does_not_depend_on_removed_cgi_module() -> None:
    boundary = "emo-agent-boundary"
    raw = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="sample.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8") + b"fake-audio-bytes" + f"\r\n--{boundary}--\r\n".encode("utf-8")

    audio_bytes, filename, content_type = _parse_multipart_audio(
        raw,
        f"multipart/form-data; boundary={boundary}",
    )

    assert audio_bytes == b"fake-audio-bytes"
    assert filename == "sample.wav"
    assert content_type == "audio/wav"


def test_analyze_audio_bytes_combines_backend_analysis_with_agent_response(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    class FakeAnalyzer:
        def analyze(self, audio_bytes: bytes, filename: str | None = None, content_type: str | None = None) -> dict:
            assert audio_bytes == b"fake wav"
            assert filename == "sample.wav"
            assert content_type == "audio/wav"
            return {
                "asr_text": "今天真的很累，感觉没人理解我。",
                "audio_prediction": "sad",
                "text_prediction": "sad",
                "fusion_prediction": "sad",
                "audio_confidence": 0.74,
                "text_confidence": 0.71,
                "fusion_confidence": 0.86,
                "emotion_curve": [],
                "keywords": [{"keyword": "很累", "category": "fatigue"}],
                "emotion_change_points": [],
                "possible_reasons": [{"type": "fusion_conditioned_reason", "text": "可能与疲惫感有关。"}],
            }

    result = analyze_audio_bytes(
        b"fake wav",
        filename="sample.wav",
        content_type="audio/wav",
        analyzer=FakeAnalyzer(),
    )

    assert result["ok"] is True
    assert result["analysis"]["fusion_prediction"] == "sad"
    assert result["agent_feedback"]["provider"] == "local_template"


def test_audio_upload_uses_unified_companion_prompt_and_persists_once(tmp_path) -> None:
    calls: list[dict] = []

    class FakeAnalyzer:
        def analyze(self, *_args, **_kwargs) -> dict:
            return {
                "asr_text": "今天真的很累。",
                "audio_prediction": "sad",
                "text_prediction": "sad",
                "fusion_prediction": "sad",
                "fusion_confidence": 0.86,
                "emotion_curve": [],
                "keywords": [{"keyword": "很累", "category": "fatigue"}],
                "emotion_change_points": [],
                "possible_reasons": [{"text": "可能与疲惫感有关。"}],
            }

    def fake_post(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
        calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        return {"choices": [{"message": {"role": "assistant", "content": "听起来你今天很累，我们可以先慢一点。"}}]}

    memory = MemoryService(tmp_path / "memory.sqlite3")
    agent = EmotionCompanionAgent(
        config=DeepSeekConfig(api_key="test-key"),
        post_json=fake_post,
    )
    app = create_app(
        analyzer=FakeAnalyzer(),
        agent=agent,
        memory=memory,
        frontend_dist=None,
        security=SecuritySettings(llm_budget_db=str(tmp_path / "quota.sqlite3")),
    )

    from fastapi.testclient import TestClient

    response = TestClient(app).post(
        "/api/analyze-audio",
        files={"file": ("sample.wav", b"audio", "audio/wav")},
        headers={"X-User-Id": "u1", "X-Session-Id": "s1", "X-Request-Id": "audio-1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_feedback"] == {
        "companion_response": "听起来你今天很累，我们可以先慢一点。",
        "provider": "deepseek",
    }
    assert len(calls) == 1
    assert calls[0]["url"].endswith("/chat/completions")
    assert "心伴" in calls[0]["payload"]["messages"][0]["content"]
    assert "中文语音情绪分析原型系统" not in json.dumps(calls[0]["payload"], ensure_ascii=False)
    assert "response_format" not in calls[0]["payload"]
    assert "local_fusion_prediction" in calls[0]["payload"]["messages"][-1]["content"]
    messages = memory.get_messages("u1", "s1", limit=20)
    assert [(item["role"], item["content"]) for item in messages] == [
        ("user", "今天真的很累。"),
        ("assistant", "听起来你今天很累，我们可以先慢一点。"),
    ]
    assert memory.summary("u1", "s1")["long_term"]["trend"]["event_count"] == 1


def test_audio_decoder_discovers_imageio_ffmpeg_when_system_path_missing() -> None:
    decoder = AudioBytesDecoder()

    assert decoder.ffmpeg_path is not None
    assert decoder.supported_extensions >= {".wav", ".mp3", ".m4a", ".flac", ".webm"}


def test_temp_file_cleanup_ignores_windows_file_lock() -> None:
    class LockedPath:
        def unlink(self, missing_ok: bool = False) -> None:
            error = PermissionError("file is locked")
            error.winerror = 32
            raise error

    _remove_temp_file(LockedPath())


def test_audio_decoder_uses_ffmpeg_without_ffprobe(monkeypatch) -> None:
    captured = {}

    class FakeCompletedProcess:
        returncode = 0
        stderr = b""
        stdout = b"\x00\x00\x00\x00\x00\x00\x80?"

    def fake_run(command, capture_output: bool, check: bool):
        captured["command"] = command
        captured["capture_output"] = capture_output
        captured["check"] = check
        return FakeCompletedProcess()

    monkeypatch.setattr("backend.runtime_pipeline.subprocess.run", fake_run)
    decoder = AudioBytesDecoder(ffmpeg_path="ffmpeg-test.exe")

    waveform, sample_rate = decoder._decode_with_ffmpeg(b"audio", suffix=".m4a")

    assert sample_rate == 16000
    assert waveform.tolist() == [0.0, 1.0]
    assert captured["command"][0] == "ffmpeg-test.exe"
    assert "ffprobe" not in captured["command"]
    assert captured["capture_output"] is True
    assert captured["check"] is False


def test_predict_with_pipeline_maps_probabilities_to_label_names() -> None:
    class FakePipeline:
        def predict_proba(self, features):
            assert features.shape == (1, 3)
            return [[0.2, 0.7, 0.1]]

    result = predict_with_pipeline(FakePipeline(), [1.0, 2.0, 3.0], ["angry", "sad", "neutral"])

    assert result == PredictionResult(
        prediction="sad",
        confidence=0.7,
        probabilities={"angry": 0.2, "sad": 0.7, "neutral": 0.1},
    )


def test_runtime_pipeline_builds_full_analysis_with_injected_components() -> None:
    class FakeAudioExtractor:
        def extract(self, waveform, sample_rate):
            assert sample_rate == 16000
            return [0.1, 0.2]

    class FakeTranscriber:
        def transcribe(self, waveform, sample_rate):
            return "今天真的很累，感觉没人理解我。"

    class FakeTextExtractor:
        def extract_batch(self, texts):
            assert texts == ["今天真的很累，感觉没人理解我。"]
            return [[0.3, 0.4, 0.5]]

    class FakeAudioClassifier:
        def predict_proba(self, features):
            return [[0.1, 0.8, 0.1]]

    class FakeTextClassifier:
        def predict_proba(self, features):
            return [[0.2, 0.7, 0.1]]

    class FakeFusionClassifier:
        def predict_proba(self, features):
            assert features.shape == (1, 5)
            return [[0.05, 0.9, 0.05]]

    runtime = EmotionBackendRuntime(
        audio_extractor=FakeAudioExtractor(),
        transcriber=FakeTranscriber(),
        text_extractor=FakeTextExtractor(),
        audio_classifier=FakeAudioClassifier(),
        text_classifier=FakeTextClassifier(),
        fusion_classifier=FakeFusionClassifier(),
        label_names=["angry", "sad", "neutral"],
    )

    analysis = runtime.analyze_waveform([0.0, 0.1, -0.1], sample_rate=16000)

    assert analysis["asr_text"] == "今天真的很累，感觉没人理解我。"
    assert analysis["audio_prediction"] == "sad"
    assert analysis["text_prediction"] == "sad"
    assert analysis["fusion_prediction"] == "sad"
    assert analysis["fusion_confidence"] == 0.9
    assert any(entry["keyword"] == "很累" for entry in analysis["keywords"])
    assert any(reason["type"] == "fusion_conditioned_reason" for reason in analysis["possible_reasons"])


def test_fastapi_app_health_and_upload_use_injected_analyzer(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    class FakeAnalyzer:
        def analyze(self, audio_bytes: bytes, filename: str | None = None, content_type: str | None = None) -> dict:
            return {
                "asr_text": "今天真的很累。",
                "audio_prediction": "sad",
                "text_prediction": "sad",
                "fusion_prediction": "sad",
                "audio_confidence": 0.7,
                "text_confidence": 0.7,
                "fusion_confidence": 0.8,
                "emotion_curve": [],
                "keywords": [{"keyword": "很累", "category": "fatigue"}],
                "emotion_change_points": [],
                "possible_reasons": [{"type": "fusion_conditioned_reason", "text": "可能与疲惫感有关。"}],
            }

    app = create_app(
        analyzer=FakeAnalyzer(),
        agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key=None)),
    )
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["deepseek"]["configured"] is False

    response = client.post(
        "/api/analyze-audio",
        files={"file": ("sample.wav", b"fake wav", "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["analysis"]["fusion_prediction"] == "sad"
    assert body["agent_feedback"]["provider"] == "local_template"


def test_fastapi_app_serves_react_dist_index(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    frontend_dist = tmp_path / "dist"
    frontend_dist.mkdir()
    (frontend_dist / "index.html").write_text(
        '<!doctype html><div id="root">React Frontend</div>',
        encoding="utf-8",
    )

    app = create_app(
        analyzer=object(),
        agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key=None)),
        frontend_dist=frontend_dist,
    )
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "React Frontend" in response.text


def test_fastapi_app_keeps_api_json_when_serving_react_dist(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    frontend_dist = tmp_path / "dist"
    frontend_dist.mkdir()
    (frontend_dist / "index.html").write_text("<div>React</div>", encoding="utf-8")

    app = create_app(
        analyzer=object(),
        agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key=None)),
        frontend_dist=frontend_dist,
    )
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["service"] == "emo-agent-local-backend"


def test_fastapi_app_accepts_raw_audio_body(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    seen = {}

    class FakeAnalyzer:
        def analyze(self, audio_bytes: bytes, filename: str | None = None, content_type: str | None = None) -> dict:
            seen["audio_bytes"] = audio_bytes
            seen["filename"] = filename
            seen["content_type"] = content_type
            return {
                "asr_text": "raw body",
                "audio_prediction": "neutral",
                "text_prediction": "neutral",
                "fusion_prediction": "neutral",
                "audio_confidence": 0.7,
                "text_confidence": 0.7,
                "fusion_confidence": 0.8,
                "emotion_curve": [],
                "keywords": [],
                "emotion_change_points": [],
                "possible_reasons": [],
            }

    app = create_app(
        analyzer=FakeAnalyzer(),
        agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key=None)),
    )
    client = TestClient(app)

    response = client.post(
        "/api/analyze-audio",
        content=b"raw audio",
        headers={"Content-Type": "audio/mp4", "X-Filename": "recording.m4a"},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert seen == {
        "audio_bytes": b"raw audio",
        "filename": "recording.m4a",
        "content_type": "audio/mp4",
    }


def test_fastapi_app_returns_json_for_analyzer_errors(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    class BrokenAnalyzer:
        def analyze(self, audio_bytes: bytes, filename: str | None = None, content_type: str | None = None) -> dict:
            raise ValueError("decode failed")

    app = create_app(
        analyzer=BrokenAnalyzer(),
        agent=EmotionCompanionAgent(config=DeepSeekConfig(api_key=None)),
    )
    client = TestClient(app)

    response = client.post(
        "/api/analyze-audio",
        files={"file": ("recording.webm", b"bad audio", "audio/webm")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body == {"ok": False, "error": "ValueError", "message": "decode failed"}


def test_upload_file_selection_updates_audio_player_preview() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_source = (project_root / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")

    assert "handleFileChange" in app_source
    assert "const previewUrl = setPreview(file)" in app_source
    assert "URL.createObjectURL(blob)" in app_source
    assert "audioUrlsRef.current.forEach" in app_source
    assert 'className="message-audio"' in app_source
    assert "message.audioUrl" in app_source
    assert "chat-audio-preview" not in app_source


def test_browser_recording_uses_supported_mime_type_and_filename() -> None:
    project_root = Path(__file__).resolve().parents[1]
    recording_source = (project_root / "frontend" / "src" / "recording.ts").read_text(encoding="utf-8")
    api_source = (project_root / "frontend" / "src" / "api.ts").read_text(encoding="utf-8")
    app_source = (project_root / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")

    assert "createMediaRecorder" in recording_source
    assert "MediaRecorder.isTypeSupported" in recording_source
    assert "window.MediaRecorder" in recording_source
    assert "audio/mp4" in recording_source
    assert "extensionForMimeType(mimeType)" in app_source
    assert "runAnalysis(blob, filename, previewUrl)" in app_source
    assert "raw_fetch" in api_source
    assert "X-Filename" in api_source
    assert "stage=" in api_source


def test_react_frontend_keeps_recording_and_upload_fallbacks() -> None:
    project_root = Path(__file__).resolve().parents[1]
    recording_source = (project_root / "frontend" / "src" / "recording.ts").read_text(encoding="utf-8")
    api_source = (project_root / "frontend" / "src" / "api.ts").read_text(encoding="utf-8")
    app_source = (project_root / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")

    assert "MediaRecorder.isTypeSupported" in recording_source
    assert 'return "未连接麦克风"' in recording_source
    assert "Requested device not found" not in app_source
    assert "typedError.technicalDetail" in app_source
    assert "无法连接分析服务" in api_source
    assert "分析服务配置不完整" in api_source
    assert "window.MediaRecorder" in recording_source
    assert "audio/mp4" in recording_source
    assert 'ACCEPTED_AUDIO_EXTENSIONS = ".wav,.mp3,.m4a,.flac,.webm"' in recording_source
    assert 'aria-label="上传已有录音或音频文件"' in app_source
    assert 'aria-label={uiState === "recording" ? "停止录音并分析" : "开始麦克风录音"}' in app_source
    assert "runAnalysis(file, file.name, previewUrl)" in app_source
    assert "runAnalysis(blob, filename, previewUrl)" in app_source
    assert api_source.count('fetch("/api/analyze-audio"') == 2
    assert "raw_fetch" in api_source
    assert "X-Filename" in api_source
    assert "stage=" in api_source


def test_figma_layout_keeps_text_input_and_removes_extra_controls() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_source = (project_root / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")

    assert '"用文字倾诉，或按麦克风说话…"' in app_source
    assert "会话历史" in app_source
    assert "快速标记情绪" in app_source
    assert "舒缓工具" in app_source
    assert "♀ 她" not in app_source
    assert "♂ 他" not in app_source
    assert "type Gender" not in app_source
    assert "setGender" not in app_source
    assert "隐私保护" not in app_source
    assert "DeepSeek 已连接" not in app_source


def test_frontend_exposes_search_evidence_and_formats_basic_markdown() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_source = (project_root / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    api_source = (project_root / "frontend" / "src" / "api.ts").read_text(encoding="utf-8")

    assert 'event === "tool_start" || event === "tool_result"' in api_source
    assert "正在使用 DeepSeek 原生联网搜索" in app_source
    assert "已联网检索" in app_source
    assert "response.search_required && response.fallback_reason" in app_source
    assert "function messageBlocks" in app_source
    assert "Markdown single newlines are soft wraps" in app_source
    assert 'className="bullet-content"' in app_source
    assert "inlineMessageParts" in app_source
    assert 'part.startsWith("**")' in app_source
