"""Local backend service helpers for the Module 7 web demo."""

from __future__ import annotations

import json
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from backend.agent_service import AgentInput, DeepSeekConfig, EmotionCompanionAgent
from backend.runtime_pipeline import LazyFullAudioAnalyzer
from backend.tool_service import WebSearchConfig


class PlaceholderAudioAnalyzer:
    """Replaceable analyzer until the full Whisper runtime pipeline is wired."""

    def analyze(self, audio_bytes: bytes, filename: str | None = None, content_type: str | None = None) -> dict[str, Any]:
        if not audio_bytes:
            raise ValueError("audio_bytes must not be empty.")
        return {
            "asr_text": "",
            "audio_prediction": None,
            "text_prediction": None,
            "fusion_prediction": "unknown",
            "audio_confidence": None,
            "text_confidence": None,
            "fusion_confidence": None,
            "emotion_curve": [],
            "keywords": [],
            "emotion_change_points": [],
            "possible_reasons": [
                {
                    "type": "backend_placeholder",
                    "text": "后端服务已接收音频，但真实 Whisper 分类和 ASR 推理管线尚未接入。",
                }
            ],
            "metadata": {
                "filename": filename,
                "content_type": content_type,
                "audio_bytes": len(audio_bytes),
            },
        }


def _parse_multipart_audio(raw: bytes, content_type: str) -> tuple[bytes, str | None, str | None]:
    """Extract the first form-data field named ``file`` without the removed cgi module."""
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=email_policy).parsebytes(header + raw)
    if not message.is_multipart():
        raise ValueError("request body is not valid multipart form data.")
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        if part.get_param("name", header="content-disposition") != "file":
            continue
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            payload = b""
        return payload, part.get_filename(), part.get_content_type()
    raise ValueError("multipart form must include a file field named 'file'.")


def build_health_status(
    config: DeepSeekConfig | None = None,
    search_config: WebSearchConfig | None = None,
) -> dict[str, Any]:
    config = config or DeepSeekConfig.from_env()
    search_config = search_config or WebSearchConfig.from_env()
    return {
        "status": "ok",
        "service": "emo-agent-local-backend",
        "deepseek": {
            "configured": config.configured,
            "base_url": config.base_url,
            "model": config.model,
        },
        "web_search": {
            "configured": search_config.configured,
            "provider": "zhipu",
            "engine": search_config.engine,
            "default_results": search_config.default_results,
            "max_results": search_config.max_results,
        },
        "endpoints": [
            "/api/health",
            "/api/analyze-audio",
            "/api/agent/chat",
            "/api/agent/chat/stream",
        ],
    }


def analyze_audio_bytes(
    audio_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
    analyzer: Any | None = None,
    agent: EmotionCompanionAgent | None = None,
) -> dict[str, Any]:
    analyzer = analyzer or PlaceholderAudioAnalyzer()
    agent = agent or EmotionCompanionAgent()
    analysis = analyzer.analyze(audio_bytes, filename=filename, content_type=content_type)
    feedback = agent.generate(AgentInput.from_mapping(analysis))
    return {
        "ok": True,
        "analysis": analysis,
        "agent_feedback": feedback,
    }


def create_handler(analyzer: Any | None = None, agent: EmotionCompanionAgent | None = None) -> type[BaseHTTPRequestHandler]:
    analyzer = analyzer or LazyFullAudioAnalyzer()
    agent = agent or EmotionCompanionAgent()

    class EmoAgentHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/index.html"):
                self._send(200, INDEX_HTML, content_type="text/html; charset=utf-8")
                return
            if self.path.startswith("/api/health"):
                self._send_json(200, build_health_status(agent.config))
                return
            self._send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            if not self.path.startswith("/api/analyze-audio"):
                self._send_json(404, {"ok": False, "error": "not_found"})
                return
            try:
                audio_bytes, filename, content_type = self._read_audio_upload()
                result = analyze_audio_bytes(
                    audio_bytes,
                    filename=filename,
                    content_type=content_type,
                    analyzer=analyzer,
                    agent=agent,
                )
                self._send_json(200, result)
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": type(exc).__name__, "message": str(exc)})

        def _read_audio_upload(self) -> tuple[bytes, str | None, str | None]:
            content_type = self.headers.get("Content-Type", "application/octet-stream")
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("Request body is empty.")
            raw = self.rfile.read(length)
            if content_type.startswith("multipart/form-data"):
                return _parse_multipart_audio(raw, content_type)
            return raw, self.headers.get("X-Filename"), content_type

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False), content_type="application/json; charset=utf-8")

        def _send(self, status: int, body: str, content_type: str) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return EmoAgentHandler


def run_server(host: str = "127.0.0.1", port: int = 7860, analyzer: Any | None = None) -> None:
    server = ThreadingHTTPServer((host, port), create_handler(analyzer=analyzer))
    print(f"Emo Agent backend running at http://{host}:{port}")
    server.serve_forever()


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Emo Agent</title>
  <style>
    :root { --ink: #23313f; --muted: #6b7785; --line: #dde7ee; --panel: rgba(255, 255, 255, 0.88); --soft-blue: #eaf6ff; --soft-mint: #edf9f4; --accent: #2f80ed; --shadow: 0 18px 50px rgba(56, 75, 98, 0.13); }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; color: var(--ink); font-family: "Microsoft YaHei", "Noto Sans SC", "PingFang SC", "Segoe UI", system-ui, sans-serif; background: radial-gradient(circle at 12% 8%, rgba(96, 181, 255, 0.22), transparent 28%), radial-gradient(circle at 88% 0%, rgba(255, 132, 148, 0.18), transparent 30%), linear-gradient(135deg, #f7fbff 0%, #f3fbf6 48%, #fff8f9 100%); }
    main { width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 34px 0 42px; display: grid; gap: 18px; }
    header { display: grid; gap: 8px; padding: 10px 2px 4px; }
    h1 { margin: 0; font-size: clamp(30px, 5vw, 48px); line-height: 1.08; letter-spacing: 0; }
    .subtitle { max-width: 760px; margin: 0; color: var(--muted); font-size: 16px; line-height: 1.8; }
    .layout { display: grid; grid-template-columns: minmax(280px, 360px) 1fr; gap: 18px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid rgba(221, 231, 238, 0.9); border-radius: 8px; box-shadow: var(--shadow); backdrop-filter: blur(12px); }
    .input-panel { padding: 18px; display: grid; gap: 16px; position: sticky; top: 16px; }
    .input-block { display: grid; gap: 10px; padding: 14px; border: 1px solid var(--line); border-radius: 8px; background: rgba(255, 255, 255, 0.72); }
    .block-title { margin: 0; font-size: 15px; font-weight: 700; }
    .hint { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.6; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    button, input { font: inherit; }
    input[type="file"] { width: 100%; padding: 10px; border: 1px dashed #adc3d4; border-radius: 8px; background: #f8fcff; color: var(--muted); }
    button { min-height: 40px; padding: 9px 13px; border: 1px solid #b9c8d8; background: #ffffff; color: var(--ink); border-radius: 8px; cursor: pointer; transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease; }
    button:hover:not(:disabled) { transform: translateY(-1px); border-color: #83a9cc; box-shadow: 0 8px 22px rgba(47, 128, 237, 0.13); }
    button.primary { border-color: transparent; background: linear-gradient(135deg, #2f80ed, #27ae8f); color: white; font-weight: 700; }
    button.soft { background: #f2fbf7; border-color: #b8dfcf; color: #1d7056; }
    button:disabled { opacity: 0.55; cursor: not-allowed; transform: none; box-shadow: none; }
    audio { width: 100%; height: 42px; }
    .status { min-height: 44px; padding: 12px 14px; border-radius: 8px; background: var(--soft-blue); color: #28516f; line-height: 1.55; border: 1px solid #cfe7f7; }
    .status.busy { background: #fff7e8; border-color: #f3d69a; color: #7a5722; }
    .status.error { background: #fff0f3; border-color: #f4bdc7; color: #9b2f43; }
    .results { padding: 18px; display: grid; gap: 16px; min-height: 460px; }
    .empty-state { min-height: 420px; display: grid; place-items: center; text-align: center; color: var(--muted); background: linear-gradient(135deg, rgba(234, 246, 255, 0.9), rgba(237, 249, 244, 0.9)); border: 1px dashed #b8ccda; border-radius: 8px; padding: 28px; }
    .empty-state strong { display: block; margin-bottom: 8px; color: var(--ink); font-size: 20px; }
    .summary-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .emotion-card, .feedback-card, .text-card, .debug-card { border: 1px solid var(--line); border-radius: 8px; background: rgba(255, 255, 255, 0.82); overflow: hidden; }
    .emotion-card { display: grid; gap: 10px; padding: 14px; border-top: 4px solid var(--emotion-color, #7c94a8); }
    .card-label { color: var(--muted); font-size: 13px; font-weight: 700; }
    .emotion-name { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; font-size: 24px; font-weight: 800; color: var(--emotion-color, var(--ink)); }
    .confidence { font-size: 13px; color: var(--muted); font-weight: 700; white-space: nowrap; }
    .bar-list { display: grid; gap: 8px; }
    .bar-row { display: grid; grid-template-columns: 92px 1fr 52px; gap: 10px; align-items: center; font-size: 13px; }
    .track { height: 9px; border-radius: 999px; background: #edf2f5; overflow: hidden; }
    .fill { height: 100%; width: var(--value); background: var(--emotion-color, var(--accent)); border-radius: inherit; }
    .section-title { margin: 0; font-size: 18px; line-height: 1.35; }
    .feedback-card { padding: 16px; display: grid; gap: 12px; }
    .feedback-highlight { padding: 14px; border-radius: 8px; background: linear-gradient(135deg, var(--soft-mint), #f6fcff); border: 1px solid #cfe8dc; line-height: 1.8; font-size: 16px; }
    .feedback-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .mini-card { padding: 13px; border-radius: 8px; border: 1px solid var(--line); background: #ffffff; line-height: 1.7; }
    .mini-card strong { display: block; margin-bottom: 5px; color: var(--ink); }
    .text-card { padding: 16px; display: grid; gap: 10px; }
    .asr-text { margin: 0; padding: 13px; border-radius: 8px; background: #f7fbff; border: 1px solid #d6e8f5; line-height: 1.8; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip { display: inline-flex; align-items: center; min-height: 28px; padding: 5px 10px; border-radius: 999px; color: #245071; background: #eaf6ff; border: 1px solid #cbe5f8; font-size: 13px; font-weight: 700; }
    .chip.reason { color: #6f4b12; background: #fff7e8; border-color: #efd99f; font-weight: 500; }
    .debug-card { background: #fbfdff; }
    details summary { cursor: pointer; padding: 13px 16px; color: var(--muted); font-weight: 700; }
    pre { margin: 0; padding: 0 16px 16px; max-height: 360px; overflow: auto; color: #203042; font-family: "Cascadia Mono", "Consolas", monospace; font-size: 12px; line-height: 1.55; white-space: pre-wrap; }
    @media (max-width: 880px) { main { width: min(100% - 22px, 1120px); padding-top: 22px; } .layout { grid-template-columns: 1fr; } .input-panel { position: static; } .summary-grid, .feedback-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>中文语音情绪陪伴智能体</h1>
      <p class="subtitle">上传或录制一段中文语音，系统会完成 ASR 转写、声学/文本/融合情绪识别、关键词原因分析，并生成一段温和的陪伴式反馈。</p>
    </header>
    <div class="layout">
      <section class="panel input-panel" aria-label="音频输入">
        <div class="input-block">
          <h2 class="block-title">上传音频</h2>
          <p class="hint">支持 wav、mp3、m4a、flac、webm 等常见音频格式。</p>
          <input id="file" type="file" accept="audio/*" />
          <button id="submit" class="primary">分析上传音频</button>
        </div>
        <div class="input-block">
          <h2 class="block-title">浏览器录音</h2>
          <div class="toolbar">
            <button id="record" class="soft">开始录音</button>
            <button id="stop" disabled>停止录音</button>
            <button id="analyze-recording" class="primary" disabled>分析录音</button>
          </div>
          <audio id="player" controls></audio>
          <p class="hint">可以先试听，再提交分析。首次分析会加载模型，等待时间会稍长。</p>
        </div>
        <div id="status" class="status">请选择音频文件，或点击开始录音。</div>
      </section>
      <section id="output" class="panel results" aria-live="polite">
        <div class="empty-state"><div><strong>结果会在这里变成可读的彩色卡片</strong><span>不用再从一大段 JSON 里找情绪、置信度和建议了。</span></div></div>
      </section>
    </div>
  </main>
  <script>
    const fileInput = document.getElementById('file');
    const output = document.getElementById('output');
    const statusBox = document.getElementById('status');
    const player = document.getElementById('player');
    const recordButton = document.getElementById('record');
    const stopButton = document.getElementById('stop');
    const analyzeRecordingButton = document.getElementById('analyze-recording');
    let recorder = null;
    let chunks = [];
    let recordingBlob = null;
    let recordingFilename = 'recording.webm';
    let selectedFileUrl = null;
    const emotionMeta = {
      angry: { label: '愤怒', color: '#e25555' },
      fearful: { label: '害怕', color: '#8c6be8' },
      happy: { label: '开心', color: '#f0a51f' },
      neutral: { label: '中性', color: '#4f8fc6' },
      playfulness: { label: '玩笑感', color: '#25a983' },
      sad: { label: '难过', color: '#4d74d9' },
      surprise: { label: '惊讶', color: '#e06aa7' },
      unknown: { label: '未知', color: '#7c94a8' }
    };
    function escapeHtml(value) {
      return String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#039;');
    }
    function emotionInfo(name) {
      return emotionMeta[name] || { label: name || '未知', color: '#7c94a8' };
    }
    function formatPercent(value) {
      if (typeof value !== 'number' || Number.isNaN(value)) return '暂无';
      return `${(value * 100).toFixed(value >= 0.995 ? 2 : 1)}%`;
    }
    function setStatus(message, type = 'info') {
      statusBox.className = `status ${type === 'busy' ? 'busy' : type === 'error' ? 'error' : ''}`;
      statusBox.textContent = message;
    }
    function setPlayerBlob(blob) {
      if (selectedFileUrl) URL.revokeObjectURL(selectedFileUrl);
      selectedFileUrl = URL.createObjectURL(blob);
      player.src = selectedFileUrl;
    }
    function canRecordAudio() {
      return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
    }
    function explainRecordingUnavailable() {
      const host = window.location.hostname;
      const isLocal = host === 'localhost' || host === '127.0.0.1' || host === '::1';
      if (window.location.protocol !== 'https:' && !isLocal) {
        return '当前是局域网 HTTP 访问，浏览器不会开放麦克风录音。可以先用上传音频；如果要在局域网录音，需要改成 HTTPS 访问。';
      }
      return '当前浏览器没有开放麦克风录音接口。可以先用上传音频，或换 Chrome / Edge 再试。';
    }
    function updateRecordingAvailability() {
      if (canRecordAudio()) return;
      recordButton.disabled = true;
      stopButton.disabled = true;
      analyzeRecordingButton.disabled = true;
      setStatus(explainRecordingUnavailable(), 'error');
    }
    function extensionForMimeType(mimeType) {
      const clean = String(mimeType || '').split(';')[0].trim().toLowerCase();
      if (clean === 'audio/mp4' || clean === 'audio/aac' || clean === 'audio/x-m4a') return 'm4a';
      if (clean === 'audio/mpeg' || clean === 'audio/mp3') return 'mp3';
      if (clean === 'audio/wav' || clean === 'audio/wave') return 'wav';
      if (clean === 'audio/ogg') return 'ogg';
      return 'webm';
    }
    function selectRecorderFormat() {
      const candidates = ['audio/mp4', 'audio/aac', 'audio/webm;codecs=opus', 'audio/webm'];
      if (!window.MediaRecorder || typeof MediaRecorder.isTypeSupported !== 'function') {
        return { options: {}, mimeType: '', extension: 'webm' };
      }
      for (const mimeType of candidates) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          return { options: { mimeType }, mimeType, extension: extensionForMimeType(mimeType) };
        }
      }
      return { options: {}, mimeType: '', extension: 'webm' };
    }
    function createMediaRecorder(stream) {
      const formats = [
        selectRecorderFormat(),
        { options: { mimeType: 'audio/mp4' }, mimeType: 'audio/mp4', extension: 'm4a' },
        { options: { mimeType: 'audio/aac' }, mimeType: 'audio/aac', extension: 'm4a' },
        { options: { mimeType: 'audio/webm;codecs=opus' }, mimeType: 'audio/webm;codecs=opus', extension: 'webm' },
        { options: { mimeType: 'audio/webm' }, mimeType: 'audio/webm', extension: 'webm' },
        { options: {}, mimeType: '', extension: 'webm' }
      ];
      const seen = new Set();
      let lastError = null;
      for (const format of formats) {
        const key = format.mimeType || 'default';
        if (seen.has(key)) continue;
        seen.add(key);
        try {
          return { recorder: new MediaRecorder(stream, format.options), format };
        } catch (error) {
          lastError = error;
        }
      }
      throw lastError || new Error('当前浏览器无法创建录音器');
    }
    function probabilityRows(probabilities) {
      if (!probabilities || typeof probabilities !== 'object') return '<p class="hint">暂无概率分布。</p>';
      return Object.entries(probabilities).sort((a, b) => Number(b[1]) - Number(a[1])).map(([name, value]) => {
        const info = emotionInfo(name);
        const numeric = Number(value) || 0;
        return `<div class="bar-row" style="--emotion-color: ${info.color}"><span>${escapeHtml(info.label)}</span><span class="track"><span class="fill" style="--value: ${Math.max(0, Math.min(100, numeric * 100))}%"></span></span><span>${formatPercent(numeric)}</span></div>`;
      }).join('');
    }
    function predictionCard(title, prediction, confidence, probabilities) {
      const info = emotionInfo(prediction || 'unknown');
      return `<article class="emotion-card" style="--emotion-color: ${info.color}"><div class="card-label">${escapeHtml(title)}</div><div class="emotion-name"><span>${escapeHtml(info.label)}</span><span class="confidence">${formatPercent(confidence)}</span></div><div class="bar-list">${probabilityRows(probabilities)}</div></article>`;
    }
    function renderKeywords(keywords) {
      if (!Array.isArray(keywords) || keywords.length === 0) return '<span class="chip">暂无明确关键词</span>';
      return keywords.map(item => `<span class="chip">${escapeHtml(item.keyword || item.text || item.value || JSON.stringify(item))}</span>`).join('');
    }
    function renderReasons(reasons) {
      if (!Array.isArray(reasons) || reasons.length === 0) return '<span class="chip reason">暂无明确原因线索</span>';
      return reasons.map(item => `<span class="chip reason">${escapeHtml(item.text || item.reason || JSON.stringify(item))}</span>`).join('');
    }
    function renderFeedback(feedback) {
      if (!feedback || typeof feedback !== 'object') return '';
      return `<section class="feedback-card"><h2 class="section-title">陪伴反馈</h2><div class="feedback-highlight">${escapeHtml(feedback.companion_response || '暂未生成陪伴回应。')}</div><div class="feedback-grid"><div class="mini-card"><strong>主要情绪</strong>${escapeHtml(feedback.main_emotion_summary || '暂无')}</div><div class="mini-card"><strong>情绪曲线</strong>${escapeHtml(feedback.curve_interpretation || '暂无')}</div><div class="mini-card"><strong>可能原因</strong>${escapeHtml(feedback.possible_reasons || '暂无')}</div><div class="mini-card"><strong>温和建议</strong>${escapeHtml(feedback.gentle_suggestion || '暂无')}</div></div><p class="hint">${escapeHtml(feedback.safety_note || '')}</p></section>`;
    }
    function renderResult(data) {
      const analysis = data.analysis || {};
      const feedback = data.agent_feedback || {};
      output.innerHTML = `<section class="summary-grid">${predictionCard('声学情绪', analysis.audio_prediction, analysis.audio_confidence, analysis.audio_probabilities)}${predictionCard('文本情绪', analysis.text_prediction, analysis.text_confidence, analysis.text_probabilities)}${predictionCard('融合情绪', analysis.fusion_prediction, analysis.fusion_confidence, analysis.fusion_probabilities)}</section>${renderFeedback(feedback)}<section class="text-card"><h2 class="section-title">ASR 转写文本</h2><p class="asr-text">${escapeHtml(analysis.asr_text || '未识别到明确文本。')}</p><h2 class="section-title">关键词</h2><div class="chips">${renderKeywords(analysis.keywords)}</div><h2 class="section-title">可能原因线索</h2><div class="chips">${renderReasons(analysis.possible_reasons)}</div></section><section class="debug-card"><details><summary>查看原始 JSON 调试详情</summary><pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre></details></section>`;
    }
    async function analyzeBlob(blob, filename) {
      const form = new FormData();
      form.append('file', blob, filename);
      setStatus('分析中... 首次请求会加载模型，可能需要一点时间。', 'busy');
      try {
        const response = await fetch('/api/analyze-audio', { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok || data.ok === false) throw new Error(data.message || data.error || '分析失败');
        renderResult(data);
        setStatus('分析完成。你可以继续试听音频，或换一段重新分析。');
      } catch (error) {
        output.innerHTML = `<div class="empty-state"><div><strong>分析没有完成</strong><span>${escapeHtml(error.message || error)}</span></div></div>`;
        setStatus(`分析失败：${error.message || error}`, 'error');
      }
    }
    fileInput.onchange = () => {
      const file = fileInput.files[0];
      if (!file) return;
      recordingBlob = null;
      analyzeRecordingButton.disabled = true;
      setPlayerBlob(file);
      setStatus('已选择音频，可以先播放试听，也可以直接分析。');
    };
    document.getElementById('submit').onclick = async () => {
      const file = fileInput.files[0];
      if (!file) {
        setStatus('请先选择一个音频文件。', 'error');
        return;
      }
      await analyzeBlob(file, file.name);
    };
    recordButton.onclick = async () => {
      if (!canRecordAudio()) {
        setStatus(explainRecordingUnavailable(), 'error');
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        chunks = [];
        const createdRecorder = createMediaRecorder(stream);
        const recorderFormat = createdRecorder.format;
        recorder = createdRecorder.recorder;
        recordingFilename = `recording.${recorderFormat.extension}`;
        recorder.ondataavailable = event => {
          if (event.data.size > 0) chunks.push(event.data);
        };
        recorder.onstop = () => {
          const actualMimeType = recorder.mimeType || recorderFormat.mimeType;
          recordingFilename = `recording.${extensionForMimeType(actualMimeType)}`;
          recordingBlob = actualMimeType ? new Blob(chunks, { type: actualMimeType }) : new Blob(chunks);
          setPlayerBlob(recordingBlob);
          analyzeRecordingButton.disabled = false;
          stream.getTracks().forEach(track => track.stop());
          setStatus('录音已停止，可以先试听或直接分析。');
        };
        recorder.start();
        setStatus('录音中... 说完后点击停止录音。', 'busy');
        recordButton.disabled = true;
        stopButton.disabled = false;
        analyzeRecordingButton.disabled = true;
      } catch (error) {
        setStatus(`无法开始录音：${error.message || error}`, 'error');
      }
    };
    stopButton.onclick = () => {
      if (recorder && recorder.state !== 'inactive') recorder.stop();
      recordButton.disabled = false;
      stopButton.disabled = true;
    };
    analyzeRecordingButton.onclick = async () => {
      if (!recordingBlob) return;
      await analyzeBlob(recordingBlob, recordingFilename);
    };
    updateRecordingAvailability();
  </script>
  <script>
    function safeUploadFilename(filename, blob) {
      const fallback = `recording.${extensionForMimeType(blob.type)}`;
      const base = String(filename || fallback).split(/[\\/]/).pop() || fallback;
      return base.replace(/[^\\w.+-]/g, '_') || fallback;
    }
    async function analyzeBlob(blob, filename) {
      let stage = 'prepare';
      const safeFilename = safeUploadFilename(filename, blob);
      const apiUrl = `${window.location.origin}/api/analyze-audio`;
      setStatus('正在分析音频，首次加载模型可能需要一点时间。', 'busy');
      try {
        let response;
        try {
          stage = 'formdata';
          const form = new FormData();
          form.append('file', blob, safeFilename);
          stage = 'multipart_fetch';
          response = await fetch(apiUrl, { method: 'POST', body: form });
        } catch (formError) {
          stage = 'raw_fetch';
          response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
              'Content-Type': blob.type || 'application/octet-stream',
              'X-Filename': safeFilename
            },
            body: blob
          });
        }
        stage = 'read_response';
        const responseText = await response.text();
        let data;
        try {
          data = JSON.parse(responseText);
        } catch (jsonError) {
          throw new Error(`response_not_json: ${responseText.slice(0, 120)}`);
        }
        if (!response.ok || data.ok === false) throw new Error(data.message || data.error || 'analysis_failed');
        renderResult(data);
        setStatus('分析完成。你可以继续试听音频，或换一段重新分析。');
      } catch (error) {
        const detail = `stage=${stage}; type=${blob.type || 'unknown'}; size=${blob.size || 0}; filename=${safeFilename}; error=${error.message || error}`;
        output.innerHTML = `<div class="empty-state"><div><strong>分析没有完成</strong><span>${escapeHtml(detail)}</span></div></div>`;
        setStatus(`分析失败：${detail}`, 'error');
      }
    }
  </script>
</body>
</html>
"""


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>中文语音情绪陪伴智能体</title>
  <style>
    :root {
      --ink: #26313c;
      --muted: #71808d;
      --line: #e4edf0;
      --panel: rgba(255, 255, 255, 0.88);
      --accent: #2f80ed;
      --accent-dark: #1f6f8f;
      --soft: #eaf6ff;
      --mist: #f3faf8;
      --wash: #f7fbfc;
      --shadow: 0 20px 60px rgba(56, 75, 98, 0.14);
      --ease-apple: cubic-bezier(0.22, 1, 0.36, 1);
      --ease-pop: cubic-bezier(0.2, 0.9, 0.2, 1.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Microsoft YaHei", "Noto Sans SC", "PingFang SC", "Segoe UI", system-ui, sans-serif;
      background:
        radial-gradient(circle at 12% 8%, rgba(96, 181, 255, 0.22), transparent 28%),
        radial-gradient(circle at 88% 0%, rgba(52, 184, 151, 0.16), transparent 30%),
        linear-gradient(135deg, #f7fbff 0%, #f3fbf6 48%, #f8fbff 100%);
    }
    main { width: min(1160px, calc(100% - 32px)); margin: 0 auto; padding: 34px 0 44px; display: grid; gap: 20px; }
    header { display: grid; gap: 10px; padding: 10px 2px 6px; }
    h1 {
      width: fit-content;
      margin: 0;
      max-width: 850px;
      font-family: "寒蝉全圆体", "寒蝉全圆体 Regular", "HarmonyOS Sans SC", "MiSans", "Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans SC", "PingFang SC", system-ui, sans-serif;
      font-size: clamp(34px, 5.1vw, 60px);
      font-weight: 900;
      line-height: 1.16;
      letter-spacing: 0;
      color: #2f3944;
      background: linear-gradient(100deg, #1f2933 0%, #315d7d 52%, #276d62 100%);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      text-shadow: 0 16px 36px rgba(47, 128, 237, 0.13);
      transform-origin: left center;
      transition: transform 0.55s var(--ease-apple), filter 0.55s var(--ease-apple), text-shadow 0.55s var(--ease-apple);
      cursor: default;
    }
    h1:hover {
      transform: scale(1.055) translateY(-2px);
      filter: saturate(1.08);
      text-shadow: 0 24px 48px rgba(47, 128, 237, 0.2);
    }
    .subtitle { max-width: 760px; margin: 0; color: var(--muted); font-size: 16px; line-height: 1.85; }
    .layout { display: grid; grid-template-columns: minmax(280px, 360px) 1fr; gap: 18px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid rgba(228, 237, 240, 0.95); border-radius: 14px; box-shadow: var(--shadow); backdrop-filter: blur(14px); transition: transform 0.5s var(--ease-apple), box-shadow 0.5s var(--ease-apple), border-color 0.5s var(--ease-apple); }
    .panel:hover { border-color: rgba(207, 231, 247, 0.95); box-shadow: 0 28px 72px rgba(56, 75, 98, 0.16); }
    .input-panel { padding: 18px; display: grid; gap: 16px; position: sticky; top: 16px; }
    .input-block { display: grid; gap: 10px; padding: 15px; border: 1px solid var(--line); border-radius: 14px; background: rgba(255, 255, 255, 0.78); transition: transform 0.42s var(--ease-apple), background 0.42s var(--ease-apple), border-color 0.42s var(--ease-apple); }
    .input-block:hover { transform: translateY(-2px); border-color: #cfe7f7; background: rgba(255, 255, 255, 0.92); }
    .block-title, .section-title { margin: 0; font-weight: 800; letter-spacing: 0; }
    .block-title { font-size: 15px; }
    .section-title { font-size: 18px; line-height: 1.35; }
    .hint { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.65; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    button, input { font: inherit; }
    input[type="file"] { width: 100%; padding: 10px; border: 1px dashed #adc3d4; border-radius: 14px; background: #f8fcff; color: var(--muted); }
    button { position: relative; isolation: isolate; overflow: hidden; min-height: 40px; padding: 9px 14px; border: 1px solid #b9c8d8; background: #fff; color: var(--ink); border-radius: 14px; cursor: pointer; transition: transform 0.34s var(--ease-pop), box-shadow 0.34s var(--ease-apple), border-color 0.34s var(--ease-apple), background 0.34s var(--ease-apple); }
    button::after { content: ""; position: absolute; inset: -40% auto -40% -70%; z-index: -1; width: 54%; transform: translateX(-120%) rotate(18deg); background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.55), transparent); transition: transform 0.72s var(--ease-apple); }
    button:hover:not(:disabled) { transform: translateY(-3px) scale(1.018); border-color: #83a9cc; box-shadow: 0 16px 34px rgba(47, 128, 237, 0.17); }
    button:hover:not(:disabled)::after { transform: translateX(360%) rotate(18deg); }
    button:active:not(:disabled) { transform: translateY(1px) scale(0.985); }
    button.primary { border-color: transparent; background: linear-gradient(135deg, var(--accent), var(--accent-dark)); color: white; font-weight: 800; }
    button.soft { background: #f2fbf7; border-color: #b8dfcf; color: #1d7056; }
    button:disabled { opacity: 0.55; cursor: not-allowed; transform: none; box-shadow: none; }
    .button-icon {
      display: inline-grid;
      width: 22px;
      height: 22px;
      margin-right: 7px;
      place-items: center;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.22);
      font-size: 14px;
      line-height: 1;
      transition: transform 0.36s var(--ease-pop), background 0.36s var(--ease-apple);
    }
    button:not(.primary) .button-icon { background: rgba(47, 128, 237, 0.1); color: var(--accent-dark); }
    button:hover:not(:disabled) .button-icon { transform: scale(1.38) rotate(-8deg); }
    button:active:not(:disabled) .button-icon { transform: scale(1.12); }
    audio { width: 100%; height: 42px; }
    .status { min-height: 44px; padding: 12px 14px; border-radius: 14px; background: #eaf6ff; color: #28516f; line-height: 1.55; border: 1px solid #cfe7f7; }
    .status.busy { background: #fff8ea; border-color: #f0dbab; color: #775923; }
    .status.error { background: #fff0f3; border-color: #f4bdc7; color: #9b2f43; }
    .results { padding: 18px; display: grid; gap: 16px; min-height: 460px; }
    .empty-state { min-height: 420px; display: grid; place-items: center; text-align: center; color: var(--muted); background: linear-gradient(135deg, rgba(234, 246, 255, 0.92), rgba(237, 249, 244, 0.92)); border: 1px dashed #b8ccda; border-radius: 14px; padding: 28px; }
    .empty-state strong { display: block; margin-bottom: 8px; color: var(--ink); font-size: 20px; }
    .summary-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .emotion-card, .feedback-card, .text-card, .curve-card { border: 1px solid var(--line); border-radius: 14px; background: rgba(255, 255, 255, 0.86); overflow: hidden; transition: transform 0.46s var(--ease-apple), box-shadow 0.46s var(--ease-apple), border-color 0.46s var(--ease-apple), background 0.46s var(--ease-apple); }
    .emotion-card:hover, .feedback-card:hover, .text-card:hover, .curve-card:hover { transform: translateY(-5px) scale(1.006); border-color: #cfe7f7; background: rgba(255, 255, 255, 0.94); box-shadow: 0 22px 48px rgba(56, 75, 98, 0.13); }
    .emotion-card { display: grid; gap: 10px; padding: 14px; border-top: 4px solid var(--emotion-color, #7c94a8); }
    .card-label { color: var(--muted); font-size: 13px; font-weight: 800; }
    .emotion-name { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; font-size: 24px; font-weight: 900; color: var(--emotion-color, var(--ink)); }
    .confidence { font-size: 13px; color: var(--muted); font-weight: 800; white-space: nowrap; }
    .bar-list { display: grid; gap: 8px; }
    .bar-row { display: grid; grid-template-columns: 92px 1fr 52px; gap: 10px; align-items: center; font-size: 13px; }
    .track { height: 9px; border-radius: 999px; background: #edf2f5; overflow: hidden; }
    .fill { height: 100%; width: var(--value); background: var(--emotion-color, var(--accent)); border-radius: inherit; }
    .feedback-card, .text-card, .curve-card { padding: 16px; display: grid; gap: 12px; }
    .feedback-highlight { padding: 15px; border-radius: 14px; background: linear-gradient(135deg, #edf9f4, #f6fcff); border: 1px solid #cfe8dc; line-height: 1.85; font-size: 16px; }
    .feedback-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .mini-card { padding: 13px; border-radius: 14px; border: 1px solid var(--line); background: #fff; line-height: 1.7; }
    .mini-card strong { display: block; margin-bottom: 5px; color: var(--ink); }
    .asr-text { margin: 0; padding: 13px; border-radius: 14px; background: #fbfdff; border: 1px solid #dbeaf0; line-height: 1.8; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip { display: inline-flex; align-items: center; min-height: 28px; padding: 5px 10px; border-radius: 999px; color: #245071; background: #eaf6ff; border: 1px solid #cbe5f8; font-size: 13px; font-weight: 800; }
    .chip.reason { color: #52685f; background: #f3faf8; border-color: #d2e7df; font-weight: 600; }
    .chip { transition: transform 0.16s ease, box-shadow 0.16s ease; }
    .chip:hover { transform: translateY(-1px) scale(1.03); box-shadow: 0 8px 18px rgba(159, 64, 84, 0.1); }
    .curve-wrap { width: 100%; overflow: hidden; border-radius: 14px; border: 1px solid #d8e7ef; background: linear-gradient(180deg, #fbfdff, #f6fbff); }
    .curve-svg { width: 100%; height: 210px; display: block; }
    .curve-line { fill: none; stroke-width: 4; stroke-linecap: round; stroke-linejoin: round; }
    .curve-area { opacity: 0.14; }
    .curve-point { stroke: white; stroke-width: 3; transition: r 0.28s var(--ease-pop), filter 0.28s var(--ease-apple); }
    .curve-point:hover { r: 8; filter: drop-shadow(0 8px 14px rgba(159, 64, 84, 0.22)); }
    .window-list { display: grid; gap: 8px; }
    .window-row { display: grid; grid-template-columns: 84px 92px 1fr 56px; gap: 10px; align-items: center; font-size: 13px; padding: 8px 0; border-top: 1px solid #eef3f6; }
    .window-row:first-child { border-top: 0; }
    .curve-note { padding: 13px; border-radius: 14px; background: #fff8ea; border: 1px solid #f0dbab; color: #73521f; line-height: 1.7; }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; }
      h1:hover, button:hover:not(:disabled), .emotion-card:hover, .feedback-card:hover, .text-card:hover, .curve-card:hover, .chip:hover { transform: none; }
    }
    @media (max-width: 880px) { main { width: min(100% - 22px, 1120px); padding-top: 22px; } .layout { grid-template-columns: 1fr; } .input-panel { position: static; } .summary-grid, .feedback-grid { grid-template-columns: 1fr; } .window-row { grid-template-columns: 1fr; gap: 6px; } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>中文语音情绪陪伴智能体</h1>
      <p class="subtitle">上传或录制一段中文语音，查看情绪识别、窗口变化、关键词线索和一段温和的陪伴式反馈。</p>
    </header>
    <div class="layout">
      <section class="panel input-panel" aria-label="音频输入">
        <div class="input-block">
          <h2 class="block-title">上传音频</h2>
          <p class="hint">支持 wav、mp3、m4a、flac、webm 等常见音频格式。</p>
          <input id="file" type="file" accept="audio/*" />
          <button id="submit" class="primary"><span class="button-icon">↑</span>分析上传音频</button>
        </div>
        <div class="input-block">
          <h2 class="block-title">浏览器录音</h2>
          <div class="toolbar">
            <button id="record" class="soft"><span class="button-icon">●</span>开始录音</button>
            <button id="stop" disabled><span class="button-icon">■</span>停止录音</button>
            <button id="analyze-recording" class="primary" disabled><span class="button-icon">↗</span>分析录音</button>
          </div>
          <audio id="player" controls></audio>
          <p class="hint">可以先试听，再提交分析。首次分析会加载模型，等待时间会稍长。</p>
        </div>
        <div id="status" class="status">请选择音频文件，或点击开始录音。</div>
      </section>
      <section id="output" class="panel results" aria-live="polite">
        <div class="empty-state"><div><strong>准备好后，我会把结果整理成清晰的陪伴分析</strong><span>包括主要情绪、窗口变化、关键词线索和具体建议。</span></div></div>
      </section>
    </div>
  </main>
  <script>
    const fileInput = document.getElementById('file');
    const output = document.getElementById('output');
    const statusBox = document.getElementById('status');
    const player = document.getElementById('player');
    const recordButton = document.getElementById('record');
    const stopButton = document.getElementById('stop');
    const analyzeRecordingButton = document.getElementById('analyze-recording');
    let recorder = null;
    let chunks = [];
    let recordingBlob = null;
    let recordingFilename = 'recording.webm';
    let selectedFileUrl = null;

    const emotionMeta = {
      angry: { label: '愤怒', color: '#e25555' },
      fearful: { label: '害怕', color: '#8c6be8' },
      happy: { label: '开心', color: '#f0a51f' },
      neutral: { label: '中性', color: '#4f8fc6' },
      playfulness: { label: '玩笑感', color: '#25a983' },
      sad: { label: '难过', color: '#4d74d9' },
      surprise: { label: '惊讶', color: '#e06aa7' },
      unknown: { label: '未知', color: '#7c94a8' }
    };

    function escapeHtml(value) {
      return String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#039;');
    }
    function emotionInfo(name) { return emotionMeta[name] || { label: name || '未知', color: '#7c94a8' }; }
    function formatPercent(value) {
      if (typeof value !== 'number' || Number.isNaN(value)) return '暂无';
      return `${(value * 100).toFixed(value >= 0.995 ? 2 : 1)}%`;
    }
    function setStatus(message, type = 'info') {
      statusBox.className = `status ${type === 'busy' ? 'busy' : type === 'error' ? 'error' : ''}`;
      statusBox.textContent = message;
    }
    function setPlayerBlob(blob) {
      if (selectedFileUrl) URL.revokeObjectURL(selectedFileUrl);
      selectedFileUrl = URL.createObjectURL(blob);
      player.src = selectedFileUrl;
    }
    function canRecordAudio() {
      return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
    }
    function explainRecordingUnavailable() {
      const host = window.location.hostname;
      const isLocal = host === 'localhost' || host === '127.0.0.1' || host === '::1';
      if (window.location.protocol !== 'https:' && !isLocal) {
        return '当前是局域网 HTTP 访问，浏览器不会开放麦克风录音。可以先用上传音频；如果要在局域网录音，需要改成 HTTPS 访问。';
      }
      return '当前浏览器没有开放麦克风录音接口。可以先用上传音频，或换 Chrome / Edge 再试。';
    }
    function updateRecordingAvailability() {
      if (canRecordAudio()) return;
      recordButton.disabled = true;
      stopButton.disabled = true;
      analyzeRecordingButton.disabled = true;
      setStatus(explainRecordingUnavailable(), 'error');
    }
    function extensionForMimeType(mimeType) {
      const clean = String(mimeType || '').split(';')[0].trim().toLowerCase();
      if (clean === 'audio/mp4' || clean === 'audio/aac' || clean === 'audio/x-m4a') return 'm4a';
      if (clean === 'audio/mpeg' || clean === 'audio/mp3') return 'mp3';
      if (clean === 'audio/wav' || clean === 'audio/wave') return 'wav';
      if (clean === 'audio/ogg') return 'ogg';
      return 'webm';
    }
    function selectRecorderFormat() {
      const candidates = ['audio/mp4', 'audio/aac', 'audio/webm;codecs=opus', 'audio/webm'];
      if (!window.MediaRecorder || typeof MediaRecorder.isTypeSupported !== 'function') {
        return { options: {}, mimeType: '', extension: 'webm' };
      }
      for (const mimeType of candidates) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          return { options: { mimeType }, mimeType, extension: extensionForMimeType(mimeType) };
        }
      }
      return { options: {}, mimeType: '', extension: 'webm' };
    }
    function createMediaRecorder(stream) {
      const formats = [
        selectRecorderFormat(),
        { options: { mimeType: 'audio/mp4' }, mimeType: 'audio/mp4', extension: 'm4a' },
        { options: { mimeType: 'audio/aac' }, mimeType: 'audio/aac', extension: 'm4a' },
        { options: { mimeType: 'audio/webm;codecs=opus' }, mimeType: 'audio/webm;codecs=opus', extension: 'webm' },
        { options: { mimeType: 'audio/webm' }, mimeType: 'audio/webm', extension: 'webm' },
        { options: {}, mimeType: '', extension: 'webm' }
      ];
      const seen = new Set();
      let lastError = null;
      for (const format of formats) {
        const key = format.mimeType || 'default';
        if (seen.has(key)) continue;
        seen.add(key);
        try {
          return { recorder: new MediaRecorder(stream, format.options), format };
        } catch (error) {
          lastError = error;
        }
      }
      throw lastError || new Error('当前浏览器无法创建录音器');
    }
    function probabilityRows(probabilities) {
      if (!probabilities || typeof probabilities !== 'object') return '<p class="hint">暂无概率分布。</p>';
      return Object.entries(probabilities).sort((a, b) => Number(b[1]) - Number(a[1])).map(([name, value]) => {
        const info = emotionInfo(name);
        const numeric = Number(value) || 0;
        return `<div class="bar-row" style="--emotion-color: ${info.color}"><span>${escapeHtml(info.label)}</span><span class="track"><span class="fill" style="--value: ${Math.max(0, Math.min(100, numeric * 100))}%"></span></span><span>${formatPercent(numeric)}</span></div>`;
      }).join('');
    }
    function predictionCard(title, prediction, confidence, probabilities) {
      const info = emotionInfo(prediction || 'unknown');
      return `<article class="emotion-card" style="--emotion-color: ${info.color}"><div class="card-label">${escapeHtml(title)}</div><div class="emotion-name"><span>${escapeHtml(info.label)}</span><span class="confidence">${formatPercent(confidence)}</span></div><div class="bar-list">${probabilityRows(probabilities)}</div></article>`;
    }
    function renderKeywords(keywords) {
      if (!Array.isArray(keywords) || keywords.length === 0) return '<span class="chip">暂无明确关键词</span>';
      return keywords.map(item => `<span class="chip">${escapeHtml(item.keyword || item.text || item.value || '关键词')}</span>`).join('');
    }
    function renderReasons(reasons) {
      if (!Array.isArray(reasons) || reasons.length === 0) return '<span class="chip reason">暂无明确原因线索</span>';
      return reasons.map(item => `<span class="chip reason">${escapeHtml(item.text || item.reason || '可能原因线索')}</span>`).join('');
    }
    function renderFeedback(feedback) {
      if (!feedback || typeof feedback !== 'object') return '';
      return `<section class="feedback-card"><h2 class="section-title">陪伴反馈</h2><div class="feedback-highlight">${escapeHtml(feedback.companion_response || '暂未生成陪伴回应。')}</div><div class="feedback-grid"><div class="mini-card"><strong>主要情绪</strong>${escapeHtml(feedback.main_emotion_summary || '暂无')}</div><div class="mini-card"><strong>情绪变化</strong>${escapeHtml(feedback.curve_interpretation || '暂无明显变化。')}</div><div class="mini-card"><strong>可能原因</strong>${escapeHtml(feedback.possible_reasons || '暂无')}</div><div class="mini-card"><strong>温和建议</strong>${escapeHtml(feedback.gentle_suggestion || '暂无')}</div></div><p class="hint">${escapeHtml(feedback.safety_note || '')}</p></section>`;
    }
    function windowTime(row, index) {
      const start = Number(row.start_time ?? row.start ?? row.time ?? index);
      const endValue = row.end_time ?? row.end;
      if (Number.isFinite(Number(endValue))) return `${Number(start).toFixed(1)}-${Number(endValue).toFixed(1)}s`;
      return `${Number(start).toFixed(1)}s`;
    }
    function rowPrediction(row) {
      if (row.pred_emotion) return row.pred_emotion;
      const ignored = new Set(['window_index', 'start_time', 'end_time', 'start', 'end', 'time', 'confidence', 'pred_emotion']);
      let best = 'unknown';
      let bestValue = -1;
      Object.entries(row || {}).forEach(([key, value]) => {
        const cleanKey = key.startsWith('prob_') ? key.slice(5) : key;
        const numeric = Number(value);
        if (!ignored.has(key) && emotionMeta[cleanKey] && Number.isFinite(numeric) && numeric > bestValue) {
          best = cleanKey;
          bestValue = numeric;
        }
      });
      return best;
    }
    function rowConfidence(row, prediction) {
      const direct = Number(row.confidence);
      if (Number.isFinite(direct)) return direct;
      const prob = Number(row[`prob_${prediction}`] ?? row[prediction]);
      return Number.isFinite(prob) ? prob : null;
    }
    function curveSvg(rows) {
      const points = rows.map((row, index) => {
        const prediction = rowPrediction(row);
        return { index, prediction, confidence: rowConfidence(row, prediction) ?? 0 };
      });
      const w = 720, h = 210, padX = 34, padY = 24;
      const denom = Math.max(1, points.length - 1);
      const coords = points.map(point => ({
        ...point,
        x: padX + (point.index / denom) * (w - padX * 2),
        y: h - padY - Math.max(0, Math.min(1, point.confidence)) * (h - padY * 2)
      }));
      const line = coords.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(' ');
      const area = `${line} L ${coords.at(-1).x.toFixed(1)} ${h - padY} L ${coords[0].x.toFixed(1)} ${h - padY} Z`;
      const stroke = emotionInfo(coords.at(-1).prediction).color;
      const circles = coords.map(point => `<circle class="curve-point" cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="6" fill="${emotionInfo(point.prediction).color}"><title>${emotionInfo(point.prediction).label} ${formatPercent(point.confidence)}</title></circle>`).join('');
      return `<div class="curve-wrap"><svg class="curve-svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="情绪窗口置信度变化曲线"><path d="${area}" class="curve-area" fill="${stroke}"></path><path d="${line}" class="curve-line" stroke="${stroke}"></path>${circles}</svg></div>`;
    }
    function renderEmotionCurve(analysis) {
      const rows = Array.isArray(analysis.emotion_curve) ? analysis.emotion_curve : [];
      const changes = Array.isArray(analysis.emotion_change_points) ? analysis.emotion_change_points : [];
      if (rows.length === 0) {
        return `<section class="curve-card"><h2 class="section-title">情绪窗口曲线</h2><div class="curve-note">这段音频暂时没有形成可展示的窗口曲线。通常较长音频会更容易观察到情绪随时间的变化。</div></section>`;
      }
      const windowRows = rows.slice(0, 8).map((row, index) => {
        const prediction = rowPrediction(row);
        const info = emotionInfo(prediction);
        const confidence = rowConfidence(row, prediction);
        return `<div class="window-row" style="--emotion-color: ${info.color}"><span>${escapeHtml(windowTime(row, index))}</span><strong style="color:${info.color}">${escapeHtml(info.label)}</strong><span class="track"><span class="fill" style="--value:${Math.max(0, Math.min(100, (confidence || 0) * 100))}%"></span></span><span>${formatPercent(confidence)}</span></div>`;
      }).join('');
      const changeText = changes.length > 0
        ? `检测到 ${changes.length} 个相对明显的变化点，可结合下方窗口片段一起看。`
        : '整体变化较平稳，未检测到特别突出的情绪转折点。';
      return `<section class="curve-card"><h2 class="section-title">情绪窗口曲线</h2>${curveSvg(rows)}<p class="hint">${escapeHtml(changeText)}</p><div class="window-list">${windowRows}</div></section>`;
    }
    function renderResult(data) {
      const analysis = data.analysis || {};
      const feedback = data.agent_feedback || {};
      output.innerHTML = `<section class="summary-grid">${predictionCard('声学情绪', analysis.audio_prediction, analysis.audio_confidence, analysis.audio_probabilities)}${predictionCard('文本情绪', analysis.text_prediction, analysis.text_confidence, analysis.text_probabilities)}${predictionCard('融合情绪', analysis.fusion_prediction, analysis.fusion_confidence, analysis.fusion_probabilities)}</section>${renderEmotionCurve(analysis)}${renderFeedback(feedback)}<section class="text-card"><h2 class="section-title">识别文本</h2><p class="asr-text">${escapeHtml(analysis.asr_text || '未识别到明确文本。')}</p><h2 class="section-title">关键词</h2><div class="chips">${renderKeywords(analysis.keywords)}</div><h2 class="section-title">可能原因线索</h2><div class="chips">${renderReasons(analysis.possible_reasons)}</div></section>`;
    }
    async function analyzeBlob(blob, filename) {
      const form = new FormData();
      form.append('file', blob, filename);
      setStatus('正在分析音频，首次加载模型可能需要一点时间。', 'busy');
      try {
        const response = await fetch('/api/analyze-audio', { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok || data.ok === false) throw new Error(data.message || data.error || '分析失败');
        renderResult(data);
        setStatus('分析完成。你可以继续试听音频，或换一段重新分析。');
      } catch (error) {
        output.innerHTML = `<div class="empty-state"><div><strong>分析没有完成</strong><span>${escapeHtml(error.message || error)}</span></div></div>`;
        setStatus(`分析失败：${error.message || error}`, 'error');
      }
    }
    fileInput.onchange = () => {
      const file = fileInput.files[0];
      if (!file) return;
      recordingBlob = null;
      analyzeRecordingButton.disabled = true;
      setPlayerBlob(file);
      setStatus('已选择音频，可以先播放试听，也可以直接分析。');
    };
    document.getElementById('submit').onclick = async () => {
      const file = fileInput.files[0];
      if (!file) {
        setStatus('请先选择一个音频文件。', 'error');
        return;
      }
      await analyzeBlob(file, file.name);
    };
    recordButton.onclick = async () => {
      if (!canRecordAudio()) {
        setStatus(explainRecordingUnavailable(), 'error');
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        chunks = [];
        const createdRecorder = createMediaRecorder(stream);
        const recorderFormat = createdRecorder.format;
        recorder = createdRecorder.recorder;
        recordingFilename = `recording.${recorderFormat.extension}`;
        recorder.ondataavailable = event => {
          if (event.data.size > 0) chunks.push(event.data);
        };
        recorder.onstop = () => {
          const actualMimeType = recorder.mimeType || recorderFormat.mimeType;
          recordingFilename = `recording.${extensionForMimeType(actualMimeType)}`;
          recordingBlob = actualMimeType ? new Blob(chunks, { type: actualMimeType }) : new Blob(chunks);
          setPlayerBlob(recordingBlob);
          analyzeRecordingButton.disabled = false;
          stream.getTracks().forEach(track => track.stop());
          setStatus('录音已停止，可以先试听或直接分析。');
        };
        recorder.start();
        setStatus('录音中，说完后点击停止录音。', 'busy');
        recordButton.disabled = true;
        stopButton.disabled = false;
        analyzeRecordingButton.disabled = true;
      } catch (error) {
        setStatus(`无法开始录音：${error.message || error}`, 'error');
      }
    };
    stopButton.onclick = () => {
      if (recorder && recorder.state !== 'inactive') recorder.stop();
      recordButton.disabled = false;
      stopButton.disabled = true;
    };
    analyzeRecordingButton.onclick = async () => {
      if (!recordingBlob) return;
      await analyzeBlob(recordingBlob, recordingFilename);
    };
    updateRecordingAvailability();
  </script>
  <script>
    function safeUploadFilename(filename, blob) {
      const fallback = `recording.${extensionForMimeType(blob.type)}`;
      const base = String(filename || fallback).split(/[\\/]/).pop() || fallback;
      return base.replace(/[^\\w.+-]/g, '_') || fallback;
    }
    async function analyzeBlob(blob, filename) {
      let stage = 'prepare';
      const safeFilename = safeUploadFilename(filename, blob);
      const apiUrl = `${window.location.origin}/api/analyze-audio`;
      setStatus('正在分析音频，首次加载模型可能需要一点时间。', 'busy');
      try {
        let response;
        try {
          stage = 'formdata';
          const form = new FormData();
          form.append('file', blob, safeFilename);
          stage = 'multipart_fetch';
          response = await fetch(apiUrl, { method: 'POST', body: form });
        } catch (formError) {
          stage = 'raw_fetch';
          response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
              'Content-Type': blob.type || 'application/octet-stream',
              'X-Filename': safeFilename
            },
            body: blob
          });
        }
        stage = 'read_response';
        const responseText = await response.text();
        let data;
        try {
          data = JSON.parse(responseText);
        } catch (jsonError) {
          throw new Error(`response_not_json: ${responseText.slice(0, 120)}`);
        }
        if (!response.ok || data.ok === false) throw new Error(data.message || data.error || 'analysis_failed');
        renderResult(data);
        setStatus('分析完成。你可以继续试听音频，或换一段重新分析。');
      } catch (error) {
        const detail = `stage=${stage}; type=${blob.type || 'unknown'}; size=${blob.size || 0}; filename=${safeFilename}; error=${error.message || error}`;
        output.innerHTML = `<div class="empty-state"><div><strong>分析没有完成</strong><span>${escapeHtml(detail)}</span></div></div>`;
        setStatus(`分析失败：${detail}`, 'error');
      }
    }
  </script>
</body>
</html>
"""


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>中文语音情绪陪伴智能体</title>
  <style>
    body { margin: 0; min-height: 100vh; display: grid; place-items: center; font-family: "Microsoft YaHei", "Noto Sans SC", system-ui, sans-serif; color: #26313c; background: linear-gradient(135deg, #f7fbff, #f3fbf6); }
    main { width: min(720px, calc(100% - 32px)); padding: 28px; border: 1px solid #dbe8ee; border-radius: 14px; background: rgba(255, 255, 255, .88); box-shadow: 0 20px 60px rgba(56, 75, 98, .14); }
    h1 { margin: 0 0 10px; font-size: clamp(28px, 5vw, 44px); line-height: 1.15; }
    p { margin: 0; color: #6f7e8a; line-height: 1.8; }
    code { padding: 2px 6px; border-radius: 6px; background: #eaf6ff; color: #1f6f8f; }
  </style>
</head>
<body>
  <main>
    <h1>中文语音情绪陪伴智能体</h1>
    <p>React 前端尚未构建。请在 <code>frontend</code> 目录运行 <code>npm install</code> 和 <code>npm run build</code>，然后重新启动 FastAPI 服务。</p>
  </main>
</body>
</html>
"""


if __name__ == "__main__":
    run_server()
