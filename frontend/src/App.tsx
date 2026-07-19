import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import {
  Activity,
  AudioLines,
  CheckCircle2,
  ChevronRight,
  Clock3,
  FileAudio,
  HeartHandshake,
  LoaderCircle,
  Mic,
  MicOff,
  Music2,
  NotebookPen,
  Plus,
  Send,
  Sparkles,
  Upload,
  UserRound,
  Waves,
  Wind
} from "lucide-react";
import { analyzeAudioBlob, chatWithAgent, streamAgentChat } from "./api";
import { emotionInfo, formatPercent } from "./emotions";
import {
  ACCEPTED_AUDIO_EXTENSIONS,
  canRecordAudio,
  createMediaRecorder,
  explainRecordingError,
  explainRecordingUnavailable,
  extensionForMimeType
} from "./recording";
import type { AgentToolCall, AnalysisResponse, AudioAnalysis, EmotionCurvePoint, EvidenceItem } from "./types";

type UiState = "idle" | "recording" | "analyzing" | "ready" | "error";

interface ChatMessage {
  id: number;
  role: "user" | "ai";
  text: string;
  time: string;
  emotion?: string;
  audioUrl?: string;
  searchStatus?: "searching" | "success" | "empty";
  sourceCount?: number;
}

const waveHeights = [8, 18, 28, 38, 42, 48, 44, 36, 26, 16, 10, 20, 32, 44, 40, 30, 18, 12, 22, 34, 46, 42, 32, 20, 14];

const sessionShells = [
  ["当前对话", "刚刚"],
  ["关于工作压力的倾诉", "昨天 22:15"],
  ["夜晚的孤独感", "周三 19:45"],
  ["和家人的一次交流", "上周 11:20"],
  ["对未来的想法", "上周 16:08"]
];

const quickEmotions = [
  ["平静", "#8ba8dc"], ["开心", "#ffc670"], ["悲伤", "#788cc8"],
  ["焦虑", "#ffa064"], ["愤怒", "#e65a5a"], ["感动", "#c88cdc"], ["孤独", "#6482b4"]
];

function nowTime() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

function evidenceText(item: EvidenceItem | string) {
  if (typeof item === "string") return item;
  return item.keyword || item.text || item.reason || item.value || "分析线索";
}

function inlineMessageParts(text: string): ReactNode[] {
  return text.split(/(\*\*[^*\n]+\*\*|https?:\/\/[^\s，。！？；：、）】}>]+)/g).filter(Boolean).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={index}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("http")) return (
      <a className="message-link" href={part} key={`${part}-${index}`} target="_blank" rel="noreferrer">{part}</a>
    );
    return <span key={index}>{part}</span>;
  });
}

type MessageBlock = {
  kind: "paragraph" | "heading" | "bullet";
  text: string;
  marker?: string;
};

function messageBlocks(text: string): MessageBlock[] {
  const blocks: MessageBlock[] = [];
  let current: MessageBlock | null = null;
  const flush = () => {
    if (current?.text) blocks.push(current);
    current = null;
  };

  for (const rawLine of text.replace(/\r/g, "").split("\n")) {
    const line = rawLine.trim();
    if (!line) {
      flush();
      continue;
    }
    const heading = line.match(/^#{1,6}\s+(.+)$/);
    const bullet = line.match(/^([-*•])\s+(.+)$/);
    const ordered = line.match(/^(\d+[.)])\s+(.+)$/);
    if (heading) {
      flush();
      blocks.push({ kind: "heading", text: heading[1] });
    } else if (bullet || ordered) {
      flush();
      current = {
        kind: "bullet",
        marker: ordered?.[1] || "•",
        text: bullet?.[2] || ordered?.[2] || ""
      };
    } else if (current) {
      // Markdown single newlines are soft wraps. Joining them prevents bold
      // labels, Chinese punctuation and following text becoming separate rows.
      const needsSpace = /[A-Za-z0-9]$/.test(current.text) && /^[A-Za-z0-9]/.test(line);
      current.text += `${needsSpace ? " " : ""}${line}`;
    } else {
      current = { kind: "paragraph", text: line };
    }
  }
  flush();
  return blocks;
}

function MessageText({ text }: { text: string }) {
  const blocks = messageBlocks(text);
  return (
    <div className="message-text">
      {blocks.map((block, index) => {
        const className = `message-line ${block.kind}`;
        return (
          <div className={className} key={index}>
            {block.kind === "bullet" ? (
              <><span className="bullet-mark">{block.marker}</span><span className="bullet-content">{inlineMessageParts(block.text)}</span></>
            ) : inlineMessageParts(block.text)}
          </div>
        );
      })}
    </div>
  );
}

function pointValue(point: EmotionCurvePoint, emotion: string) {
  if (typeof point.confidence === "number") return point.confidence;
  if (typeof point.probability === "number") return point.probability;
  const value = point[emotion];
  return typeof value === "number" ? value : 0;
}

function VoiceOrb({ state, emotion }: { state: UiState; emotion?: string }) {
  const info = emotionInfo(emotion);
  const active = state === "recording" || state === "analyzing";
  return (
    <div className="voice-orb-wrap" aria-hidden="true">
      <div className={`orb-halo ${active ? "is-active" : ""}`} style={{ "--orb-color": info.color } as React.CSSProperties} />
      <div className={`orb-ring orb-ring-one ${active ? "is-active" : ""}`} />
      <div className={`orb-ring orb-ring-two ${active ? "is-active" : ""}`} />
      <div className="voice-orb" style={{ "--orb-color": info.color, "--orb-glow": info.glow } as React.CSSProperties}>
        {state === "analyzing" ? <LoaderCircle className="spin" size={30} /> : state === "recording" ? <AudioLines size={31} /> : <HeartHandshake size={30} />}
      </div>
    </div>
  );
}

function Waveform({ active }: { active: boolean }) {
  return (
    <div className="waveform" aria-hidden="true">
      {waveHeights.map((height, index) => (
        <span key={index} className={active ? "active" : ""} style={{ "--wave-height": `${height / 4}`, "--wave-delay": `${index * 38}ms` } as React.CSSProperties} />
      ))}
    </div>
  );
}

function EmotionCurve({ analysis }: { analysis: AudioAnalysis }) {
  const curve = Array.isArray(analysis.emotion_curve) ? analysis.emotion_curve : [];
  const emotion = analysis.fusion_prediction || "unknown";
  const info = emotionInfo(emotion);
  if (curve.length < 2) return <div className="curve-empty"><Activity size={18} />暂无足够的单次语音窗口数据</div>;
  const width = 560;
  const height = 150;
  const pad = 14;
  const points = curve.map((point, index) => ({
    x: pad + (index / Math.max(1, curve.length - 1)) * (width - pad * 2),
    y: height - pad - Math.max(0, Math.min(1, pointValue(point, emotion))) * (height - pad * 2)
  }));
  const line = points.map((point, index) => `${index ? "L" : "M"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" ");
  const area = `${line} L ${points.at(-1)!.x.toFixed(1)} ${height - pad} L ${points[0].x.toFixed(1)} ${height - pad} Z`;
  return (
    <div className="curve-chart">
      <svg viewBox={`0 0 ${width} ${height}`} aria-label="单次语音窗口情绪曲线">
        <defs><linearGradient id="curve-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={info.color} stopOpacity=".34" /><stop offset="100%" stopColor={info.color} stopOpacity="0" /></linearGradient></defs>
        <path d={area} fill="url(#curve-fill)" /><path d={line} fill="none" stroke={info.color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div className="curve-axis"><span>开始</span><span>单次语音窗口变化</span><span>结束</span></div>
    </div>
  );
}

function Avatar({ role }: { role: "user" | "ai" }) {
  return (
    <span className={`chat-avatar ${role}`}>
      {role === "ai" ? <HeartHandshake size={15} /> : <UserRound size={15} />}
    </span>
  );
}

export default function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioUrlsRef = useRef<Set<string>>(new Set());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [uiState, setUiState] = useState<UiState>("idle");
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState("");
  const [inputText, setInputText] = useState("");
  const [isChatting, setIsChatting] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [manualEmotion, setManualEmotion] = useState("平静");
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: 1, role: "ai", text: "你好，我是心伴。今天感觉怎么样？无论你想聊什么，我都在这里陪着你。", time: nowTime(), emotion: "平静" }
  ]);

  const analysis = result?.analysis || {};
  const fusionInfo = emotionInfo(analysis.fusion_prediction);
  const currentEmotion = analysis.fusion_prediction ? fusionInfo.label : manualEmotion;
  const currentColor = analysis.fusion_prediction ? fusionInfo.color : quickEmotions.find(([label]) => label === manualEmotion)?.[1] || "#8ba8dc";
  const keywords = analysis.keywords || [];
  const reasons = analysis.possible_reasons || [];

  const confidenceRows = useMemo(() => [
    ["声学置信度", analysis.audio_confidence || 0],
    ["文本置信度", analysis.text_confidence || 0],
    ["融合置信度", analysis.fusion_confidence || 0]
  ] as const, [analysis.audio_confidence, analysis.text_confidence, analysis.fusion_confidence]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    audioUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    audioUrlsRef.current.clear();
  }, []);

  function setPreview(blob: Blob) {
    const url = URL.createObjectURL(blob);
    audioUrlsRef.current.add(url);
    return url;
  }

  function addAnalysisMessages(data: AnalysisResponse, audioMessageId: number) {
    const transcript = data.analysis?.asr_text || "我上传了一段语音。";
    const response = data.agent_feedback?.companion_response || "我已经听见了，也整理好了这段声音里的情绪线索。";
    const emotion = emotionInfo(data.analysis?.fusion_prediction).label;
    setMessages((previous) => [
      ...previous.map((message) => message.id === audioMessageId ? { ...message, text: transcript, emotion } : message),
      { id: Date.now(), role: "ai", text: response, time: nowTime(), emotion }
    ]);
  }

  async function runAnalysis(blob: Blob, filename: string, previewUrl: string) {
    const audioMessageId = Date.now();
    setMessages((previous) => [
      ...previous,
      { id: audioMessageId, role: "user", text: "正在分析这段语音…", time: nowTime(), emotion: "分析中", audioUrl: previewUrl }
    ]);
    setUiState("analyzing");
    setError("");
    try {
      const data = await analyzeAudioBlob(blob, filename);
      setResult(data);
      setUiState("ready");
      addAnalysisMessages(data, audioMessageId);
    } catch (analysisError) {
      const typedError = analysisError as Error & { technicalDetail?: string };
      console.error("音频分析异常", typedError.technicalDetail || typedError);
      setError(typedError.message || "音频分析失败，请稍后重试");
      setUiState("error");
      setMessages((previous) => previous.map((message) => message.id === audioMessageId ? { ...message, text: "这段语音暂时没有完成分析。", emotion: undefined } : message));
    }
  }

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const previewUrl = setPreview(file);
    await runAnalysis(file, file.name, previewUrl);
  }

  async function toggleRecording() {
    if (uiState === "recording") {
      recorderRef.current?.stop();
      return;
    }
    if (!canRecordAudio()) {
      const message = explainRecordingUnavailable();
      setError(message);
      setUiState("error");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      const { recorder, format } = createMediaRecorder(stream);
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => { if (event.data.size) chunksRef.current.push(event.data); };
      recorder.onstop = () => {
        const mimeType = recorder.mimeType || format.mimeType;
        const blob = new Blob(chunksRef.current, mimeType ? { type: mimeType } : undefined);
        const filename = `recording.${extensionForMimeType(mimeType)}`;
        stream.getTracks().forEach((track) => track.stop());
        const previewUrl = setPreview(blob);
        void runAnalysis(blob, filename, previewUrl);
      };
      recorder.start();
      setError("");
      setUiState("recording");
    } catch (recordError) {
      setError(explainRecordingError(recordError));
      setUiState("error");
    }
  }

  async function sendTextMessage() {
    const text = inputText.trim();
    if (!text || isChatting) return;
    const userId = Date.now();
    const assistantId = userId + 1;
    const history = messages.map((message) => ({
      role: message.role === "ai" ? "assistant" as const : "user" as const,
      content: message.text
    }));
    setMessages((previous) => [
      ...previous,
      { id: userId, role: "user", text, time: nowTime(), emotion: manualEmotion },
      { id: assistantId, role: "ai", text: "正在整理你的感受…", time: nowTime(), emotion: "平静" }
    ]);
    setInputText("");
    setIsChatting(true);
    setError("");
    const payload = { message: text, session_id: sessionId || undefined, history, latest_analysis: analysis };
    let streamedText = "";
    const updateAssistant = (nextText: string) => {
      setMessages((previous) => previous.map((message) => message.id === assistantId ? { ...message, text: nextText } : message));
    };
    const updateSearchStatus = (searchStatus: ChatMessage["searchStatus"], sourceCount = 0) => {
      setMessages((previous) => previous.map((message) => message.id === assistantId ? { ...message, searchStatus, sourceCount } : message));
    };
    const applySearchCalls = (toolCalls: AgentToolCall[]) => {
      const searchCall = [...toolCalls].reverse().find((call) => call.name === "web_search");
      if (!searchCall) return;
      const results = Array.isArray(searchCall.result?.results) ? searchCall.result.results : [];
      updateSearchStatus(searchCall.result?.available ? "success" : "empty", results.length);
    };
    try {
      const response = await streamAgentChat(payload, (chunk) => {
        streamedText += chunk;
        updateAssistant(streamedText);
      }, (event, toolEvent) => {
        if (toolEvent.name !== "web_search") return;
        if (event === "tool_start") updateSearchStatus("searching");
        if (event === "tool_result") {
          const results = Array.isArray(toolEvent.result?.results) ? toolEvent.result.results : [];
          updateSearchStatus(toolEvent.result?.available ? "success" : "empty", results.length);
        }
      });
      setSessionId(response.session_id);
      applySearchCalls(response.tool_calls || []);
      if (response.search_required && response.fallback_reason) updateSearchStatus("empty");
      if (!streamedText) updateAssistant(response.response);
    } catch (streamError) {
      console.warn("流式文字陪伴不可用，切换普通 JSON", streamError);
      try {
        const response = await chatWithAgent(payload);
        setSessionId(response.session_id);
        applySearchCalls(response.tool_calls || []);
        if (response.search_required && response.fallback_reason) updateSearchStatus("empty");
        updateAssistant(response.response);
      } catch (chatError) {
        console.error("文字陪伴服务异常", chatError);
        updateAssistant("文字陪伴服务暂时不可用，请确认后端已经启动后重试。你仍然可以继续录音或上传音频。");
      }
    } finally {
      setIsChatting(false);
    }
  }

  const stateLabel = uiState === "recording" ? "正在聆听…" : uiState === "analyzing" ? "正在理解…" : uiState === "ready" ? "已整理好这段感受" : "等待聆听";

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" /><div className="ambient ambient-two" />
      <header className="topbar">
        <div className="brand-mark"><HeartHandshake size={20} /></div>
        <div className="brand-copy"><strong>心伴</strong><span>情绪陪伴智能体</span></div>
        <div className="header-actions">
          <div className="header-emotion" style={{ "--emotion": currentColor } as React.CSSProperties}><span />当前情绪：{currentEmotion}</div>
          <button className="profile-button" aria-label="本地用户"><UserRound size={17} /></button>
        </div>
      </header>

      <div className="app-grid">
        <aside className="left-sidebar">
          <div className="side-title">会话历史</div>
          <div className="session-list">
            {sessionShells.map(([title, time], index) => (
              <button key={title} className={`session-item ${index === 0 ? "active" : ""}`}>
                <span><strong>{title}</strong><small><Clock3 size={10} />{time}</small></span>
              </button>
            ))}
          </div>
          <button className="new-session"><Plus size={14} />新建会话</button>
        </aside>

        <main className="main-stage">
          <section className="voice-stage">
            <VoiceOrb state={uiState} emotion={analysis.fusion_prediction} />
            <Waveform active={uiState === "recording" || uiState === "analyzing"} />
            <div className="state-copy"><strong>{stateLabel}</strong><span>{uiState === "recording" ? "再次点击麦克风结束录音" : "用文字、录音或已有音频，说说此刻的感受"}</span></div>
          </section>

          <Tabs.Root defaultValue="chat" className="tabs-root">
            <Tabs.List className="figma-tabs" aria-label="内容视图">
              <Tabs.Trigger value="chat"><HeartHandshake size={14} />对话</Tabs.Trigger>
              <Tabs.Trigger value="mood"><Activity size={14} />情绪</Tabs.Trigger>
            </Tabs.List>

            <Tabs.Content value="chat" className="chat-tab">
              <div className="messages-list">
                {messages.map((message) => (
                  <article key={message.id} className={`message-row ${message.role}`}>
                    <Avatar role={message.role} />
                    <div className="message-stack">
                      <div className="message-bubble">
                        <MessageText text={message.text} />
                        {message.role === "ai" && message.searchStatus && (
                          <div className={`search-evidence ${message.searchStatus}`}>
                            {message.searchStatus === "searching" && <><LoaderCircle size={13} />正在调用智谱夸克搜索</>}
                            {message.searchStatus === "success" && <><CheckCircle2 size={13} />已联网检索 · {message.sourceCount || 0} 条可用来源</>}
                            {message.searchStatus === "empty" && <><MicOff size={13} />联网搜索未获得可用来源</>}
                          </div>
                        )}
                        {message.audioUrl && <audio className="message-audio" controls preload="metadata" src={message.audioUrl} />}
                      </div>
                      <div className="message-meta"><span>{message.time}</span>{message.emotion && <span style={{ color: currentColor }}>{message.emotion}</span>}</div>
                    </div>
                  </article>
                ))}
                {uiState === "analyzing" && <article className="message-row ai"><Avatar role="ai" /><div className="typing-bubble"><span /><span /><span /></div></article>}
                <div ref={messagesEndRef} />
              </div>

              {error && <div className="error-banner" role="alert"><MicOff size={16} /><span>{error}</span></div>}
              <div className="figma-composer">
                <input ref={fileInputRef} className="sr-only" type="file" accept={ACCEPTED_AUDIO_EXTENSIONS} onChange={handleFileChange} />
                <button className={`round-action mic-action ${uiState === "recording" ? "recording" : ""}`} onClick={toggleRecording} disabled={uiState === "analyzing"} aria-label={uiState === "recording" ? "停止录音并分析" : "开始麦克风录音"}>
                  {uiState === "recording" ? <MicOff size={19} /> : uiState === "analyzing" ? <LoaderCircle className="spin" size={19} /> : <Mic size={19} />}
                </button>
                <div className="text-input-shell">
                  <button className="upload-inline" onClick={() => fileInputRef.current?.click()} disabled={uiState === "recording" || uiState === "analyzing"} aria-label="上传已有录音或音频文件"><Upload size={17} /></button>
                  <input value={inputText} onChange={(event) => setInputText(event.target.value)} onKeyDown={(event) => event.key === "Enter" && void sendTextMessage()} disabled={isChatting} placeholder={isChatting ? "正在回复…" : "用文字倾诉，或按麦克风说话…"} />
                  <span className="format-hint">WAV · MP3 · M4A · FLAC · WEBM</span>
                </div>
                <button className="round-action send-action" onClick={() => void sendTextMessage()} disabled={!inputText.trim() || isChatting} aria-label="发送文字消息">{isChatting ? <LoaderCircle className="spin" size={17} /> : <Send size={17} />}</button>
              </div>
            </Tabs.Content>

            <Tabs.Content value="mood" className="mood-tab">
              <section className="mood-overview">
                <div className="mood-ring" style={{ "--emotion": currentColor } as React.CSSProperties}><Activity size={22} /></div>
                <div><small>本次情绪概览</small><strong>{currentEmotion}</strong><p>{analysis.fusion_prediction ? `融合置信度 ${formatPercent(analysis.fusion_confidence)}` : "完成一次语音分析后显示真实数据"}</p></div>
              </section>
              <section className="mood-card">
                <div className="panel-heading"><Activity size={15} />单次语音窗口曲线</div>
                <EmotionCurve analysis={analysis} />
              </section>
              <div className="mood-columns">
                <section className="mood-card"><div className="panel-heading"><Sparkles size={15} />关键词线索</div><div className="chip-list">{keywords.length ? keywords.map((item, index) => <span key={index}>{evidenceText(item)}</span>) : <span>暂无真实数据</span>}</div></section>
                <section className="mood-card"><div className="panel-heading"><ChevronRight size={15} />可能原因</div><div className="reason-list">{reasons.length ? reasons.slice(0, 3).map((item, index) => <p key={index}>{evidenceText(item)}</p>) : <p>完成分析后显示原因线索</p>}</div></section>
              </div>
            </Tabs.Content>
          </Tabs.Root>
        </main>

        <aside className="right-sidebar">
          <div className="side-title">情感分析</div>
          <section className="current-emotion" style={{ "--emotion": currentColor, "--emotion-glow": `${currentColor}33` } as React.CSSProperties}>
            <div><span>当前识别情绪</span><strong>{currentEmotion}</strong></div>
            {confidenceRows.map(([label, value]) => <div className="metric-row" key={label}><span>{label}</span><small>{value ? formatPercent(value) : "—"}</small><div><i style={{ width: `${value * 100}%` }} /></div></div>)}
          </section>

          <section className="quick-emotions">
            <span className="panel-label">快速标记情绪</span>
            <div>{quickEmotions.map(([label, color]) => <button key={label} className={manualEmotion === label && !analysis.fusion_prediction ? "active" : ""} style={{ "--emotion": color } as React.CSSProperties} onClick={() => setManualEmotion(label)}>{label}</button>)}</div>
          </section>

          <section className="affirmation-card"><span className="panel-label">今日心语</span><p>“你不必假装一切都好。<br />允许自己感受，<br />是勇气的一种形式。”</p></section>

          <section className="comfort-tools">
            <span className="panel-label">舒缓工具</span>
            <button><Wind size={18} /><span><strong>呼吸练习</strong><small>4-7-8 放松呼吸法</small></span></button>
            <button><Music2 size={18} /><span><strong>白噪音</strong><small>雨声 · 海浪 · 森林</small></span></button>
            <button><NotebookPen size={18} /><span><strong>情绪日记</strong><small>记录你的感受</small></span></button>
          </section>

          <section className="analysis-source"><FileAudio size={15} /><span>{result ? <><strong>真实语音分析已接入</strong><small>声学 · ASR · 多模态融合</small></> : <><strong>等待语音输入</strong><small>录音或上传音频后显示结果</small></>}</span>{result && <CheckCircle2 size={15} />}</section>
        </aside>
      </div>
    </div>
  );
}
