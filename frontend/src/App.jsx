import { useEffect, useRef, useState } from "react";
import { analyzeAudioBlob, fetchHealth } from "./api";
import AudioInputPanel from "./components/AudioInputPanel";
import ResultsPanel from "./components/ResultsPanel";
import {
  canRecordAudio,
  createMediaRecorder,
  explainRecordingError,
  explainRecordingUnavailable,
  extensionForMimeType
} from "./recording";

export default function App() {
  const fileInputRef = useRef(null);
  const playerRef = useRef(null);
  const selectedFileUrl = useRef(null);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);

  const [canRecord, setCanRecord] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingBlob, setRecordingBlob] = useState(null);
  const [recordingFilename, setRecordingFilename] = useState("recording.webm");
  const [status, setStatus] = useState("请选择音频文件，或点击开始录音。");
  const [statusType, setStatusType] = useState("info");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [healthText, setHealthText] = useState("后端状态检查中");

  useEffect(() => {
    const available = canRecordAudio();
    setCanRecord(available);
    if (!available) {
      setStatus(explainRecordingUnavailable());
      setStatusType("error");
    }
    fetchHealth()
      .then((health) => setHealthText(health.deepseek?.configured ? "DeepSeek 已配置" : "本地模板回退可用"))
      .catch(() => setHealthText("后端状态暂不可用"));
    return () => {
      if (selectedFileUrl.current) URL.revokeObjectURL(selectedFileUrl.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  function setPlayerBlob(blob) {
    if (selectedFileUrl.current) URL.revokeObjectURL(selectedFileUrl.current);
    selectedFileUrl.current = URL.createObjectURL(blob);
    if (playerRef.current) playerRef.current.src = selectedFileUrl.current;
  }

  function setInfo(message, type = "info") {
    setStatus(message);
    setStatusType(type);
  }

  function handleFileChange() {
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;
    setRecordingBlob(null);
    setPlayerBlob(file);
    setInfo("已选择音频，可以先播放试听，也可以直接分析。");
  }

  async function analyzeBlob(blob, filename) {
    setIsLoading(true);
    setError("");
    setInfo("正在分析音频，首次加载模型可能需要一点时间。", "busy");
    try {
      const data = await analyzeAudioBlob(blob, filename);
      setResult(data);
      setInfo("分析完成。你可以继续试听音频，或换一段重新分析。");
    } catch (analysisError) {
      console.error("音频分析异常", analysisError.technicalDetail || analysisError);
      setResult(null);
      setError(analysisError.message || String(analysisError));
      setInfo(`分析失败：${analysisError.message || analysisError}`, "error");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleAnalyzeFile() {
    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      setInfo("请先选择一个音频文件。", "error");
      return;
    }
    await analyzeBlob(file, file.name);
  }

  async function handleStartRecording() {
    if (!canRecordAudio()) {
      setCanRecord(false);
      setInfo(explainRecordingUnavailable(), "error");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      const createdRecorder = createMediaRecorder(stream);
      const recorder = createdRecorder.recorder;
      const recorderFormat = createdRecorder.format;
      recorderRef.current = recorder;
      setRecordingFilename(`recording.${recorderFormat.extension}`);
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const actualMimeType = recorder.mimeType || recorderFormat.mimeType;
        const filename = `recording.${extensionForMimeType(actualMimeType)}`;
        const blob = actualMimeType ? new Blob(chunksRef.current, { type: actualMimeType }) : new Blob(chunksRef.current);
        setRecordingFilename(filename);
        setRecordingBlob(blob);
        setPlayerBlob(blob);
        stream.getTracks().forEach((track) => track.stop());
        setIsRecording(false);
        setInfo("录音已停止，可以先试听或直接分析。");
      };
      recorder.start();
      setIsRecording(true);
      setRecordingBlob(null);
      setInfo("录音中... 说完后点击停止录音。", "busy");
    } catch (recordError) {
      setIsRecording(false);
      setInfo(explainRecordingError(recordError), "error");
    }
  }

  function handleStopRecording() {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
  }

  async function handleAnalyzeRecording() {
    if (!recordingBlob) return;
    await analyzeBlob(recordingBlob, recordingFilename);
  }

  return (
    <main>
      <header>
        <p className="eyebrow">{healthText}</p>
        <h1>中文语音情绪陪伴智能体</h1>
        <p className="subtitle">上传或录制一段中文语音，查看情绪识别、窗口变化、关键词线索和一段温和的陪伴式反馈。</p>
      </header>
      <div className="layout">
        <AudioInputPanel
          fileInputRef={fileInputRef}
          playerRef={playerRef}
          status={status}
          statusType={statusType}
          canRecord={canRecord}
          isRecording={isRecording}
          hasRecording={Boolean(recordingBlob)}
          onFileChange={handleFileChange}
          onAnalyzeFile={handleAnalyzeFile}
          onStartRecording={handleStartRecording}
          onStopRecording={handleStopRecording}
          onAnalyzeRecording={handleAnalyzeRecording}
        />
        <ResultsPanel result={result} isLoading={isLoading} error={error} />
      </div>
    </main>
  );
}
