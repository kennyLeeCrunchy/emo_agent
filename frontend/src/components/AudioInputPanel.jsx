import { useEffect } from "react";

export default function AudioInputPanel({
  fileInputRef,
  playerRef,
  status,
  statusType,
  canRecord,
  isRecording,
  hasRecording,
  onFileChange,
  onAnalyzeFile,
  onStartRecording,
  onStopRecording,
  onAnalyzeRecording
}) {
  useEffect(() => {
    if (!canRecord && statusType !== "error") {
      return;
    }
  }, [canRecord, statusType]);

  return (
    <section className="panel input-panel" aria-label="音频输入">
      <div className="input-block">
        <h2 className="block-title">上传音频</h2>
        <p className="hint">支持 wav、mp3、m4a、flac、webm 等常见音频格式。</p>
        <input ref={fileInputRef} onChange={onFileChange} type="file" accept="audio/*" />
        <button className="primary" type="button" onClick={onAnalyzeFile}>
          <span className="button-icon">UP</span>
          分析上传音频
        </button>
      </div>
      <div className="input-block">
        <h2 className="block-title">浏览器录音</h2>
        <div className="toolbar">
          <button className="soft" type="button" onClick={onStartRecording} disabled={!canRecord || isRecording}>
            <span className="button-icon">REC</span>
            开始录音
          </button>
          <button type="button" onClick={onStopRecording} disabled={!isRecording}>
            <span className="button-icon">STOP</span>
            停止录音
          </button>
          <button className="primary" type="button" onClick={onAnalyzeRecording} disabled={!hasRecording || isRecording}>
            <span className="button-icon">GO</span>
            分析录音
          </button>
        </div>
        <audio ref={playerRef} controls />
        <p className="hint">可以先试听，再提交分析。首次分析会加载模型，等待时间会稍长。</p>
      </div>
      <div className={`status ${statusType === "busy" ? "busy" : statusType === "error" ? "error" : ""}`}>{status}</div>
    </section>
  );
}
