export function canRecordAudio() {
  return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
}

export function explainRecordingUnavailable() {
  const host = window.location.hostname;
  const isLocal = host === "localhost" || host === "127.0.0.1" || host === "::1";
  if (window.location.protocol !== "https:" && !isLocal) {
    return "当前是局域网 HTTP 访问，浏览器不会开放麦克风录音。可以先用上传音频；如果要在局域网录音，需要改成 HTTPS 访问。";
  }
  return "当前浏览器没有开放麦克风录音接口。可以先用上传音频，或换 Chrome / Edge 再试。";
}

export function explainRecordingError(error) {
  const name = String(error?.name || "");
  const message = String(error?.message || error || "");
  if (name === "NotFoundError" || /requested device not found/i.test(message)) {
    return "未连接麦克风";
  }
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "未开启麦克风权限，请在浏览器设置中允许访问麦克风";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "无法使用麦克风，请检查是否被其他程序占用";
  }
  if (name === "NotSupportedError") {
    return "当前浏览器不支持可用的录音格式，请更换 Chrome 或 Edge 后重试";
  }
  if (name === "SecurityError") {
    return "当前页面无权使用麦克风，请通过 HTTPS 或本机地址访问";
  }
  if (name === "AbortError") {
    return "麦克风启动失败，请重新连接设备后重试";
  }
  return "无法开始录音，请检查麦克风后重试";
}

export function extensionForMimeType(mimeType) {
  const clean = String(mimeType || "").split(";")[0].trim().toLowerCase();
  if (clean === "audio/mp4" || clean === "audio/aac" || clean === "audio/x-m4a") return "m4a";
  if (clean === "audio/mpeg" || clean === "audio/mp3") return "mp3";
  if (clean === "audio/wav" || clean === "audio/wave") return "wav";
  if (clean === "audio/ogg") return "ogg";
  return "webm";
}

export function selectRecorderFormat() {
  const candidates = ["audio/mp4", "audio/aac", "audio/webm;codecs=opus", "audio/webm"];
  if (!window.MediaRecorder || typeof MediaRecorder.isTypeSupported !== "function") {
    return { options: {}, mimeType: "", extension: "webm" };
  }
  for (const mimeType of candidates) {
    if (MediaRecorder.isTypeSupported(mimeType)) {
      return { options: { mimeType }, mimeType, extension: extensionForMimeType(mimeType) };
    }
  }
  return { options: {}, mimeType: "", extension: "webm" };
}

export function createMediaRecorder(stream) {
  const formats = [
    selectRecorderFormat(),
    { options: { mimeType: "audio/mp4" }, mimeType: "audio/mp4", extension: "m4a" },
    { options: { mimeType: "audio/aac" }, mimeType: "audio/aac", extension: "m4a" },
    { options: { mimeType: "audio/webm;codecs=opus" }, mimeType: "audio/webm;codecs=opus", extension: "webm" },
    { options: { mimeType: "audio/webm" }, mimeType: "audio/webm", extension: "webm" },
    { options: {}, mimeType: "", extension: "webm" }
  ];
  const seen = new Set();
  let lastError = null;
  for (const format of formats) {
    const key = format.mimeType || "default";
    if (seen.has(key)) continue;
    seen.add(key);
    try {
      return { recorder: new MediaRecorder(stream, format.options), format };
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("当前浏览器无法创建录音器");
}

export function safeUploadFilename(filename, blob) {
  const fallback = `recording.${extensionForMimeType(blob.type)}`;
  const base = String(filename || fallback).split(/[\\/]/).pop() || fallback;
  return base.replace(/[^\w.+-]/g, "_") || fallback;
}
