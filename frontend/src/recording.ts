export const ACCEPTED_AUDIO_EXTENSIONS = ".wav,.mp3,.m4a,.flac,.webm";

export function canRecordAudio() {
  return Boolean(
    navigator.mediaDevices
    && typeof navigator.mediaDevices.getUserMedia === "function"
    && typeof window.MediaRecorder === "function"
  );
}

export function explainRecordingUnavailable() {
  const host = window.location.hostname;
  const isLocal = host === "localhost" || host === "127.0.0.1" || host === "::1";
  if (window.location.protocol !== "https:" && !isLocal) {
    return "当前是局域网 HTTP 访问，浏览器不会开放麦克风。请上传音频，或改用 HTTPS。";
  }
  return "当前浏览器没有开放麦克风录音接口，请上传音频或更换 Chrome / Edge。";
}

export function explainRecordingError(error: unknown) {
  const candidate = error as { name?: string; message?: string };
  const name = String(candidate?.name || "");
  const message = String(candidate?.message || error || "");
  if (name === "NotFoundError" || /requested device not found/i.test(message)) return "未连接麦克风";
  if (name === "NotAllowedError" || name === "PermissionDeniedError") return "未开启麦克风权限，请在浏览器设置中允许访问麦克风";
  if (name === "NotReadableError" || name === "TrackStartError") return "无法使用麦克风，请检查是否被其他程序占用";
  if (name === "NotSupportedError") return "当前浏览器不支持可用的录音格式，请更换 Chrome 或 Edge";
  if (name === "SecurityError") return "当前页面无权使用麦克风，请通过 HTTPS 或本机地址访问";
  if (name === "AbortError") return "麦克风启动失败，请重新连接设备后重试";
  return "无法开始录音，请检查麦克风后重试";
}

export function extensionForMimeType(mimeType?: string) {
  const clean = String(mimeType || "").split(";")[0].trim().toLowerCase();
  if (["audio/mp4", "audio/aac", "audio/x-m4a"].includes(clean)) return "m4a";
  if (["audio/mpeg", "audio/mp3"].includes(clean)) return "mp3";
  if (["audio/wav", "audio/wave"].includes(clean)) return "wav";
  if (clean === "audio/flac") return "flac";
  return "webm";
}

interface RecorderFormat { options: MediaRecorderOptions; mimeType: string; extension: string }

export function createMediaRecorder(stream: MediaStream): { recorder: MediaRecorder; format: RecorderFormat } {
  const mimeTypes = ["audio/mp4", "audio/aac", "audio/webm;codecs=opus", "audio/webm", ""];
  let lastError: unknown;
  for (const mimeType of mimeTypes) {
    if (mimeType && typeof MediaRecorder.isTypeSupported === "function" && !MediaRecorder.isTypeSupported(mimeType)) continue;
    try {
      const options = mimeType ? { mimeType } : {};
      return { recorder: new MediaRecorder(stream, options), format: { options, mimeType, extension: extensionForMimeType(mimeType) } };
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("当前浏览器无法创建录音器");
}

export function safeUploadFilename(filename: string | undefined, blob: Blob) {
  const fallback = `recording.${extensionForMimeType(blob.type)}`;
  const base = String(filename || fallback).split(/[\\/]/).pop() || fallback;
  return base.replace(/[^\w.+-]/g, "_") || fallback;
}
