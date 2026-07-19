import { safeUploadFilename } from "./recording";
import type { AgentChatRequest, AgentChatResponse, AgentStreamToolEvent, AnalysisResponse, HealthResponse } from "./types";

function analysisErrorMessage(error: unknown, stage: string, responseStatus: number) {
  const message = String((error as Error)?.message || error || "");
  if (/failed to fetch|networkerror|load failed/i.test(message) || stage.endsWith("_fetch")) return "无法连接分析服务，请确认后端已经启动";
  if (/timeout|timed out/i.test(message) || responseStatus === 408 || responseStatus === 504) return "音频分析超时，请稍后重试";
  if (responseStatus === 413) return "音频文件过大，请缩短录音或选择较小的文件";
  if (/response_not_json/i.test(message)) return "分析服务返回异常，请重启后端后重试";
  if (/no such file|label_map|pipeline|model|joblib/i.test(message)) return "分析服务配置不完整，请联系管理员检查模型文件";
  if (/ffmpeg|decode|unsupported|format|audio_bytes must not be empty/i.test(message)) return "无法读取这个音频文件，请更换音频格式后重试";
  return "音频分析失败，请稍后重试";
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch("/api/health");
  if (!response.ok) throw new Error("健康检查失败");
  return response.json() as Promise<HealthResponse>;
}

export async function analyzeAudioBlob(blob: Blob, filename?: string): Promise<AnalysisResponse> {
  let stage = "prepare";
  let responseStatus = 0;
  const safeFilename = safeUploadFilename(filename, blob);
  try {
    let response: Response;
    try {
      stage = "formdata";
      const form = new FormData();
      form.append("file", blob, safeFilename);
      stage = "multipart_fetch";
      response = await fetch("/api/analyze-audio", { method: "POST", body: form });
    } catch (formError) {
      console.warn("multipart 上传失败，切换 raw body", formError);
      stage = "raw_fetch";
      response = await fetch("/api/analyze-audio", {
        method: "POST",
        headers: { "Content-Type": blob.type || "application/octet-stream", "X-Filename": safeFilename },
        body: blob
      });
    }
    responseStatus = response.status;
    stage = "read_response";
    const responseText = await response.text();
    let data: AnalysisResponse & { message?: string; error?: string };
    try {
      data = JSON.parse(responseText) as typeof data;
    } catch {
      throw new Error(`response_not_json: ${responseText.slice(0, 120)}`);
    }
    if (!response.ok || data.ok === false) throw new Error(data.message || data.error || "analysis_failed");
    return data;
  } catch (error) {
    const detail = `stage=${stage}; type=${blob.type || "unknown"}; size=${blob.size}; filename=${safeFilename}; error=${(error as Error)?.message || error}`;
    console.error("音频分析失败", detail);
    const userError = new Error(analysisErrorMessage(error, stage, responseStatus)) as Error & { technicalDetail?: string };
    userError.technicalDetail = detail;
    throw userError;
  }
}

export async function chatWithAgent(payload: AgentChatRequest): Promise<AgentChatResponse> {
  const response = await fetch("/api/agent/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await response.json() as AgentChatResponse & { message?: string };
  if (!response.ok || !data.ok) throw new Error(data.message || "文字陪伴服务暂时不可用");
  return data;
}

export async function streamAgentChat(
  payload: AgentChatRequest,
  onDelta: (text: string) => void,
  onToolEvent?: (event: "tool_start" | "tool_result", payload: AgentStreamToolEvent) => void
): Promise<AgentChatResponse> {
  const response = await fetch("/api/agent/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
    body: JSON.stringify(payload)
  });
  if (!response.ok || !response.body) throw new Error("文字陪伴流式服务暂时不可用");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult: AgentChatResponse | null = null;
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      const parsed = JSON.parse(data) as AgentChatResponse & { text?: string };
      if (event === "delta" && parsed.text) onDelta(parsed.text);
      if ((event === "tool_start" || event === "tool_result") && onToolEvent) {
        onToolEvent(event, parsed as unknown as AgentStreamToolEvent);
      }
      if (event === "final") finalResult = parsed;
    }
    if (done) break;
  }
  if (!finalResult) throw new Error("文字陪伴流式响应不完整");
  return finalResult;
}
