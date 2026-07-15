import { safeUploadFilename } from "./recording";

function analysisErrorMessage(error, stage, responseStatus) {
  const message = String(error?.message || error || "");
  if (/failed to fetch|networkerror|load failed/i.test(message) || stage.endsWith("_fetch")) {
    return "无法连接分析服务，请确认后端已经启动";
  }
  if (/timeout|timed out/i.test(message) || responseStatus === 408 || responseStatus === 504) {
    return "音频分析超时，请稍后重试";
  }
  if (responseStatus === 413) {
    return "音频文件过大，请缩短录音或选择较小的文件";
  }
  if (/response_not_json/i.test(message)) {
    return "分析服务返回异常，请重启后端后重试";
  }
  if (/no such file|label_map|pipeline|model|joblib/i.test(message)) {
    return "分析服务配置不完整，请联系管理员检查模型文件";
  }
  if (/ffmpeg|decode|unsupported|format|audio_bytes must not be empty/i.test(message)) {
    return "无法读取这个音频文件，请更换音频格式后重试";
  }
  return "音频分析失败，请稍后重试";
}

export async function fetchHealth() {
  const response = await fetch("/api/health");
  if (!response.ok) throw new Error("health_check_failed");
  return response.json();
}

export async function analyzeAudioBlob(blob, filename) {
  let stage = "prepare";
  let responseStatus = 0;
  const safeFilename = safeUploadFilename(filename, blob);
  try {
    let response;
    try {
      stage = "formdata";
      const form = new FormData();
      form.append("file", blob, safeFilename);
      stage = "multipart_fetch";
      response = await fetch("/api/analyze-audio", { method: "POST", body: form });
    } catch (formError) {
      stage = "raw_fetch";
      response = await fetch("/api/analyze-audio", {
        method: "POST",
        headers: {
          "Content-Type": blob.type || "application/octet-stream",
          "X-Filename": safeFilename
        },
        body: blob
      });
    }

    responseStatus = response.status;
    stage = "read_response";
    const responseText = await response.text();
    let data;
    try {
      data = JSON.parse(responseText);
    } catch (error) {
      throw new Error(`response_not_json: ${responseText.slice(0, 120)}`);
    }
    if (!response.ok || data.ok === false) {
      throw new Error(data.message || data.error || "analysis_failed");
    }
    return data;
  } catch (error) {
    const detail = `stage=${stage}; type=${blob.type || "unknown"}; size=${blob.size || 0}; filename=${safeFilename}; error=${error.message || error}`;
    console.error("音频分析失败", detail);
    const userError = new Error(analysisErrorMessage(error, stage, responseStatus));
    userError.technicalDetail = detail;
    throw userError;
  }
}
