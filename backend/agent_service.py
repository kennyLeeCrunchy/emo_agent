"""DeepSeek-backed emotion companion Agent with a local safe fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AgentInput:
    asr_text: str
    audio_prediction: str | None = None
    text_prediction: str | None = None
    fusion_prediction: str | None = None
    audio_confidence: float | None = None
    text_confidence: float | None = None
    fusion_confidence: float | None = None
    emotion_curve: list[dict[str, Any]] = field(default_factory=list)
    keywords: list[Any] = field(default_factory=list)
    emotion_change_points: list[Any] = field(default_factory=list)
    possible_reasons: list[Any] = field(default_factory=list)
    source_snippets: list[Any] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AgentInput":
        allowed = cls.__dataclass_fields__.keys()
        filtered = {key: data.get(key) for key in allowed if key in data}
        if "asr_text" not in filtered:
            filtered["asr_text"] = ""
        return cls(**filtered)

@dataclass
class DeepSeekConfig:
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    timeout: float = 30.0
    max_tokens: int = 1000
    summary_max_tokens: int = 1500
    max_input_chars: int = 2000
    max_history_chars: int = 200000
    max_tool_result_chars: int = 3000
    max_tool_results_tokens: int = 6000
    context_trigger_tokens: int = 49152
    context_target_tokens: int = 32768
    context_hard_tokens: int = 65536
    recent_message_limit: int = 20
    recent_message_keep: int = 8
    summary_retry_attempts: int = 3
    summary_retry_delay_seconds: float = 30.0
    summary_cooldown_seconds: int = 600
    enable_web_search: bool = True

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        return cls(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            timeout=float(os.environ.get("DEEPSEEK_TIMEOUT", "30")),
            max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "1000")),
            summary_max_tokens=int(os.environ.get("DEEPSEEK_SUMMARY_MAX_TOKENS", "1500")),
            max_input_chars=int(os.environ.get("DEEPSEEK_MAX_INPUT_CHARS", "2000")),
            max_history_chars=int(os.environ.get("DEEPSEEK_MAX_HISTORY_CHARS", "200000")),
            max_tool_result_chars=int(os.environ.get("DEEPSEEK_MAX_TOOL_RESULT_CHARS", "3000")),
            max_tool_results_tokens=int(os.environ.get("DEEPSEEK_MAX_TOOL_RESULTS_TOKENS", "6000")),
            context_trigger_tokens=int(os.environ.get("DEEPSEEK_CONTEXT_TRIGGER_TOKENS", "49152")),
            context_target_tokens=int(os.environ.get("DEEPSEEK_CONTEXT_TARGET_TOKENS", "32768")),
            context_hard_tokens=int(os.environ.get("DEEPSEEK_CONTEXT_HARD_TOKENS", "65536")),
            recent_message_limit=int(os.environ.get("DEEPSEEK_RECENT_MESSAGE_LIMIT", "20")),
            recent_message_keep=int(os.environ.get("DEEPSEEK_RECENT_MESSAGE_KEEP", "8")),
            summary_retry_attempts=int(os.environ.get("DEEPSEEK_SUMMARY_RETRY_ATTEMPTS", "3")),
            summary_retry_delay_seconds=float(os.environ.get("DEEPSEEK_SUMMARY_RETRY_DELAY_SECONDS", "30")),
            summary_cooldown_seconds=int(os.environ.get("DEEPSEEK_SUMMARY_COOLDOWN_SECONDS", "600")),
            enable_web_search=os.environ.get("DEEPSEEK_ENABLE_WEB_SEARCH", "true").strip().lower()
            in {"1", "true", "yes", "on"},
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def responses_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/responses"

    def __repr__(self) -> str:
        masked_key = "***" if self.api_key else None
        return (
            "DeepSeekConfig("
            f"api_key={masked_key}, base_url={self.base_url!r}, "
            f"model={self.model!r}, timeout={self.timeout!r})"
        )


def default_post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    import requests

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


class EmotionCompanionAgent:
    def __init__(
        self,
        config: DeepSeekConfig | None = None,
        post_json: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or DeepSeekConfig.from_env()
        self.post_json = post_json or default_post_json


def local_template_response(agent_input: AgentInput, reason: str = "fallback") -> dict[str, Any]:
    fusion = agent_input.fusion_prediction or "unknown"
    confidence = _format_confidence(agent_input.fusion_confidence)
    reasons = _reasons_to_text(agent_input.possible_reasons)
    uncertainty = list(agent_input.uncertainty_notes)
    if reason:
        uncertainty.append(reason)
    if agent_input.audio_prediction and agent_input.text_prediction and agent_input.audio_prediction != agent_input.text_prediction:
        uncertainty.append("audio_text_prediction_mismatch")
    if agent_input.fusion_confidence is not None and agent_input.fusion_confidence < 0.6:
        uncertainty.append("low_fusion_confidence")

    safety_note = "本系统不用于医疗诊断；如果出现自伤、伤害他人或严重危机表达，请尽快联系可信赖的人或当地专业支持。"
    if _has_crisis_signal(agent_input):
        safety_note = "文本中似乎出现严重危机表达。本系统不用于医疗诊断，请立刻联系可信赖的人、当地紧急服务或专业支持。"

    return {
        "main_emotion_summary": f"从当前结构化证据看，融合结果可能偏向 {fusion}{confidence}，这个判断仍应结合 ASR 文本和声学线索谨慎理解。",
        "curve_interpretation": _curve_summary(agent_input),
        "possible_reasons": reasons or f"目前可引用的关键词线索较少，可以先谨慎参考融合情绪 {fusion}，避免把模型输出当作确定原因。",
        "companion_response": "听起来这段表达里有一些需要被看见的感受。你不需要马上把所有事情解释清楚，先把此刻的感受放在这里，也是可以的。",
        "gentle_suggestion": "可以先喝一点水，做三次慢呼吸，然后用一句话写下此刻最明显的感受。",
        "safety_note": safety_note,
        "uncertainty_notes": list(dict.fromkeys(uncertainty)),
        "provider": "local_template",
    }


def _format_confidence(confidence: float | None) -> str:
    if confidence is None:
        return ""
    return f"（置信度 {confidence:.2f}）"


def _reasons_to_text(reasons: list[Any]) -> str:
    texts: list[str] = []
    for item in reasons:
        if isinstance(item, dict):
            text = item.get("text")
        else:
            text = item
        if text:
            texts.append(str(text))
    return " ".join(texts)


def _curve_summary(agent_input: AgentInput) -> str:
    if not agent_input.emotion_curve:
        return "当前没有可用的情绪曲线，因此暂时无法可靠判断情绪随时间的变化趋势。"
    if agent_input.emotion_change_points:
        return "情绪曲线中出现了可能的转折点，建议结合对应时间附近的文本或语音片段谨慎解释。"
    return "情绪曲线提供了时间序列参考，但当前未检测到明确转折点，整体趋势需要结合置信度一起理解。"


def _has_crisis_signal(agent_input: AgentInput) -> bool:
    text = agent_input.asr_text or ""
    crisis_terms = ["不想活", "自杀", "伤害自己", "结束生命", "活不下去"]
    if any(term in text for term in crisis_terms):
        return True
    for item in agent_input.keywords:
        if isinstance(item, dict) and item.get("category") == "self_harm_risk":
            return True
    return False
