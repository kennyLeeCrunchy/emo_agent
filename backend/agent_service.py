"""DeepSeek-backed emotion companion Agent with a local safe fallback."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


AGENT_OUTPUT_FIELDS = [
    "main_emotion_summary",
    "curve_interpretation",
    "possible_reasons",
    "companion_response",
    "gentle_suggestion",
    "safety_note",
    "uncertainty_notes",
]


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

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeepSeekConfig:
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        return cls(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            timeout=float(os.environ.get("DEEPSEEK_TIMEOUT", "30")),
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

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

    def generate(self, agent_input: AgentInput | dict[str, Any]) -> dict[str, Any]:
        if isinstance(agent_input, dict):
            agent_input = AgentInput.from_mapping(agent_input)

        if not self.config.configured:
            return local_template_response(agent_input, reason="missing_api_key")

        try:
            raw = self.post_json(
                self.config.chat_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                payload=build_deepseek_payload(agent_input, self.config.model),
                timeout=self.config.timeout,
            )
            parsed = parse_deepseek_response(raw)
            parsed["provider"] = "deepseek"
            return ensure_agent_output(parsed, fallback_input=agent_input)
        except Exception as exc:
            return local_template_response(agent_input, reason=f"deepseek_error:{type(exc).__name__}")


def build_deepseek_payload(agent_input: AgentInput, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "agent_input": agent_input.to_payload(),
                        "privacy_note": "Only structured text evidence and model outputs are provided. Raw audio is not included.",
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
    }


def parse_deepseek_response(raw: dict[str, Any]) -> dict[str, Any]:
    content = raw["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("DeepSeek content must be a JSON object.")
    return parsed


def ensure_agent_output(output: dict[str, Any], fallback_input: AgentInput) -> dict[str, Any]:
    fallback = local_template_response(fallback_input, reason="incomplete_deepseek_output")
    provider = output.get("provider", "deepseek")
    normalized = {field: output.get(field, fallback[field]) for field in AGENT_OUTPUT_FIELDS}
    if not isinstance(normalized["uncertainty_notes"], list):
        normalized["uncertainty_notes"] = [str(normalized["uncertainty_notes"])]
    normalized["provider"] = provider
    return normalized


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


SYSTEM_PROMPT = """你是一个中文语音情绪分析原型系统中的“情绪陪伴 Agent”。

你的任务不是重新进行情绪分类，也不是做心理诊断。你只能基于用户提供的结构化输入进行解释和回应。

必须遵守：
1. 不进行医学、心理疾病或精神健康诊断。
2. 不给用户贴病理标签。
3. 不使用“你一定”“你肯定”等绝对化表达。
4. 使用“可能”“看起来”“似乎”“从当前证据看”等审慎表达。
5. 不夸大模型能力，不把模型预测说成事实。
6. 当置信度低、模态结果不一致或 ASR 文本噪声较大时，必须明确保留不确定性。
7. 如出现自伤、自杀、伤害他人或严重危机表达，应温和建议联系可信赖的人、当地紧急服务或专业支持。

输出必须是 JSON 对象，字段为：
main_emotion_summary, curve_interpretation, possible_reasons,
companion_response, gentle_suggestion, safety_note, uncertainty_notes。
"""
