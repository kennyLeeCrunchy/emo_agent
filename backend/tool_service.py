"""Controlled tool registry for the text-first emotion companion Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from backend.memory_service import MemoryService


@dataclass
class ToolContext:
    session_id: str = "local"
    user_id: str = "local-user"
    current_message: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    latest_analysis: dict[str, Any] = field(default_factory=dict)
    memory_context: str = ""


class EmotionToolService:
    """Expose a small allow-listed set of JSON tools to the orchestrator."""

    def __init__(
        self,
        memory: MemoryService | None = None,
    ) -> None:
        self.memory = memory or MemoryService()
        self._handlers: dict[str, Callable[[dict[str, Any], ToolContext], dict[str, Any]]] = {
            "detect_emotion": self._detect_emotion,
            "session_emotion_curve": self._session_emotion_curve,
            "emotion_reason_analysis": self._emotion_reason_analysis,
            "memory_emotion_trend": self._memory_emotion_trend,
            "memory_search": self._memory_search,
            "memory_write": self._memory_write,
            "report_generation": self._report_generation,
        }

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def definitions(self) -> list[dict[str, Any]]:
        descriptions = {
            "detect_emotion": "基于本轮文字和最近一次真实语音分析，给出审慎的情绪线索。",
            "session_emotion_curve": "读取当前会话最近一次语音的窗口情绪曲线。",
            "emotion_reason_analysis": "读取或整理当前会话中的关键词和可能原因线索。",
            "memory_emotion_trend": "读取本地保存的跨会话长期情绪趋势，必须与单次语音窗口曲线区分。",
            "memory_search": "检索用户本地长期记忆中的历史情绪摘要、触发因素和已确认偏好。",
            "memory_write": "写入结构化长期记忆。偏好只在用户明确表达或要求记住时写入，并将 confirmed 设为 true。",
            "report_generation": "根据当前会话和最近分析生成简短的非诊断性摘要。",
        }
        definitions = []
        for name in self.names:
            if name == "memory_emotion_trend":
                parameters = {
                    "type": "object",
                    "properties": {
                        "range": {
                            "type": "string",
                            "enum": ["week", "month", "quarter", "year"],
                            "description": "需要聚合的长期时间范围。",
                        },
                    },
                    "additionalProperties": False,
                }
            elif name == "memory_search":
                parameters = {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "maxLength": 200, "description": "要检索的历史主题或偏好。"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                }
            elif name == "memory_write":
                parameters = {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["emotion_event", "trigger", "preference", "summary"],
                        },
                        "content": {"type": "string", "maxLength": 1000},
                        "emotion": {"type": "string", "maxLength": 40},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "triggers": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                        "key": {"type": "string", "maxLength": 80},
                        "value": {"type": "string", "maxLength": 500},
                        "confirmed": {
                            "type": "boolean",
                            "description": "偏好仅在用户本轮明确表达或要求记住时设为 true。",
                        },
                    },
                    "required": ["kind"],
                    "additionalProperties": False,
                }
            else:
                parameters = {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "需要分析的用户文字。"},
                        "query": {"type": "string", "description": "检索或摘要目标。"},
                    },
                    "additionalProperties": False,
                }
            definitions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": descriptions[name],
                    "parameters": parameters,
                },
            })
        return definitions

    def execute(self, name: str, arguments: dict[str, Any] | None, context: ToolContext) -> dict[str, Any]:
        if name not in self._handlers:
            raise ValueError(f"Tool is not allowed: {name}")
        return self._handlers[name](arguments or {}, context)

    def _detect_emotion(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        text = str(arguments.get("text") or "")
        analysis = context.latest_analysis
        if analysis.get("fusion_prediction"):
            return {
                "source": "latest_audio_analysis",
                "emotion": analysis.get("fusion_prediction"),
                "confidence": analysis.get("fusion_confidence"),
                "note": "来自最近一次真实语音融合分析。",
            }
        lexicon = {
            "sad": ("难过", "伤心", "失落", "孤独", "累"),
            "angry": ("生气", "愤怒", "恼火"),
            "fearful": ("害怕", "担心", "焦虑", "紧张"),
            "happy": ("开心", "高兴", "快乐"),
        }
        emotion = next((label for label, words in lexicon.items() if any(word in text for word in words)), "unknown")
        return {
            "source": "text_keyword_hint",
            "emotion": emotion,
            "confidence": None,
            "note": "仅为文字关键词线索，不替代语音模型或专业判断。",
        }

    def _session_emotion_curve(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        del arguments
        curve = context.latest_analysis.get("emotion_curve") or []
        return {
            "source": "latest_audio_analysis",
            "available": bool(curve),
            "points": curve,
            "change_points": context.latest_analysis.get("emotion_change_points") or [],
        }

    def _emotion_reason_analysis(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        return {
            "text": str(arguments.get("text") or ""),
            "keywords": context.latest_analysis.get("keywords") or [],
            "possible_reasons": context.latest_analysis.get("possible_reasons") or [],
            "note": "原因只作为可能线索，不作确定归因。",
        }

    def _memory_emotion_trend(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        return self.memory.emotion_trend(context.user_id, arguments.get("range") or "month")

    def _memory_search(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        return self.memory.search(
            context.user_id,
            str(arguments.get("query") or arguments.get("text") or ""),
            arguments.get("limit") or 8,
        )

    def _memory_write(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        kind = str(arguments.get("kind") or "")
        if kind == "preference" and not any(
            term in context.current_message for term in ("记住", "以后请", "我喜欢", "我不喜欢", "偏好")
        ):
            return {"ok": False, "available": True, "error": "preference_not_explicit"}
        if kind in {"emotion_event", "trigger", "summary"} and not context.latest_analysis.get("fusion_prediction"):
            return {"ok": False, "available": True, "error": "structured_analysis_required"}
        return self.memory.write_memory(context.user_id, context.session_id, arguments)

    def _report_generation(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        analysis = context.latest_analysis
        return {
            "title": "当前会话情绪摘要",
            "request": arguments.get("query") or arguments.get("text"),
            "fusion_prediction": analysis.get("fusion_prediction"),
            "fusion_confidence": analysis.get("fusion_confidence"),
            "keywords": analysis.get("keywords") or [],
            "possible_reasons": analysis.get("possible_reasons") or [],
            "message_count": len(context.history),
            "safety_note": "该摘要不用于医疗或心理诊断。",
        }
