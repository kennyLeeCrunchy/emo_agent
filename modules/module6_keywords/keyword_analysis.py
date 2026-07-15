"""Rule-based keyword and reason-cue analysis for Module 6.

This module intentionally stays local and lightweight. It consumes ASR text
and model outputs, then returns structured evidence for the later Agent layer.
It does not call any LLM API and does not train or load emotion models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KeywordAnalysisInput:
    asr_text: str
    text_prediction: str | None = None
    fusion_prediction: str | None = None
    fusion_confidence: float | None = None
    emotion_change_points: list[dict[str, Any]] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)


KEYWORD_RULES: dict[str, dict[str, Any]] = {
    "sadness": {
        "label": "悲伤/低落",
        "reason_phrase": "低落或难过感",
        "terms": ["难过", "伤心", "失落", "委屈", "想哭", "低落", "崩溃", "心酸"],
    },
    "anger": {
        "label": "愤怒/烦躁",
        "reason_phrase": "烦躁或不满",
        "terms": ["生气", "气死", "烦死", "火大", "愤怒", "讨厌", "受不了"],
    },
    "fear": {
        "label": "害怕/担心",
        "reason_phrase": "担心或不安",
        "terms": ["害怕", "生怕", "可怕", "担心", "紧张", "焦虑", "不安"],
    },
    "happiness": {
        "label": "积极/开心",
        "reason_phrase": "开心或轻松感",
        "terms": ["开心", "高兴", "快乐", "舒服", "轻松", "喜欢", "太好了"],
    },
    "surprise": {
        "label": "惊讶",
        "reason_phrase": "惊讶或意外",
        "terms": ["惊讶", "没想到", "居然", "竟然", "意外", "吓一跳"],
    },
    "fatigue": {
        "label": "疲惫",
        "reason_phrase": "疲惫感",
        "terms": ["很累", "累了", "疲惫", "疲劳", "撑不住", "没力气"],
    },
    "stress": {
        "label": "压力",
        "reason_phrase": "压力感",
        "terms": ["压力", "来不及", "太多事", "忙不过来", "负担", "压着"],
    },
    "loneliness": {
        "label": "孤独/缺少支持",
        "reason_phrase": "孤独感或缺少支持",
        "terms": ["没人理解", "没人管", "一个人", "孤独", "孤单", "没人陪", "没人懂"],
    },
    "negation": {
        "label": "否定表达",
        "reason_phrase": "否定或回避表达",
        "terms": ["不想", "没有", "没关系", "不行", "不能", "不要", "不是"],
    },
    "self_harm_risk": {
        "label": "严重危机表达",
        "reason_phrase": "严重危机表达",
        "terms": ["不想活", "自杀", "伤害自己", "结束生命", "活不下去"],
    },
}


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def analyze_keywords_and_reasons(item: KeywordAnalysisInput | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, dict):
        item = KeywordAnalysisInput(**item)

    text = normalize_text(item.asr_text)
    keywords = extract_keywords(text, item.segments)
    snippets = build_source_snippets(text, keywords)
    reasons = build_possible_reasons(item, keywords)

    return {
        "asr_text": text,
        "text_prediction": item.text_prediction,
        "fusion_prediction": item.fusion_prediction,
        "fusion_confidence": item.fusion_confidence,
        "keywords": keywords,
        "source_snippets": snippets,
        "possible_reasons": reasons,
        "emotion_change_points": item.emotion_change_points,
    }


def extract_keywords(text: str, segments: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if not text:
        return []

    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for category, rule in KEYWORD_RULES.items():
        for term in sorted(rule["terms"], key=len, reverse=True):
            start = text.find(term)
            while start != -1:
                key = (category, term, start)
                if key not in seen:
                    end = start + len(term)
                    segment = find_segment_for_span(text, start, end, segments or [])
                    matches.append(
                        {
                            "keyword": term,
                            "category": category,
                            "category_label": rule["label"],
                            "start_char": start,
                            "end_char": end,
                            "snippet": make_snippet(text, start, end),
                            "start_time": segment.get("start_time"),
                            "end_time": segment.get("end_time"),
                        }
                    )
                    seen.add(key)
                start = text.find(term, start + 1)

    matches.sort(key=lambda entry: (entry["start_char"], entry["category"], -len(entry["keyword"])))
    return matches


def find_segment_for_span(
    full_text: str,
    start: int,
    end: int,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    matched_text = full_text[start:end]
    for segment in segments:
        segment_text = normalize_text(segment.get("text", ""))
        if matched_text and matched_text in segment_text:
            return segment
    return {}


def make_snippet(text: str, start: int, end: int, radius: int = 12) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right].strip(" ，。！？,.!?")


def build_source_snippets(text: str, keywords: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snippets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for keyword in keywords:
        snippet_text = keyword["snippet"]
        if snippet_text in seen:
            continue
        snippets.append(
            {
                "text": snippet_text,
                "keywords": [entry["keyword"] for entry in keywords if entry["snippet"] == snippet_text],
                "start_time": keyword.get("start_time"),
                "end_time": keyword.get("end_time"),
            }
        )
        seen.add(snippet_text)
    return snippets


def build_possible_reasons(item: KeywordAnalysisInput, keywords: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not keywords:
        return [
            {
                "type": "insufficient_text_evidence",
                "text": "没有提取到明确的文本关键词，因此原因分析应主要参考融合情绪结果和情绪曲线，避免过度解释。",
            }
        ]

    categories = {entry["category"] for entry in keywords}
    reason_phrases = collect_reason_phrases(categories)
    fusion = item.fusion_prediction or "unknown"
    confidence = format_confidence(item.fusion_confidence)

    reasons = [
        {
            "type": "fusion_conditioned_reason",
            "text": f"融合结果偏向 {fusion}{confidence}，文本中出现{join_cn(reason_phrases)}等线索，因此可以谨慎理解为可能与这些体验有关。",
        }
    ]

    if item.text_prediction and item.fusion_prediction and item.text_prediction != item.fusion_prediction:
        reasons.append(
            {
                "type": "modality_mismatch",
                "text": f"文本线索偏向 {item.text_prediction}，但融合结果更接近 {item.fusion_prediction}，说明文本证据和声学证据可能不完全一致，建议保留这种不确定性。",
            }
        )

    if item.emotion_change_points:
        reasons.append(
            {
                "type": "change_point_context",
                "text": "情绪变化点附近可优先引用这些关键词片段，但仍应使用可能、看起来、似乎等审慎表达。",
            }
        )

    if "self_harm_risk" in categories:
        reasons.append(
            {
                "type": "safety_notice",
                "text": "文本中出现严重危机表达，应建议联系可信赖的人或专业支持，并避免做医学诊断。",
            }
        )

    return reasons


def collect_reason_phrases(categories: set[str]) -> list[str]:
    phrases = []
    for category in KEYWORD_RULES:
        if category in categories:
            phrases.append(KEYWORD_RULES[category]["reason_phrase"])
    return phrases


def format_confidence(confidence: float | None) -> str:
    if confidence is None:
        return ""
    return f"（置信度 {confidence:.2f}）"


def join_cn(items: list[str]) -> str:
    unique = list(dict.fromkeys(items))
    if not unique:
        return "有限的文本"
    if len(unique) == 1:
        return unique[0]
    return "、".join(unique)
