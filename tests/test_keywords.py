"""
Tests for Module 6 keyword and reason-cue analysis.

The tests keep Module 6 fully local: no LLM API, no model loading, and no
network. They verify rule-based keyword extraction, source snippets, fusion
result conditioning, and safe fallback behavior for Agent inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.module6_keywords.keyword_analysis import (
    KeywordAnalysisInput,
    analyze_keywords_and_reasons,
)


def test_analyze_keywords_extracts_categories_snippets_and_fusion_reason() -> None:
    item = KeywordAnalysisInput(
        asr_text="今天真的很累，感觉没人理解我，但是我不想麻烦别人。",
        text_prediction="sad",
        fusion_prediction="sad",
        fusion_confidence=0.86,
        emotion_change_points=[{"time": 2.0, "emotion": "sad", "delta": 0.31}],
        segments=[{"start_time": 0.0, "end_time": 3.0, "text": "今天真的很累，感觉没人理解我。"}],
    )

    result = analyze_keywords_and_reasons(item)

    assert result["fusion_prediction"] == "sad"
    assert result["fusion_confidence"] == 0.86
    assert {entry["category"] for entry in result["keywords"]} >= {"fatigue", "loneliness", "negation"}
    assert any(entry["keyword"] == "很累" and entry["start_time"] == 0.0 for entry in result["keywords"])
    assert any("很累" in snippet["text"] for snippet in result["source_snippets"])
    assert any("融合结果偏向 sad" in reason["text"] for reason in result["possible_reasons"])
    assert any("疲惫" in reason["text"] and "缺少支持" in reason["text"] for reason in result["possible_reasons"])


def test_analyze_keywords_preserves_modality_mismatch_as_cautious_reason() -> None:
    item = KeywordAnalysisInput(
        asr_text="我真的有点难过，不过没关系。",
        text_prediction="sad",
        fusion_prediction="neutral",
        fusion_confidence=0.58,
    )

    result = analyze_keywords_and_reasons(item)

    assert any(entry["category"] == "sadness" for entry in result["keywords"])
    assert any(reason["type"] == "modality_mismatch" for reason in result["possible_reasons"])
    assert any("文本线索偏向 sad" in reason["text"] for reason in result["possible_reasons"])
    assert all("一定" not in reason["text"] and "肯定" not in reason["text"] for reason in result["possible_reasons"])


def test_analyze_keywords_handles_empty_text_without_crashing() -> None:
    item = KeywordAnalysisInput(
        asr_text="   ",
        text_prediction="neutral",
        fusion_prediction="neutral",
        fusion_confidence=0.44,
    )

    result = analyze_keywords_and_reasons(item)

    assert result["keywords"] == []
    assert result["source_snippets"] == []
    assert result["possible_reasons"][0]["type"] == "insufficient_text_evidence"
    assert "没有提取到明确的文本关键词" in result["possible_reasons"][0]["text"]


def test_analyze_keywords_avoids_single_character_false_positive_inside_words() -> None:
    item = KeywordAnalysisInput(
        asr_text="困难迫近的阁楼里，我生怕有什么可怕的东西突然冒出来。",
        fusion_prediction="fearful",
        fusion_confidence=0.92,
    )

    result = analyze_keywords_and_reasons(item)

    assert not any(entry["keyword"] == "困" for entry in result["keywords"])
    assert not any(entry["category"] == "fatigue" for entry in result["keywords"])
    assert [entry["keyword"] for entry in result["keywords"]].count("生怕") == 1
    assert [entry["keyword"] for entry in result["keywords"]].count("可怕") == 1
