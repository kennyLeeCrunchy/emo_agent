"""
Tests for Module 5 multimodal emotion curve helpers.

The tests avoid loading Whisper, RoBERTa, or real model files. They cover the
window timeline, fusion feature construction, probability time series, curve
summary, and artifact writing with toy arrays.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.module5_emotion_curve.emotion_curve import (
    CurveConfig,
    analyze_multimodal_emotion_curve,
    build_curve_summary,
    build_multimodal_curve,
    build_plot_annotations,
    build_window_fusion_features,
    iter_sliding_windows,
    save_curve_outputs,
)
from modules.module5_emotion_curve.run_multimodal_curve import select_sample, write_run_metadata


class ToyFusionModel:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        rows = []
        for row in features:
            sad_score = float(row[0] + row[2])
            angry_score = float(row[1] + row[3])
            neutral_score = 1.0
            scores = np.array([angry_score, neutral_score, sad_score], dtype=np.float64)
            scores = np.maximum(scores, 0.01)
            rows.append(scores / scores.sum())
        return np.vstack(rows)


class ToyAudioExtractor:
    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.array([float(waveform.mean() / 100000.0), float(waveform.std() / 100000.0)], dtype=np.float32)


class ToyTextExtractor:
    def extract_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([[0.1, 0.2 + len(texts[0]) / 100.0]], dtype=np.float32)


def test_iter_sliding_windows_keeps_short_audio_as_one_window() -> None:
    waveform = np.arange(8000, dtype=np.float32)

    windows = list(iter_sliding_windows(waveform, sample_rate=16000, window_seconds=2.0, step_seconds=1.0))

    assert len(windows) == 1
    assert windows[0].index == 0
    assert windows[0].start_time == 0.0
    assert windows[0].end_time == 0.5
    np.testing.assert_array_equal(windows[0].waveform, waveform)


def test_iter_sliding_windows_uses_fixed_step_and_keeps_tail() -> None:
    waveform = np.arange(5 * 16000, dtype=np.float32)

    windows = list(iter_sliding_windows(waveform, sample_rate=16000, window_seconds=2.0, step_seconds=1.5))

    assert [(item.start_time, item.end_time) for item in windows] == [(0.0, 2.0), (1.5, 3.5), (3.0, 5.0)]
    assert [len(item.waveform) for item in windows] == [32000, 32000, 32000]


def test_iter_sliding_windows_skips_tiny_tail_overlap() -> None:
    waveform = np.arange(int(8.49 * 16000), dtype=np.float32)

    windows = list(iter_sliding_windows(waveform, sample_rate=16000, window_seconds=2.0, step_seconds=2.0))

    assert [(item.start_time, item.end_time) for item in windows] == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)]


def test_build_window_fusion_features_concatenates_audio_then_text_per_window() -> None:
    audio = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    text = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)

    fused = build_window_fusion_features(audio, text, text_mode="repeat_global")

    assert fused.dtype == np.float32
    np.testing.assert_allclose(
        fused,
        np.array([[1.0, 2.0, 10.0, 20.0, 30.0], [3.0, 4.0, 10.0, 20.0, 30.0]], dtype=np.float32),
    )


def test_build_multimodal_curve_outputs_probabilities_and_predictions() -> None:
    windows = list(iter_sliding_windows(np.arange(4 * 16000, dtype=np.float32), 16000, 2.0, 2.0))
    audio = np.array([[0.05, 0.05], [2.0, 0.1]], dtype=np.float32)
    text = np.array([[0.05, 0.1]], dtype=np.float32)
    label_names = ["angry", "neutral", "sad"]

    curve = build_multimodal_curve(
        windows=windows,
        audio_embeddings=audio,
        text_embeddings=text,
        fusion_model=ToyFusionModel(),
        label_names=label_names,
        text_mode="repeat_global",
    )

    assert list(curve["window_index"]) == [0, 1]
    assert {"prob_angry", "prob_neutral", "prob_sad", "pred_emotion", "confidence"}.issubset(curve.columns)
    np.testing.assert_allclose(curve[["prob_angry", "prob_neutral", "prob_sad"]].sum(axis=1), np.ones(2))
    assert curve.loc[1, "pred_emotion"] == "sad"


def test_build_curve_summary_finds_dominant_emotion_change_points_and_high_segments() -> None:
    windows = list(iter_sliding_windows(np.arange(6 * 16000, dtype=np.float32), 16000, 2.0, 2.0))
    audio = np.array([[0.05, 0.05], [8.0, 0.1], [0.1, 8.2]], dtype=np.float32)
    text = np.array([[0.05, 0.1]], dtype=np.float32)
    label_names = ["angry", "neutral", "sad"]
    curve = build_multimodal_curve(windows, audio, text, ToyFusionModel(), label_names)

    summary = build_curve_summary(curve, label_names, high_emotion_threshold=0.45, change_delta=0.2)

    assert summary["dominant_emotion"] in {"angry", "sad"}
    assert summary["volatility_strength"] > 0
    assert len(summary["emotion_change_points"]) >= 1
    assert {item["emotion"] for item in summary["high_emotion_segments"]} == {"sad", "angry"}


def test_save_curve_outputs_writes_csv_json_and_figure(tmp_path: Path) -> None:
    windows = list(iter_sliding_windows(np.arange(4 * 16000, dtype=np.float32), 16000, 2.0, 2.0))
    audio = np.array([[0.05, 0.05], [2.0, 0.1]], dtype=np.float32)
    text = np.array([[0.05, 0.1]], dtype=np.float32)
    label_names = ["angry", "neutral", "sad"]
    curve = build_multimodal_curve(windows, audio, text, ToyFusionModel(), label_names)
    summary = build_curve_summary(curve, label_names)
    config = CurveConfig(window_seconds=2.0, step_seconds=2.0, text_mode="repeat_global")

    paths = save_curve_outputs(
        curve=curve,
        summary=summary,
        label_names=label_names,
        output_dir=tmp_path,
        run_id="toy_curve",
        config=config,
    )

    assert paths["curve_csv"].exists()
    assert paths["summary_json"].exists()
    assert paths["figure_png"].exists()
    assert paths["config_json"].exists()


def test_analyze_multimodal_emotion_curve_uses_injected_extractors() -> None:
    waveform = np.linspace(0.0, 32000.0, 4 * 16000, dtype=np.float32)
    label_names = ["angry", "neutral", "sad"]

    curve, summary = analyze_multimodal_emotion_curve(
        waveform=waveform,
        sample_rate=16000,
        asr_text="今天有点累",
        audio_extractor=ToyAudioExtractor(),
        text_extractor=ToyTextExtractor(),
        fusion_model=ToyFusionModel(),
        label_names=label_names,
        config=CurveConfig(window_seconds=2.0, step_seconds=2.0, text_mode="repeat_global"),
    )

    assert len(curve) == 2
    assert summary["dominant_emotion"] in label_names
    assert "prob_sad" in curve.columns


def test_select_sample_can_choose_first_row_from_requested_split() -> None:
    metadata = pd.DataFrame(
        {
            "sample_id": ["train-a", "test-a", "test-b"],
            "split": ["train", "test", "test"],
        }
    )

    row = select_sample(metadata, sample_index=0, sample_id=None, sample_split="test")

    assert row["sample_id"] == "test-a"


def test_write_run_metadata_records_split(tmp_path: Path) -> None:
    sample_row = pd.Series(
        {
            "sample_id": "test-a",
            "split": "test",
            "emotion": "fearful",
            "label_id": 1,
            "parquet_file": "train-00000-of-00008.parquet",
            "row_id": 105,
        }
    )
    args = Namespace(
        fusion_model_path="fusion.joblib",
        whisper_model="openai/whisper-tiny",
        text_model="roberta",
    )
    (tmp_path / "curves").mkdir()

    path = write_run_metadata(tmp_path, "sample_test_a", sample_row, "测试文本", args)

    assert '"split": "test"' in path.read_text(encoding="utf-8")


def test_build_plot_annotations_exposes_summary_layers_for_figure() -> None:
    summary = {
        "dominant_emotion": "angry",
        "volatility_strength": 0.73,
        "emotion_change_points": [{"time": 2.0}, {"time": 4.0}],
        "high_emotion_segments": [{"emotion": "sad", "start_time": 1.0, "end_time": 3.0}],
    }

    annotations = build_plot_annotations(summary)

    assert annotations["info_text"] == "dominant=angry | volatility=0.730"
    assert annotations["change_times"] == [2.0, 4.0]
    assert annotations["high_segments"] == [{"emotion": "sad", "start_time": 1.0, "end_time": 3.0}]
