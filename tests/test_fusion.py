"""
Tests for Module 4 Whisper+BERT/RoBERTa early fusion.

These tests use toy arrays and do not load Whisper or BERT weights. They cover
audio/text id alignment, feature concatenation, fusion Logistic Regression
outputs, baseline comparison, and modality-disagreement export.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.module4_fusion.train_fusion_logistic import (
    build_fusion_features,
    build_model_comparison,
    build_modality_disagreements,
    label_names_from_map,
    load_fusion_inputs,
    train_and_save,
    validate_modal_alignment,
)


def make_ids() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"id-{idx}" for idx in range(21)],
            "text": [f"测试文本 {idx}" for idx in range(21)],
            "emotion": ["angry", "happy", "sad"] * 7,
            "label_id": [0, 1, 2] * 7,
            "split": ["train"] * 12 + ["val"] * 3 + ["test"] * 6,
            "parquet_file": ["part.parquet"] * 21,
            "row_id": list(range(21)),
        }
    )


def test_validate_modal_alignment_rejects_sample_order_mismatch() -> None:
    audio_ids = make_ids().drop(columns=["text"])
    text_ids = make_ids()
    text_ids.loc[1, "sample_id"] = "different"

    try:
        validate_modal_alignment(audio_ids, text_ids)
    except ValueError as exc:
        assert "sample_id" in str(exc)
    else:
        raise AssertionError("Expected sample_id mismatch to raise ValueError")


def test_build_fusion_features_concatenates_audio_then_text() -> None:
    audio = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    text = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)

    fused = build_fusion_features(audio, text)

    assert fused.dtype == np.float32
    assert fused.shape == (2, 5)
    np.testing.assert_allclose(
        fused,
        np.array([[1.0, 2.0, 10.0, 20.0, 30.0], [3.0, 4.0, 40.0, 50.0, 60.0]], dtype=np.float32),
    )


def test_load_fusion_inputs_checks_rows_and_label_map(tmp_path: Path) -> None:
    audio_ids = make_ids().drop(columns=["text"])
    text_ids = make_ids()
    audio_path = tmp_path / "audio.npy"
    text_path = tmp_path / "text.npy"
    audio_ids_path = tmp_path / "audio_ids.csv"
    text_ids_path = tmp_path / "text_ids.csv"
    label_map_path = tmp_path / "label_map.json"

    np.save(audio_path, np.zeros((21, 2), dtype=np.float32))
    np.save(text_path, np.zeros((21, 3), dtype=np.float32))
    audio_ids.to_csv(audio_ids_path, index=False, encoding="utf-8-sig")
    text_ids.to_csv(text_ids_path, index=False, encoding="utf-8-sig")
    label_map_path.write_text(json.dumps({"angry": 0, "happy": 1, "sad": 2}), encoding="utf-8")

    features, ids, label_map, feature_info = load_fusion_inputs(
        audio_feature_path=audio_path,
        audio_ids_path=audio_ids_path,
        text_feature_path=text_path,
        text_ids_path=text_ids_path,
        label_map_path=label_map_path,
    )

    assert features.shape == (21, 5)
    assert "text" in ids.columns
    assert label_names_from_map(label_map) == ["angry", "happy", "sad"]
    assert feature_info == {"audio_dim": 2, "text_dim": 3, "fusion_dim": 5}


def test_train_and_save_writes_expected_fusion_outputs(tmp_path: Path) -> None:
    ids = make_ids()
    labels = ids["label_id"].to_numpy()
    audio_features = np.column_stack(
        [
            labels == 0,
            labels == 1,
            np.arange(len(labels)) / 100.0,
        ]
    ).astype(np.float32)
    text_features = np.column_stack(
        [
            labels == 2,
            labels == 1,
            np.arange(len(labels)) / 50.0,
        ]
    ).astype(np.float32)

    audio_path = tmp_path / "audio.npy"
    text_path = tmp_path / "text.npy"
    audio_ids_path = tmp_path / "audio_ids.csv"
    text_ids_path = tmp_path / "text_ids.csv"
    label_map_path = tmp_path / "label_map.json"
    output_dir = tmp_path / "01_fusion_logistic_classifier"

    np.save(audio_path, audio_features)
    np.save(text_path, text_features)
    ids.drop(columns=["text"]).to_csv(audio_ids_path, index=False, encoding="utf-8-sig")
    ids.to_csv(text_ids_path, index=False, encoding="utf-8-sig")
    label_map_path.write_text(json.dumps({"angry": 0, "happy": 1, "sad": 2}), encoding="utf-8")

    result = train_and_save(
        audio_feature_path=audio_path,
        audio_ids_path=audio_ids_path,
        text_feature_path=text_path,
        text_ids_path=text_ids_path,
        label_map_path=label_map_path,
        output_dir=output_dir,
        max_iter=1000,
    )

    assert (output_dir / "model" / "fusion_logistic_pipeline.joblib").exists()
    assert (output_dir / "model" / "config.json").exists()
    assert (output_dir / "metrics" / "metrics.csv").exists()
    assert (output_dir / "metrics" / "classification_report.csv").exists()
    assert (output_dir / "metrics" / "predictions.csv").exists()
    assert (output_dir / "metrics" / "model_comparison.csv").exists()
    assert (output_dir / "metrics" / "modality_disagreements.csv").exists()
    assert (output_dir / "figures" / "confusion_matrix_test.png").exists()
    assert set(result["metrics"]["split"]) == {"train", "val", "test"}
    assert result["config"]["feature_info"] == {"audio_dim": 3, "text_dim": 3, "fusion_dim": 6}


def test_build_model_comparison_keeps_audio_text_and_fusion_rows() -> None:
    audio = pd.DataFrame({"split": ["test"], "macro_f1": [0.9], "accuracy": [0.91]})
    text = pd.DataFrame({"split": ["test"], "macro_f1": [1.0], "accuracy": [1.0]})
    fusion = pd.DataFrame({"split": ["test"], "macro_f1": [0.98], "accuracy": [0.99]})

    comparison = build_model_comparison(audio, text, fusion)

    assert set(comparison["model"]) == {"audio_whisper", "text_roberta", "fusion_whisper_roberta"}
    assert comparison.loc[comparison["model"].eq("fusion_whisper_roberta"), "macro_f1"].iloc[0] == 0.98


def test_build_modality_disagreements_marks_audio_text_prediction_conflicts() -> None:
    audio_predictions = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "split": ["test", "test"],
            "pred_emotion": ["sad", "happy"],
            "confidence": [0.7, 0.8],
        }
    )
    text_predictions = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "text": ["我没事", "真开心"],
            "emotion": ["neutral", "happy"],
            "label_id": [0, 1],
            "split": ["test", "test"],
            "pred_emotion": ["neutral", "happy"],
            "confidence": [0.6, 0.9],
        }
    )
    fusion_predictions = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "split": ["test", "test"],
            "pred_emotion": ["sad", "happy"],
            "confidence": [0.65, 0.95],
        }
    )

    disagreements = build_modality_disagreements(audio_predictions, text_predictions, fusion_predictions)

    assert list(disagreements["sample_id"]) == ["a"]
    assert disagreements.iloc[0]["audio_pred_emotion"] == "sad"
    assert disagreements.iloc[0]["text_pred_emotion"] == "neutral"
    assert disagreements.iloc[0]["fusion_pred_emotion"] == "sad"
