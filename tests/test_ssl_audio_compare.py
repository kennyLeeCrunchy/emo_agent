"""
Test the lightweight HuBERT/Whisper comparison helpers without loading large models.

These tests cover waveform preparation, id alignment, and Logistic Regression
metric generation while keeping the expensive SSL encoders out of the test path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.module2_ssl_audio.compare_ssl_features import (
    evaluate_feature_set,
    select_ssl_recommendation,
    validate_matching_ids,
)
from modules.module2_ssl_audio.extract_ssl_features import prepare_waveform


def test_prepare_waveform_resamples_and_returns_finite_mono_audio() -> None:
    waveform = np.sin(np.linspace(0, 2 * np.pi, 800, dtype=np.float32))

    prepared = prepare_waveform(waveform, sample_rate=8000, target_sample_rate=16000)

    assert prepared.dtype == np.float32
    assert prepared.ndim == 1
    assert prepared.shape[0] > waveform.shape[0]
    assert np.isfinite(prepared).all()


def test_validate_matching_ids_rejects_misaligned_rows() -> None:
    reference = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "emotion": ["angry", "happy"],
            "label_id": [0, 1],
            "split": ["train", "test"],
        }
    )
    candidate = reference.copy()
    candidate.loc[1, "sample_id"] = "c"

    try:
        validate_matching_ids(reference, candidate)
    except ValueError as exc:
        assert "sample_id" in str(exc)
    else:
        raise AssertionError("Expected id mismatch to raise ValueError")


def test_evaluate_feature_set_returns_val_and_test_metrics() -> None:
    ids = pd.DataFrame(
        {
            "sample_id": [f"id-{idx}" for idx in range(18)],
            "emotion": ["angry", "happy", "sad"] * 6,
            "label_id": [0, 1, 2] * 6,
            "split": ["train"] * 12 + ["val"] * 3 + ["test"] * 3,
        }
    )
    labels = ids["label_id"].to_numpy()
    features = np.column_stack(
        [
            labels == 0,
            labels == 1,
            labels == 2,
            np.arange(len(labels)) / 100.0,
        ]
    ).astype(np.float32)

    metrics, predictions = evaluate_feature_set(
        feature_name="toy",
        features=features,
        ids=ids,
        pca_components=2,
    )

    assert set(metrics["eval_split"]) == {"val", "test"}
    assert set(metrics["feature_set"]) == {"toy"}
    assert {"sample_id", "pred_label_id", "correct"}.issubset(predictions.columns)


def test_select_ssl_recommendation_uses_best_validation_variant() -> None:
    metrics = pd.DataFrame(
        [
            {"feature_set": "hubert", "variant": "raw", "eval_split": "val", "accuracy": 0.3, "macro_f1": 0.3, "balanced_accuracy": 0.3, "feature_dim": 768},
            {"feature_set": "hubert", "variant": "pca64", "eval_split": "val", "accuracy": 0.5, "macro_f1": 0.4, "balanced_accuracy": 0.5, "feature_dim": 768},
            {"feature_set": "whisper", "variant": "raw", "eval_split": "val", "accuracy": 0.7, "macro_f1": 0.6, "balanced_accuracy": 0.7, "feature_dim": 384},
            {"feature_set": "hubert", "variant": "raw", "eval_split": "test", "accuracy": 0.4, "macro_f1": 0.3, "balanced_accuracy": 0.4, "feature_dim": 768},
            {"feature_set": "hubert", "variant": "pca64", "eval_split": "test", "accuracy": 0.6, "macro_f1": 0.5, "balanced_accuracy": 0.6, "feature_dim": 768},
            {"feature_set": "whisper", "variant": "raw", "eval_split": "test", "accuracy": 0.8, "macro_f1": 0.7, "balanced_accuracy": 0.8, "feature_dim": 384},
        ]
    )

    summary = select_ssl_recommendation(metrics)

    assert summary.iloc[0]["feature_set"] == "whisper"
    assert summary.loc[summary["feature_set"].eq("hubert"), "selected_variant"].iloc[0] == "pca64"
