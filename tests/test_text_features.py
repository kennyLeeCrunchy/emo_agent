"""
Tests for Module 3 text embeddings and text Logistic Regression helpers.

The tests avoid loading BERT/RoBERTa weights. They cover text cleaning,
attention-mask mean pooling, id validation, feature/id alignment, and the
lightweight classifier evaluation path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.module3_text_features.extract_text_features import (
    clean_text,
    mean_pool_last_hidden_state,
    validate_text_ids,
)
from modules.module3_text_features.train_text_logistic import (
    evaluate_split,
    label_names_from_map,
    load_inputs,
    train_and_save,
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


def test_clean_text_collapses_whitespace_and_rejects_empty_text() -> None:
    assert clean_text("  今天\t真的\n很累  ") == "今天 真的 很累"

    try:
        clean_text(" \n\t ")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected empty text to raise ValueError")


def test_mean_pool_last_hidden_state_uses_attention_mask() -> None:
    hidden = torch.tensor(
        [
            [
                [1.0, 1.0],
                [3.0, 5.0],
                [99.0, 99.0],
            ]
        ]
    )
    mask = torch.tensor([[1, 1, 0]])

    pooled = mean_pool_last_hidden_state(hidden, mask)

    np.testing.assert_allclose(pooled.numpy(), np.array([[2.0, 3.0]], dtype=np.float32))


def test_validate_text_ids_rejects_missing_columns_and_bad_rows() -> None:
    ids = make_ids()
    validate_text_ids(ids)

    missing = ids.drop(columns=["text"])
    try:
        validate_text_ids(missing)
    except ValueError as exc:
        assert "text" in str(exc)
    else:
        raise AssertionError("Expected missing text column to raise ValueError")

    duplicate = pd.concat([ids, ids.head(1)], ignore_index=True)
    try:
        validate_text_ids(duplicate)
    except ValueError as exc:
        assert "sample_id" in str(exc)
    else:
        raise AssertionError("Expected duplicate sample_id to raise ValueError")


def test_load_inputs_rejects_feature_id_row_mismatch(tmp_path: Path) -> None:
    ids_path = tmp_path / "text_ids.csv"
    feature_path = tmp_path / "text_embeddings.npy"
    label_map_path = tmp_path / "label_map.json"
    make_ids().to_csv(ids_path, index=False, encoding="utf-8-sig")
    np.save(feature_path, np.zeros((20, 4), dtype=np.float32))
    label_map_path.write_text(json.dumps({"angry": 0, "happy": 1, "sad": 2}), encoding="utf-8")

    try:
        load_inputs(feature_path, ids_path, label_map_path)
    except ValueError as exc:
        assert "row mismatch" in str(exc)
    else:
        raise AssertionError("Expected feature/id mismatch to raise ValueError")


def test_evaluate_split_outputs_probabilities_for_each_label() -> None:
    ids = make_ids()
    labels = ids["label_id"].to_numpy()
    features = np.column_stack(
        [
            labels == 0,
            labels == 1,
            labels == 2,
            np.arange(len(labels)) / 100.0,
        ]
    ).astype(np.float32)
    label_names = ["angry", "happy", "sad"]

    result = train_and_save_arrays_for_test(features, ids, label_names)
    metrics, predictions, matrix = evaluate_split(
        result,
        features,
        ids,
        split_name="test",
        label_names=label_names,
    )

    assert metrics["split"] == "test"
    assert metrics["n_samples"] == 6
    assert matrix.shape == (3, 3)
    assert {"pred_label_id", "pred_emotion", "confidence", "prob_angry", "prob_happy", "prob_sad"}.issubset(
        predictions.columns
    )
    np.testing.assert_allclose(
        predictions[["prob_angry", "prob_happy", "prob_sad"]].sum(axis=1).to_numpy(),
        np.ones(len(predictions)),
    )


def test_train_and_save_writes_expected_text_model_outputs(tmp_path: Path) -> None:
    ids = make_ids()
    labels = ids["label_id"].to_numpy()
    features = np.column_stack(
        [
            labels == 0,
            labels == 1,
            labels == 2,
            np.arange(len(labels)) / 50.0,
        ]
    ).astype(np.float32)

    feature_path = tmp_path / "text_embeddings.npy"
    ids_path = tmp_path / "text_ids.csv"
    label_map_path = tmp_path / "label_map.json"
    output_dir = tmp_path / "02_text_logistic_classifier"
    np.save(feature_path, features)
    ids.to_csv(ids_path, index=False, encoding="utf-8-sig")
    label_map_path.write_text(json.dumps({"angry": 0, "happy": 1, "sad": 2}), encoding="utf-8")

    result = train_and_save(
        feature_path=feature_path,
        ids_path=ids_path,
        label_map_path=label_map_path,
        output_dir=output_dir,
        max_iter=1000,
    )

    assert (output_dir / "model" / "text_logistic_pipeline.joblib").exists()
    assert (output_dir / "model" / "config.json").exists()
    assert (output_dir / "metrics" / "metrics.csv").exists()
    assert (output_dir / "metrics" / "classification_report.csv").exists()
    assert (output_dir / "metrics" / "predictions.csv").exists()
    assert set(result["metrics"]["split"]) == {"train", "val", "test"}


def train_and_save_arrays_for_test(features: np.ndarray, ids: pd.DataFrame, label_names: list[str]):
    from modules.module3_text_features.train_text_logistic import build_pipeline

    model = build_pipeline(max_iter=1000)
    train_mask = ids["split"].eq("train").to_numpy()
    model.fit(features[train_mask], ids.loc[train_mask, "label_id"])
    assert label_names_from_map({"angry": 0, "happy": 1, "sad": 2}) == label_names
    return model
