"""
Train the Module 4 Whisper+BERT/RoBERTa early-fusion Logistic Regression model.

This script consumes cached Whisper audio embeddings and cached Chinese
BERT/RoBERTa text embeddings, validates strict sample alignment, concatenates
audio then text features, and saves model artifacts, metrics, confusion
matrices, predictions, baseline comparisons, and modality-disagreement rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE2_OUTPUTS_DIR = PROJECT_ROOT / "modules" / "module2_ssl_audio" / "outputs"
MODULE3_OUTPUTS_DIR = PROJECT_ROOT / "modules" / "module3_text_features" / "outputs"
MODULE4_OUTPUTS_DIR = PROJECT_ROOT / "modules" / "module4_fusion" / "outputs"
DEFAULT_AUDIO_FEATURE_PATH = MODULE2_OUTPUTS_DIR / "whisper_full" / "whisper" / "audio_whisper.npy"
DEFAULT_AUDIO_IDS_PATH = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs_disjoint_split" / "01_full_samples" / "metadata.csv"
DEFAULT_TEXT_FEATURE_PATH = MODULE3_OUTPUTS_DIR / "text_full" / "01_roberta_features_full_samples" / "text_embeddings.npy"
DEFAULT_TEXT_IDS_PATH = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs_disjoint_split" / "01_full_samples" / "metadata.csv"
DEFAULT_LABEL_MAP_PATH = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs" / "label_map.json"
DEFAULT_OUTPUT_DIR = MODULE4_OUTPUTS_DIR / "fusion_text_disjoint" / "01_fusion_logistic_classifier"
DEFAULT_AUDIO_METRICS_PATH = MODULE2_OUTPUTS_DIR / "whisper_text_disjoint" / "audio_logistic" / "metrics" / "metrics.csv"
DEFAULT_TEXT_METRICS_PATH = MODULE3_OUTPUTS_DIR / "text_unique" / "02_text_logistic_classifier" / "metrics" / "metrics.csv"
DEFAULT_AUDIO_PREDICTIONS_PATH = MODULE2_OUTPUTS_DIR / "whisper_text_disjoint" / "audio_logistic" / "metrics" / "predictions.csv"
DEFAULT_TEXT_PREDICTIONS_PATH = MODULE3_OUTPUTS_DIR / "text_unique" / "02_text_logistic_classifier" / "metrics" / "predictions.csv"
RANDOM_STATE = 42

MODEL_DIRNAME = "model"
METRICS_DIRNAME = "metrics"
FIGURES_DIRNAME = "figures"
PIPELINE_FILENAME = "fusion_logistic_pipeline.joblib"
CONFIG_FILENAME = "config.json"
METRICS_FILENAME = "metrics.csv"
CLASSIFICATION_REPORT_FILENAME = "classification_report.csv"
PREDICTIONS_FILENAME = "predictions.csv"
MODEL_COMPARISON_FILENAME = "model_comparison.csv"
MODALITY_DISAGREEMENTS_FILENAME = "modality_disagreements.csv"

DISAGREEMENT_COLUMNS = [
    "sample_id",
    "split",
    "text",
    "emotion",
    "label_id",
    "audio_pred_emotion",
    "audio_confidence",
    "text_pred_emotion",
    "text_confidence",
    "fusion_pred_emotion",
    "fusion_confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Whisper+BERT/RoBERTa early-fusion Logistic Regression model.")
    parser.add_argument("--audio-feature-path", type=Path, default=DEFAULT_AUDIO_FEATURE_PATH)
    parser.add_argument("--audio-ids-path", type=Path, default=DEFAULT_AUDIO_IDS_PATH)
    parser.add_argument("--text-feature-path", type=Path, default=DEFAULT_TEXT_FEATURE_PATH)
    parser.add_argument("--text-ids-path", type=Path, default=DEFAULT_TEXT_IDS_PATH)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audio-metrics-path", type=Path, default=DEFAULT_AUDIO_METRICS_PATH)
    parser.add_argument("--text-metrics-path", type=Path, default=DEFAULT_TEXT_METRICS_PATH)
    parser.add_argument("--audio-predictions-path", type=Path, default=DEFAULT_AUDIO_PREDICTIONS_PATH)
    parser.add_argument("--text-predictions-path", type=Path, default=DEFAULT_TEXT_PREDICTIONS_PATH)
    parser.add_argument("--max-iter", type=int, default=5000)
    return parser.parse_args()


def validate_modal_alignment(audio_ids: pd.DataFrame, text_ids: pd.DataFrame) -> None:
    if len(audio_ids) != len(text_ids):
        raise ValueError(f"Audio/text id row count mismatch: {len(audio_ids)} != {len(text_ids)}")
    required_audio = {"sample_id", "emotion", "label_id", "split"}
    required_text = {"sample_id", "text", "emotion", "label_id", "split"}
    missing_audio = required_audio.difference(audio_ids.columns)
    missing_text = required_text.difference(text_ids.columns)
    if missing_audio:
        raise ValueError(f"Missing audio id columns: {sorted(missing_audio)}")
    if missing_text:
        raise ValueError(f"Missing text id columns: {sorted(missing_text)}")
    for column in ["sample_id", "emotion", "label_id", "split"]:
        if not audio_ids[column].astype(str).equals(text_ids[column].astype(str)):
            raise ValueError(f"Audio/text id order mismatch in column: {column}")
    if audio_ids["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values found in audio ids.")


def build_fusion_features(audio_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
    if audio_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("Audio and text features must both be 2D arrays.")
    if audio_features.shape[0] != text_features.shape[0]:
        raise ValueError(f"Audio/text feature row mismatch: {audio_features.shape[0]} != {text_features.shape[0]}")
    if not np.isfinite(audio_features).all():
        raise ValueError("Audio feature matrix contains NaN or Inf values.")
    if not np.isfinite(text_features).all():
        raise ValueError("Text feature matrix contains NaN or Inf values.")
    return np.concatenate([audio_features.astype(np.float32), text_features.astype(np.float32)], axis=1).astype(np.float32)


def label_names_from_map(label_map: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]


def load_fusion_inputs(
    audio_feature_path: Path,
    audio_ids_path: Path,
    text_feature_path: Path,
    text_ids_path: Path,
    label_map_path: Path,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, int], dict[str, int]]:
    audio_features = np.load(audio_feature_path).astype(np.float32)
    text_features = np.load(text_feature_path).astype(np.float32)
    audio_ids = pd.read_csv(audio_ids_path)
    text_ids = pd.read_csv(text_ids_path)
    label_map = {str(key): int(value) for key, value in json.loads(label_map_path.read_text(encoding="utf-8")).items()}

    if len(audio_ids) != audio_features.shape[0]:
        raise ValueError(f"Audio feature/id row mismatch: {audio_features.shape[0]} != {len(audio_ids)}")
    if len(text_ids) != text_features.shape[0]:
        raise ValueError(f"Text feature/id row mismatch: {text_features.shape[0]} != {len(text_ids)}")
    validate_modal_alignment(audio_ids, text_ids)
    fusion_features = build_fusion_features(audio_features, text_features)
    ids = text_ids.copy()
    feature_info = {
        "audio_dim": int(audio_features.shape[1]),
        "text_dim": int(text_features.shape[1]),
        "fusion_dim": int(fusion_features.shape[1]),
    }
    return fusion_features, ids, label_map, feature_info


def build_pipeline(max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_split(
    model: Pipeline,
    features: np.ndarray,
    ids: pd.DataFrame,
    split_name: str,
    label_names: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    mask = ids["split"].eq(split_name).to_numpy()
    if mask.sum() == 0:
        raise ValueError(f"No rows found for split: {split_name}")
    y_true = ids.loc[mask, "label_id"].to_numpy()
    y_pred = model.predict(features[mask])
    probabilities = model.predict_proba(features[mask])

    metrics = {
        "split": split_name,
        "n_samples": int(mask.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    predictions = ids.loc[mask, ["sample_id", "text", "emotion", "label_id", "split"]].copy()
    predictions["pred_label_id"] = y_pred
    predictions["pred_emotion"] = [label_names[int(label_id)] for label_id in y_pred]
    predictions["correct"] = predictions["label_id"].to_numpy() == y_pred
    predictions["confidence"] = probabilities.max(axis=1)
    for label_id, label_name in enumerate(label_names):
        predictions[f"prob_{label_name}"] = probabilities[:, label_id]

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    return metrics, predictions, matrix


def classification_report_frame(
    model: Pipeline,
    features: np.ndarray,
    ids: pd.DataFrame,
    split_name: str,
    label_names: list[str],
) -> pd.DataFrame:
    mask = ids["split"].eq(split_name).to_numpy()
    y_true = ids.loc[mask, "label_id"].to_numpy()
    y_pred = model.predict(features[mask])
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    frame = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    frame.insert(0, "split", split_name)
    return frame


def plot_confusion_matrix(matrix: np.ndarray, label_names: list[str], title: str, output_path: Path) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(label_names)), label_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(label_names)), label_names)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                f"{matrix[row, col]}\n{normalized[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if normalized[row, col] > 0.5 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_model_comparison(audio_metrics: pd.DataFrame, text_metrics: pd.DataFrame, fusion_metrics: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for model_name, metrics in [
        ("audio_whisper", audio_metrics),
        ("text_roberta", text_metrics),
        ("fusion_whisper_roberta", fusion_metrics),
    ]:
        frame = metrics.copy()
        frame.insert(0, "model", model_name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_modality_disagreements(
    audio_predictions: pd.DataFrame,
    text_predictions: pd.DataFrame,
    fusion_predictions: pd.DataFrame,
) -> pd.DataFrame:
    audio_cols = ["sample_id", "split", "pred_emotion", "confidence"]
    text_cols = ["sample_id", "text", "emotion", "label_id", "split", "pred_emotion", "confidence"]
    fusion_cols = ["sample_id", "split", "pred_emotion", "confidence"]

    merged = text_predictions[text_cols].merge(
        audio_predictions[audio_cols].rename(
            columns={"pred_emotion": "audio_pred_emotion", "confidence": "audio_confidence", "split": "audio_split"}
        ),
        on="sample_id",
        how="inner",
    )
    merged = merged.merge(
        fusion_predictions[fusion_cols].rename(
            columns={"pred_emotion": "fusion_pred_emotion", "confidence": "fusion_confidence", "split": "fusion_split"}
        ),
        on="sample_id",
        how="inner",
    )
    merged = merged.rename(columns={"pred_emotion": "text_pred_emotion", "confidence": "text_confidence"})
    disagreement = merged[merged["audio_pred_emotion"].ne(merged["text_pred_emotion"])].copy()
    if disagreement.empty:
        return pd.DataFrame(columns=DISAGREEMENT_COLUMNS)
    return disagreement[DISAGREEMENT_COLUMNS].reset_index(drop=True)


def maybe_read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def write_optional_comparison_files(
    metrics_dir: Path,
    fusion_metrics: pd.DataFrame,
    fusion_predictions: pd.DataFrame,
    audio_metrics_path: Path,
    text_metrics_path: Path,
    audio_predictions_path: Path,
    text_predictions_path: Path,
) -> dict[str, str]:
    comparison_path = metrics_dir / MODEL_COMPARISON_FILENAME
    disagreement_path = metrics_dir / MODALITY_DISAGREEMENTS_FILENAME

    audio_metrics = maybe_read_csv(audio_metrics_path)
    text_metrics = maybe_read_csv(text_metrics_path)
    if audio_metrics is not None and text_metrics is not None:
        comparison = build_model_comparison(audio_metrics, text_metrics, fusion_metrics)
    else:
        comparison = fusion_metrics.copy()
        comparison.insert(0, "model", "fusion_whisper_roberta")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    audio_predictions = maybe_read_csv(audio_predictions_path)
    text_predictions = maybe_read_csv(text_predictions_path)
    if audio_predictions is not None and text_predictions is not None:
        disagreements = build_modality_disagreements(audio_predictions, text_predictions, fusion_predictions)
    else:
        disagreements = pd.DataFrame(columns=DISAGREEMENT_COLUMNS)
    disagreements.to_csv(disagreement_path, index=False, encoding="utf-8-sig")

    return {
        "model_comparison_path": str(comparison_path),
        "modality_disagreements_path": str(disagreement_path),
    }


def train_and_save(
    audio_feature_path: Path,
    audio_ids_path: Path,
    text_feature_path: Path,
    text_ids_path: Path,
    label_map_path: Path,
    output_dir: Path,
    max_iter: int,
    audio_metrics_path: Path = DEFAULT_AUDIO_METRICS_PATH,
    text_metrics_path: Path = DEFAULT_TEXT_METRICS_PATH,
    audio_predictions_path: Path = DEFAULT_AUDIO_PREDICTIONS_PATH,
    text_predictions_path: Path = DEFAULT_TEXT_PREDICTIONS_PATH,
) -> dict[str, Any]:
    features, ids, label_map, feature_info = load_fusion_inputs(
        audio_feature_path=audio_feature_path,
        audio_ids_path=audio_ids_path,
        text_feature_path=text_feature_path,
        text_ids_path=text_ids_path,
        label_map_path=label_map_path,
    )
    label_names = label_names_from_map(label_map)
    train_mask = ids["split"].eq("train").to_numpy()
    if train_mask.sum() == 0:
        raise ValueError("No train rows found.")

    model = build_pipeline(max_iter=max_iter)
    model.fit(features[train_mask], ids.loc[train_mask, "label_id"])

    model_dir = output_dir / MODEL_DIRNAME
    metrics_dir = output_dir / METRICS_DIRNAME
    figures_dir = output_dir / FIGURES_DIRNAME
    for directory in [model_dir, metrics_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    report_frames: list[pd.DataFrame] = []
    for split_name in ["train", "val", "test"]:
        metrics, predictions, matrix = evaluate_split(model, features, ids, split_name, label_names)
        metrics_rows.append(metrics)
        prediction_frames.append(predictions)
        report_frames.append(classification_report_frame(model, features, ids, split_name, label_names))
        pd.DataFrame(matrix, index=label_names, columns=label_names).to_csv(
            metrics_dir / f"confusion_matrix_{split_name}.csv",
            encoding="utf-8-sig",
        )
        plot_confusion_matrix(
            matrix,
            label_names,
            title=f"Fusion Logistic Confusion Matrix ({split_name})",
            output_path=figures_dir / f"confusion_matrix_{split_name}.png",
        )

    metrics_frame = pd.DataFrame(metrics_rows)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    report_frame = pd.concat(report_frames, ignore_index=True)

    metrics_frame.to_csv(metrics_dir / METRICS_FILENAME, index=False, encoding="utf-8-sig")
    predictions_frame.to_csv(metrics_dir / PREDICTIONS_FILENAME, index=False, encoding="utf-8-sig")
    report_frame.to_csv(metrics_dir / CLASSIFICATION_REPORT_FILENAME, index=False, encoding="utf-8-sig")
    extra_paths = write_optional_comparison_files(
        metrics_dir=metrics_dir,
        fusion_metrics=metrics_frame,
        fusion_predictions=predictions_frame,
        audio_metrics_path=audio_metrics_path,
        text_metrics_path=text_metrics_path,
        audio_predictions_path=audio_predictions_path,
        text_predictions_path=text_predictions_path,
    )
    joblib.dump(model, model_dir / PIPELINE_FILENAME)

    config = {
        "audio_feature_path": str(audio_feature_path),
        "audio_ids_path": str(audio_ids_path),
        "text_feature_path": str(text_feature_path),
        "text_ids_path": str(text_ids_path),
        "label_map_path": str(label_map_path),
        "feature_shape": list(features.shape),
        "feature_info": feature_info,
        "label_map": label_map,
        "label_names": label_names,
        "split_counts": ids["split"].value_counts().to_dict(),
        "model": "StandardScaler + LogisticRegression",
        "feature_source": "Whisper audio embedding + Chinese BERT/RoBERTa text embedding",
        "fusion_method": "early_concat_audio_then_text",
        "random_state": RANDOM_STATE,
        "max_iter": max_iter,
        "class_weight": "balanced",
        "audio_metrics_path": str(audio_metrics_path),
        "text_metrics_path": str(text_metrics_path),
        "audio_predictions_path": str(audio_predictions_path),
        "text_predictions_path": str(text_predictions_path),
        **extra_paths,
    }
    (model_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "metrics": metrics_frame,
        "predictions": predictions_frame,
        "report": report_frame,
        "config": config,
        "output_dir": output_dir,
    }


def main() -> int:
    args = parse_args()
    result = train_and_save(
        audio_feature_path=args.audio_feature_path,
        audio_ids_path=args.audio_ids_path,
        text_feature_path=args.text_feature_path,
        text_ids_path=args.text_ids_path,
        label_map_path=args.label_map,
        output_dir=args.output_dir,
        max_iter=args.max_iter,
        audio_metrics_path=args.audio_metrics_path,
        text_metrics_path=args.text_metrics_path,
        audio_predictions_path=args.audio_predictions_path,
        text_predictions_path=args.text_predictions_path,
    )
    print(result["metrics"].to_string(index=False))
    print(f"\nWrote outputs to: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
