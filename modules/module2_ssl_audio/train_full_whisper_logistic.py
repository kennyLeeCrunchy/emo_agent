"""
Train the formal full-data Whisper raw embedding + Logistic Regression model.

This script uses the 4160 full-data Whisper Encoder embeddings without PCA,
consumes the split from the provided ids file, and saves the classifier, scaler,
metrics, confusion matrix, classification report, and per-sample probabilities.

训练正式的全量 Whisper raw embedding + Logistic Regression 音频情绪模型。
脚本不使用 PCA，直接使用 384 维 Whisper Encoder embedding，并保存模型、指标、
混淆矩阵、分类报告和每条样本的类别概率。
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
MODULE_OUTPUTS_DIR = PROJECT_ROOT / "modules" / "module2_ssl_audio" / "outputs"
DEFAULT_FEATURE_PATH = MODULE_OUTPUTS_DIR / "whisper_full" / "whisper" / "audio_whisper.npy"
DEFAULT_IDS_PATH = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs_disjoint_split" / "01_full_samples" / "metadata.csv"
DEFAULT_LABEL_MAP_PATH = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs" / "label_map.json"
DEFAULT_OUTPUT_DIR = MODULE_OUTPUTS_DIR / "whisper_text_disjoint" / "audio_logistic"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full Whisper + Logistic Regression model.")
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--ids-path", type=Path, default=DEFAULT_IDS_PATH)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-iter", type=int, default=5000)
    return parser.parse_args()


def load_inputs(feature_path: Path, ids_path: Path, label_map_path: Path) -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    features = np.load(feature_path).astype(np.float32)
    ids = pd.read_csv(ids_path)
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))

    required = {"sample_id", "emotion", "label_id", "split"}
    missing = required.difference(ids.columns)
    if missing:
        raise ValueError(f"Missing id columns: {sorted(missing)}")
    if len(ids) != features.shape[0]:
        raise ValueError(f"Feature/id row mismatch: {features.shape[0]} != {len(ids)}")
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains NaN or Inf values.")
    return features, ids, {str(key): int(value) for key, value in label_map.items()}


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


def label_names_from_map(label_map: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]


def evaluate_split(
    model: Pipeline,
    features: np.ndarray,
    ids: pd.DataFrame,
    split_name: str,
    label_names: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    mask = ids["split"].eq(split_name).to_numpy()
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

    predictions = ids.loc[mask, ["sample_id", "emotion", "label_id", "split"]].copy()
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


def train_and_save(
    feature_path: Path,
    ids_path: Path,
    label_map_path: Path,
    output_dir: Path,
    max_iter: int,
) -> dict[str, Any]:
    features, ids, label_map = load_inputs(feature_path, ids_path, label_map_path)
    label_names = label_names_from_map(label_map)
    train_mask = ids["split"].eq("train").to_numpy()
    if train_mask.sum() == 0:
        raise ValueError("No train rows found.")

    model = build_pipeline(max_iter=max_iter)
    model.fit(features[train_mask], ids.loc[train_mask, "label_id"])

    model_dir = output_dir / "model"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
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
            title=f"Whisper Logistic Confusion Matrix ({split_name})",
            output_path=figures_dir / f"confusion_matrix_{split_name}.png",
        )

    metrics_frame = pd.DataFrame(metrics_rows)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    report_frame = pd.concat(report_frames, ignore_index=True)

    metrics_frame.to_csv(metrics_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions_frame.to_csv(metrics_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    report_frame.to_csv(metrics_dir / "classification_report.csv", index=False, encoding="utf-8-sig")
    joblib.dump(model, model_dir / "whisper_logistic_pipeline.joblib")

    config = {
        "feature_path": str(feature_path),
        "ids_path": str(ids_path),
        "label_map_path": str(label_map_path),
        "feature_shape": list(features.shape),
        "label_map": label_map,
        "label_names": label_names,
        "split_counts": ids["split"].value_counts().to_dict(),
        "model": "StandardScaler + LogisticRegression",
        "uses_pca": False,
        "random_state": RANDOM_STATE,
        "max_iter": max_iter,
        "class_weight": "balanced",
    }
    (model_dir / "config.json").write_text(
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
        feature_path=args.feature_path,
        ids_path=args.ids_path,
        label_map_path=args.label_map,
        output_dir=args.output_dir,
        max_iter=args.max_iter,
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
