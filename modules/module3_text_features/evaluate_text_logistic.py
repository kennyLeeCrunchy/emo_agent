"""
Evaluate a saved Module 3 text Logistic Regression model on another text feature set.

This is useful for checking distribution shift, for example a classifier trained
on gold dataset text and evaluated on Whisper ASR transcripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.module3_text_features.train_text_logistic import (
    CLASSIFICATION_REPORT_FILENAME,
    CONFIG_FILENAME,
    FIGURES_DIRNAME,
    METRICS_DIRNAME,
    METRICS_FILENAME,
    PREDICTIONS_FILENAME,
    classification_report_frame,
    evaluate_split,
    label_names_from_map,
    load_inputs,
    plot_confusion_matrix,
)


MODULE_OUTPUTS_DIR = PROJECT_ROOT / "modules" / "module3_text_features" / "outputs"
DEFAULT_MODEL_PATH = (
    MODULE_OUTPUTS_DIR
    / "text_unique"
    / "02_text_logistic_classifier"
    / "model"
    / "text_logistic_pipeline.joblib"
)
DEFAULT_FEATURE_PATH = (
    MODULE_OUTPUTS_DIR
    / "text_unique_asr"
    / "02_roberta_features_asr_text"
    / "text_embeddings.npy"
)
DEFAULT_IDS_PATH = (
    MODULE_OUTPUTS_DIR
    / "text_unique_asr"
    / "02_roberta_features_asr_text"
    / "text_ids.csv"
)
DEFAULT_LABEL_MAP_PATH = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs" / "label_map.json"
DEFAULT_OUTPUT_DIR = (
    MODULE_OUTPUTS_DIR
    / "text_unique_asr"
    / "03_text_logistic_classifier_gold_train_asr_eval"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved text Logistic Regression model.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--ids-path", type=Path, default=DEFAULT_IDS_PATH)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-note", default="gold_text_trained_model_evaluated_on_asr_text")
    return parser.parse_args()


def evaluate_and_save(
    model_path: Path,
    feature_path: Path,
    ids_path: Path,
    label_map_path: Path,
    output_dir: Path,
    source_note: str,
) -> dict[str, Any]:
    features, ids, label_map = load_inputs(feature_path, ids_path, label_map_path)
    label_names = label_names_from_map(label_map)
    model = joblib.load(model_path)

    metrics_dir = output_dir / METRICS_DIRNAME
    figures_dir = output_dir / FIGURES_DIRNAME
    for directory in [metrics_dir, figures_dir]:
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
            title=f"Text Logistic ASR Eval Confusion Matrix ({split_name})",
            output_path=figures_dir / f"confusion_matrix_{split_name}.png",
        )

    metrics_frame = pd.DataFrame(metrics_rows)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    report_frame = pd.concat(report_frames, ignore_index=True)

    metrics_frame.to_csv(metrics_dir / METRICS_FILENAME, index=False, encoding="utf-8-sig")
    predictions_frame.to_csv(metrics_dir / PREDICTIONS_FILENAME, index=False, encoding="utf-8-sig")
    report_frame.to_csv(metrics_dir / CLASSIFICATION_REPORT_FILENAME, index=False, encoding="utf-8-sig")

    config = {
        "model_path": str(model_path),
        "feature_path": str(feature_path),
        "ids_path": str(ids_path),
        "label_map_path": str(label_map_path),
        "feature_shape": list(features.shape),
        "split_counts": ids["split"].value_counts().to_dict(),
        "source_note": source_note,
        "label_map": label_map,
        "label_names": label_names,
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"metrics": metrics_frame, "output_dir": output_dir}


def main() -> int:
    args = parse_args()
    result = evaluate_and_save(
        model_path=args.model_path,
        feature_path=args.feature_path,
        ids_path=args.ids_path,
        label_map_path=args.label_map,
        output_dir=args.output_dir,
        source_note=args.source_note,
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
