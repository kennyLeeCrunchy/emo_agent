"""
Compare MFCC, HuBERT, and Whisper features with the same Logistic Regression reader.

The script keeps the sample order and train/validation/test split fixed, evaluates
raw and PCA64 variants for high-dimensional SSL embeddings, and writes metrics,
predictions, and a compact bar chart for reporting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MFCC_DIR = PROJECT_ROOT / "modules" / "module2_audio_features" / "outputs" / "mfcc"
DEFAULT_SSL_DIR = PROJECT_ROOT / "modules" / "module2_ssl_audio" / "outputs"
DEFAULT_OUTPUT_DIR = DEFAULT_SSL_DIR / "comparison"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight MFCC/HuBERT/Whisper comparison.")
    parser.add_argument("--mfcc-dir", type=Path, default=DEFAULT_MFCC_DIR)
    parser.add_argument("--ssl-dir", type=Path, default=DEFAULT_SSL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pca-components", type=int, default=64)
    return parser.parse_args()


def validate_matching_ids(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    if len(reference) != len(candidate):
        raise ValueError(f"Feature id row count mismatch: {len(reference)} != {len(candidate)}")
    for column in ["sample_id", "emotion", "label_id", "split"]:
        if not reference[column].astype(str).equals(candidate[column].astype(str)):
            raise ValueError(f"Feature id order mismatch in column: {column}")


def load_feature_set(feature_name: str, feature_path: Path, ids_path: Path) -> tuple[str, np.ndarray, pd.DataFrame]:
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing ids file: {ids_path}")
    features = np.load(feature_path).astype(np.float32)
    ids = pd.read_csv(ids_path)
    if len(ids) != features.shape[0]:
        raise ValueError(f"{feature_name} ids/features row mismatch: {len(ids)} != {features.shape[0]}")
    return feature_name, features, ids


def build_logistic_pipeline(
    train_features: np.ndarray,
    pca_components: int | None,
) -> Pipeline:
    steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
    if pca_components is not None:
        usable_components = min(pca_components, train_features.shape[0] - 1, train_features.shape[1])
        if usable_components >= 1:
            steps.append(
                (
                    "pca",
                    PCA(n_components=usable_components, random_state=RANDOM_STATE),
                )
            )
    steps.append(
        (
            "logistic",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        )
    )
    return Pipeline(steps)


def evaluate_feature_set(
    feature_name: str,
    features: np.ndarray,
    ids: pd.DataFrame,
    pca_components: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = ids["split"].eq("train").to_numpy()
    eval_mask = ids["split"].isin(["val", "test"]).to_numpy()
    if not train_mask.any() or not eval_mask.any():
        raise ValueError("Expected train plus val/test rows in ids.")

    classifier = build_logistic_pipeline(features[train_mask], pca_components=pca_components)
    classifier.fit(features[train_mask], ids.loc[train_mask, "label_id"])

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    variant = "raw" if pca_components is None else f"pca{pca_components}"

    for split_name in ["val", "test"]:
        split_mask = ids["split"].eq(split_name).to_numpy()
        y_true = ids.loc[split_mask, "label_id"].to_numpy()
        y_pred = classifier.predict(features[split_mask])
        metric_rows.append(
            {
                "feature_set": feature_name,
                "variant": variant,
                "pca_components": pca_components if pca_components is not None else 0,
                "eval_split": split_name,
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "n_samples": int(split_mask.sum()),
                "feature_dim": int(features.shape[1]),
            }
        )

        predictions = ids.loc[split_mask, ["sample_id", "emotion", "label_id", "split"]].copy()
        predictions["feature_set"] = feature_name
        predictions["variant"] = variant
        predictions["pred_label_id"] = y_pred
        predictions["correct"] = predictions["label_id"].to_numpy() == y_pred
        prediction_rows.append(predictions)

    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def select_ssl_recommendation(metrics: pd.DataFrame) -> pd.DataFrame:
    ssl_metrics = metrics[metrics["feature_set"].isin(["hubert", "whisper"])].copy()
    val_metrics = ssl_metrics[ssl_metrics["eval_split"].eq("val")].copy()
    test_metrics = ssl_metrics[ssl_metrics["eval_split"].eq("test")].copy()

    selected_rows = []
    for feature_set, group in val_metrics.groupby("feature_set", sort=True):
        best = group.sort_values(
            ["accuracy", "macro_f1", "balanced_accuracy"],
            ascending=False,
        ).iloc[0]
        paired_test = test_metrics[
            test_metrics["feature_set"].eq(feature_set)
            & test_metrics["variant"].eq(best["variant"])
        ].iloc[0]
        selected_rows.append(
            {
                "feature_set": feature_set,
                "selected_variant": best["variant"],
                "val_accuracy": best["accuracy"],
                "val_macro_f1": best["macro_f1"],
                "test_accuracy": paired_test["accuracy"],
                "test_macro_f1": paired_test["macro_f1"],
                "feature_dim": best["feature_dim"],
            }
        )

    summary = pd.DataFrame(selected_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["val_accuracy", "val_macro_f1", "test_accuracy"],
            ascending=False,
        ).reset_index(drop=True)
    return summary


def plot_metric_bars(metrics: pd.DataFrame, output_path: Path) -> None:
    test_metrics = metrics[metrics["eval_split"].eq("test")].copy()
    test_metrics["name"] = test_metrics["feature_set"] + "-" + test_metrics["variant"]
    test_metrics = test_metrics.sort_values("accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#4c78a8" if name.startswith("mfcc") else "#f58518" if name.startswith("hubert") else "#54a24b" for name in test_metrics["name"]]
    ax.bar(test_metrics["name"], test_metrics["accuracy"], color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Lightweight Audio Feature Comparison")
    ax.tick_params(axis="x", labelrotation=25)
    for index, value in enumerate(test_metrics["accuracy"]):
        ax.text(index, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_comparison(
    mfcc_dir: Path,
    ssl_dir: Path,
    output_dir: Path,
    pca_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_sets = [
        load_feature_set("mfcc", mfcc_dir / "audio_mfcc.npy", mfcc_dir / "audio_ids.csv"),
        load_feature_set("hubert", ssl_dir / "hubert" / "audio_hubert.npy", ssl_dir / "hubert" / "audio_ids.csv"),
        load_feature_set("whisper", ssl_dir / "whisper" / "audio_whisper.npy", ssl_dir / "whisper" / "audio_ids.csv"),
    ]

    reference_ids = feature_sets[0][2]
    for feature_name, _, ids in feature_sets[1:]:
        try:
            validate_matching_ids(reference_ids, ids)
        except ValueError as exc:
            raise ValueError(f"{feature_name}: {exc}") from exc

    metrics_parts: list[pd.DataFrame] = []
    predictions_parts: list[pd.DataFrame] = []
    for feature_name, features, ids in feature_sets:
        variants = [None] if feature_name == "mfcc" else [None, pca_components]
        for variant_pca in variants:
            metrics, predictions = evaluate_feature_set(
                feature_name=feature_name,
                features=features,
                ids=ids,
                pca_components=variant_pca,
            )
            metrics_parts.append(metrics)
            predictions_parts.append(predictions)

    all_metrics = pd.concat(metrics_parts, ignore_index=True)
    all_predictions = pd.concat(predictions_parts, ignore_index=True)
    recommendation = select_ssl_recommendation(all_metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    all_predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    recommendation.to_csv(output_dir / "ssl_recommendation.csv", index=False, encoding="utf-8-sig")
    plot_metric_bars(all_metrics, output_dir / "test_accuracy_comparison.png")
    return all_metrics, all_predictions, recommendation


def main() -> int:
    args = parse_args()
    metrics, _, recommendation = run_comparison(
        mfcc_dir=args.mfcc_dir,
        ssl_dir=args.ssl_dir,
        output_dir=args.output_dir,
        pca_components=args.pca_components,
    )
    print(metrics.to_string(index=False))
    print("\nSSL recommendation by validation performance:")
    print(recommendation.to_string(index=False))
    print(f"\nWrote outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
