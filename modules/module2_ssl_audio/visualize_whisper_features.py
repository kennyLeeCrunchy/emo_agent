"""
Visualize full-data Whisper Encoder embeddings before classifier training.

This script loads the 4160 full-data Whisper embeddings, checks alignment with
audio ids, and writes interpretable PCA and class-level figures for reporting.

在正式训练分类器之前，可视化全量 Whisper Encoder embedding。
脚本读取 4160 条全量特征，检查样本对齐，并输出适合汇报展示的 PCA 和类别层面图表。
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_OUTPUTS_DIR = PROJECT_ROOT / "modules" / "module2_ssl_audio" / "outputs"
DEFAULT_FEATURE_PATH = MODULE_OUTPUTS_DIR / "whisper_full" / "whisper" / "audio_whisper.npy"
DEFAULT_IDS_PATH = MODULE_OUTPUTS_DIR / "whisper_full" / "whisper" / "audio_ids.csv"
DEFAULT_OUTPUT_DIR = MODULE_OUTPUTS_DIR / "whisper_full" / "figures"
EMOTION_ORDER = ["angry", "fearful", "happy", "neutral", "playfulness", "sad", "surprise"]
SPLIT_COLORS = {"train": "#4c78a8", "val": "#f58518", "test": "#54a24b"}
EMOTION_COLORS = {
    "angry": "#d62728",
    "fearful": "#9467bd",
    "happy": "#ffbf00",
    "neutral": "#7f7f7f",
    "playfulness": "#2ca02c",
    "sad": "#1f77b4",
    "surprise": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize full Whisper embeddings.")
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--ids-path", type=Path, default=DEFAULT_IDS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-dims", type=int, default=40)
    return parser.parse_args()


def load_inputs(feature_path: Path, ids_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    features = np.load(feature_path).astype(np.float32)
    ids = pd.read_csv(ids_path)
    if features.shape[0] != len(ids):
        raise ValueError(f"Feature/id row mismatch: {features.shape[0]} != {len(ids)}")
    required = {"sample_id", "emotion", "label_id", "split"}
    missing = required.difference(ids.columns)
    if missing:
        raise ValueError(f"Missing id columns: {sorted(missing)}")
    return features, ids


def compute_pca_coordinates(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled)
    return coordinates, pca.explained_variance_ratio_


def save_pca_csv(coordinates: np.ndarray, ids: pd.DataFrame, explained: np.ndarray, output_dir: Path) -> Path:
    frame = ids[["sample_id", "emotion", "label_id", "split"]].copy()
    frame["pca_1"] = coordinates[:, 0]
    frame["pca_2"] = coordinates[:, 1]
    frame["pca1_explained_variance"] = float(explained[0])
    frame["pca2_explained_variance"] = float(explained[1])
    path = output_dir / "whisper_pca_coordinates.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def plot_pca_by_emotion(coordinates: np.ndarray, ids: pd.DataFrame, explained: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for emotion in EMOTION_ORDER:
        mask = ids["emotion"].eq(emotion).to_numpy()
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=18,
            alpha=0.68,
            color=EMOTION_COLORS.get(emotion),
            label=f"{emotion} ({mask.sum()})",
            edgecolors="none",
        )

    centroids = []
    for emotion in EMOTION_ORDER:
        mask = ids["emotion"].eq(emotion).to_numpy()
        center = coordinates[mask].mean(axis=0)
        centroids.append(center)
        ax.text(center[0], center[1], emotion, fontsize=10, weight="bold")
    centroids = np.vstack(centroids)
    ax.scatter(centroids[:, 0], centroids[:, 1], s=80, marker="x", color="black", linewidths=1.4)

    ax.set_title("Full Whisper Embeddings PCA by Emotion")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pca_by_split(coordinates: np.ndarray, ids: pd.DataFrame, explained: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for split in ["train", "val", "test"]:
        mask = ids["split"].eq(split).to_numpy()
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=16,
            alpha=0.58,
            color=SPLIT_COLORS[split],
            label=f"{split} ({mask.sum()})",
            edgecolors="none",
        )
    ax.set_title("Full Whisper Embeddings PCA by Split")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_class_means(features: np.ndarray, ids: pd.DataFrame, scale: bool = True) -> np.ndarray:
    values = StandardScaler().fit_transform(features) if scale else features
    return np.vstack([values[ids["emotion"].eq(emotion).to_numpy()].mean(axis=0) for emotion in EMOTION_ORDER])


def plot_centroid_similarity(class_means: np.ndarray, output_path: Path) -> None:
    similarity = cosine_similarity(class_means)
    fig, ax = plt.subplots(figsize=(7.3, 6.4))
    image = ax.imshow(similarity, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(EMOTION_ORDER)), EMOTION_ORDER, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(EMOTION_ORDER)), EMOTION_ORDER)
    ax.set_title("Whisper Class-Centroid Cosine Similarity")
    for row in range(similarity.shape[0]):
        for col in range(similarity.shape[1]):
            ax.text(col, row, f"{similarity[row, col]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_variable_dims_heatmap(features: np.ndarray, ids: pd.DataFrame, top_dims: int, output_path: Path) -> None:
    class_means = compute_class_means(features, ids)
    dim_variance = class_means.var(axis=0)
    selected_dims = np.argsort(dim_variance)[-top_dims:][::-1]
    selected = class_means[:, selected_dims]
    selected = StandardScaler().fit_transform(selected.T).T

    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    image = ax.imshow(selected, aspect="auto", cmap="coolwarm", vmin=-2.0, vmax=2.0)
    ax.set_yticks(np.arange(len(EMOTION_ORDER)), EMOTION_ORDER)
    ax.set_xticks(np.arange(len(selected_dims)), [str(dim) for dim in selected_dims], rotation=90, fontsize=7)
    ax.set_title(f"Top {top_dims} Class-Varying Whisper Dimensions")
    ax.set_xlabel("Embedding dimension index")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def visualize(feature_path: Path, ids_path: Path, output_dir: Path, top_dims: int) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features, ids = load_inputs(feature_path, ids_path)
    coordinates, explained = compute_pca_coordinates(features)
    class_means = compute_class_means(features, ids)

    paths = {
        "pca_coordinates": save_pca_csv(coordinates, ids, explained, output_dir),
        "pca_by_emotion": output_dir / "whisper_pca_by_emotion.png",
        "pca_by_split": output_dir / "whisper_pca_by_split.png",
        "centroid_similarity": output_dir / "whisper_centroid_cosine_similarity.png",
        "top_dims_heatmap": output_dir / "whisper_top_variable_dims_heatmap.png",
    }
    plot_pca_by_emotion(coordinates, ids, explained, paths["pca_by_emotion"])
    plot_pca_by_split(coordinates, ids, explained, paths["pca_by_split"])
    plot_centroid_similarity(class_means, paths["centroid_similarity"])
    plot_top_variable_dims_heatmap(features, ids, top_dims, paths["top_dims_heatmap"])
    return paths


def main() -> int:
    args = parse_args()
    paths = visualize(args.feature_path, args.ids_path, args.output_dir, args.top_dims)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
