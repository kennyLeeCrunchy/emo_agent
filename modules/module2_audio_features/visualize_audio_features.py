"""
生成 MFCC/传统声学特征的汇报用可视化图表。
Create report-friendly visualizations for MFCC and traditional acoustic features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from modules.module2_audio_features.mfcc_features import (
    DEFAULT_PARQUET_DIR,
    FEATURE_NAMES,
    frame_signal,
    mel_filterbank,
    read_audio_column,
    read_waveform,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "modules" / "module2_audio_features" / "outputs" / "mfcc"
DEFAULT_FEATURES = DEFAULT_FEATURE_DIR / "audio_mfcc.npy"
DEFAULT_IDS = DEFAULT_FEATURE_DIR / "audio_ids.csv"
DEFAULT_FEATURE_NAMES = DEFAULT_FEATURE_DIR / "audio_feature_names.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "modules" / "module2_audio_features" / "figures"
DEFAULT_SUMMARY = DEFAULT_FEATURE_DIR / "audio_feature_summary_by_emotion.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize audio feature outputs.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--feature-names", type=Path, default=DEFAULT_FEATURE_NAMES)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_feature_names(path: Path) -> list[str]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return FEATURE_NAMES


def plot_pca(features: np.ndarray, ids: pd.DataFrame, figures_dir: Path) -> Path:
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    for emotion in sorted(ids["emotion"].unique()):
        mask = ids["emotion"] == emotion
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=26,
            alpha=0.78,
            label=emotion,
            edgecolors="none",
        )
    ax.set_title("Audio Feature PCA Preview")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path = figures_dir / "audio_feature_pca_preview.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_heatmap(
    features: np.ndarray,
    ids: pd.DataFrame,
    feature_names: list[str],
    figures_dir: Path,
    summary_path: Path,
) -> tuple[Path, Path]:
    scaled = StandardScaler().fit_transform(features)
    scaled_df = pd.DataFrame(scaled, columns=feature_names)
    scaled_df["emotion"] = ids["emotion"].values

    mean_by_emotion = scaled_df.groupby("emotion").mean().sort_index()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    mean_by_emotion.to_csv(summary_path, encoding="utf-8-sig")

    emotion_variance = mean_by_emotion.var(axis=0).sort_values(ascending=False)
    selected_features = emotion_variance.head(14).index.tolist()
    heatmap_values = mean_by_emotion[selected_features].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
    image = ax.imshow(heatmap_values, aspect="auto", cmap="RdBu_r", vmin=-2.2, vmax=2.2)
    ax.set_title("Most Emotion-Separating Audio Features")
    ax.set_yticks(range(len(mean_by_emotion.index)))
    ax.set_yticklabels(mean_by_emotion.index)
    ax.set_xticks(range(len(selected_features)))
    ax.set_xticklabels(selected_features, rotation=45, ha="right", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Standardized class mean")
    fig.tight_layout()

    output_path = figures_dir / "audio_feature_class_heatmap_preview.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path, summary_path


def compute_log_mel_and_mfcc(waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    emphasized = np.append(waveform[:1], waveform[1:] - 0.97 * waveform[:-1])
    frames = frame_signal(emphasized, sample_rate)
    windowed = frames * np.hamming(frames.shape[1]).astype(np.float32)
    fft_size = 1 << int(np.ceil(np.log2(frames.shape[1])))
    power = (np.abs(np.fft.rfft(windowed, n=fft_size, axis=1)) ** 2) / float(fft_size)
    filters = mel_filterbank(sample_rate, fft_size)
    log_mel = np.log(np.maximum(power @ filters.T, 1e-10))

    from scipy.fftpack import dct

    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :13]
    return log_mel.T, mfcc.T


def plot_extraction_example(ids: pd.DataFrame, parquet_dir: Path, figures_dir: Path) -> Path:
    first = ids.iloc[0]
    audio_column = read_audio_column(parquet_dir, str(first["parquet_file"]))
    waveform, sample_rate = read_waveform(audio_column.iloc[int(first["row_id"])])
    duration = waveform.size / sample_rate
    time = np.linspace(0, duration, waveform.size, endpoint=False)
    log_mel, mfcc = compute_log_mel_and_mfcc(waveform, sample_rate)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), dpi=160)
    axes[0].plot(time, waveform, color="#2f6f9f", linewidth=0.8)
    axes[0].set_title(f"Example waveform: {first['emotion']} / {first['sample_id']}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.2)

    mel_image = axes[1].imshow(log_mel, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Log-Mel Energy")
    axes[1].set_ylabel("Mel filter")
    fig.colorbar(mel_image, ax=axes[1], fraction=0.025, pad=0.02)

    mfcc_image = axes[2].imshow(mfcc, aspect="auto", origin="lower", cmap="viridis")
    axes[2].set_title("MFCC Coefficients Over Time")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("MFCC index")
    fig.colorbar(mfcc_image, ax=axes[2], fraction=0.025, pad=0.02)
    fig.tight_layout()

    output_path = figures_dir / "audio_feature_extraction_example.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    features = np.load(args.features)
    ids = pd.read_csv(args.ids)
    feature_names = load_feature_names(args.feature_names)

    pca_path = plot_pca(features, ids, args.figures_dir)
    heatmap_path, summary_path = plot_heatmap(
        features, ids, feature_names, args.figures_dir, args.summary
    )
    example_path = plot_extraction_example(ids, args.parquet_dir, args.figures_dir)

    print(f"Loaded features: {features.shape}")
    print(f"Wrote PCA figure: {pca_path}")
    print(f"Wrote heatmap figure: {heatmap_path}")
    print(f"Wrote extraction example: {example_path}")
    print(f"Wrote emotion feature summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
