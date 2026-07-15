"""Build multimodal sliding-window emotion curves.

This module keeps the core curve logic independent from heavyweight Whisper and
RoBERTa model loading so it can be tested with toy arrays. Runtime scripts can
provide window-level audio embeddings, text embeddings, and the trained fusion
pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.module4_fusion.train_fusion_logistic import build_fusion_features


@dataclass(frozen=True)
class AudioWindow:
    index: int
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    waveform: np.ndarray


@dataclass(frozen=True)
class CurveConfig:
    window_seconds: float = 2.0
    step_seconds: float = 1.0
    text_mode: str = "repeat_global"
    high_emotion_threshold: float = 0.6
    change_delta: float = 0.25


def iter_sliding_windows(
    waveform: np.ndarray,
    sample_rate: int,
    window_seconds: float = 2.0,
    step_seconds: float = 1.0,
) -> Iterable[AudioWindow]:
    """Yield fixed-duration sliding windows and keep a final tail window."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive.")

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise ValueError("waveform must not be empty.")

    total_samples = int(waveform.shape[0])
    window_size = max(1, int(round(window_seconds * sample_rate)))
    step_size = max(1, int(round(step_seconds * sample_rate)))

    if total_samples <= window_size:
        starts = [0]
    else:
        starts = list(range(0, total_samples - window_size + 1, step_size))
        tail_start = total_samples - window_size
        uncovered_tail = total_samples - (starts[-1] + window_size)
        if starts[-1] != tail_start and uncovered_tail >= step_size / 2:
            starts.append(tail_start)

    for index, start in enumerate(starts):
        end = min(start + window_size, total_samples)
        yield AudioWindow(
            index=index,
            start_sample=start,
            end_sample=end,
            start_time=round(start / sample_rate, 6),
            end_time=round(end / sample_rate, 6),
            waveform=waveform[start:end].copy(),
        )


def build_window_fusion_features(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    text_mode: str = "repeat_global",
) -> np.ndarray:
    """Concatenate window audio embeddings with text embeddings."""
    audio = np.asarray(audio_embeddings, dtype=np.float32)
    text = np.asarray(text_embeddings, dtype=np.float32)
    if audio.ndim != 2:
        raise ValueError("audio_embeddings must be a 2D array.")
    if text.ndim != 2:
        raise ValueError("text_embeddings must be a 2D array.")
    if text_mode == "repeat_global":
        if text.shape[0] != 1:
            raise ValueError("repeat_global text_mode expects exactly one text embedding row.")
        text = np.repeat(text, audio.shape[0], axis=0)
    elif text_mode == "per_window":
        if text.shape[0] != audio.shape[0]:
            raise ValueError("per_window text_mode expects one text embedding row per audio window.")
    else:
        raise ValueError(f"Unsupported text_mode: {text_mode}")
    return build_fusion_features(audio, text)


def build_multimodal_curve(
    windows: list[AudioWindow],
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    fusion_model: Any,
    label_names: list[str],
    text_mode: str = "repeat_global",
) -> pd.DataFrame:
    """Return a window-level probability time series from a fusion model."""
    if not windows:
        raise ValueError("windows must not be empty.")
    if len(windows) != np.asarray(audio_embeddings).shape[0]:
        raise ValueError("windows and audio_embeddings row count mismatch.")
    if not label_names:
        raise ValueError("label_names must not be empty.")

    features = build_window_fusion_features(audio_embeddings, text_embeddings, text_mode=text_mode)
    probabilities = np.asarray(fusion_model.predict_proba(features), dtype=np.float64)
    if probabilities.shape != (len(windows), len(label_names)):
        raise ValueError(
            "predict_proba shape mismatch: "
            f"expected {(len(windows), len(label_names))}, got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("predicted probabilities contain NaN or Inf values.")

    pred_ids = probabilities.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for row_index, window in enumerate(windows):
        row: dict[str, Any] = {
            "window_index": int(window.index),
            "start_time": float(window.start_time),
            "end_time": float(window.end_time),
            "duration": float(window.end_time - window.start_time),
            "pred_label_id": int(pred_ids[row_index]),
            "pred_emotion": label_names[int(pred_ids[row_index])],
            "confidence": float(probabilities[row_index, pred_ids[row_index]]),
        }
        for label_index, label_name in enumerate(label_names):
            row[f"prob_{label_name}"] = float(probabilities[row_index, label_index])
        rows.append(row)
    return pd.DataFrame(rows)


def extract_window_audio_embeddings(
    windows: list[AudioWindow],
    audio_extractor: Any,
    sample_rate: int,
) -> np.ndarray:
    """Extract one audio embedding per window using an injected extractor."""
    rows = [audio_extractor.extract(window.waveform, sample_rate) for window in windows]
    features = np.vstack(rows).astype(np.float32)
    if features.shape[0] != len(windows):
        raise RuntimeError("Window audio embedding row count mismatch.")
    if not np.isfinite(features).all():
        raise ValueError("Window audio embeddings contain NaN or Inf values.")
    return features


def extract_asr_text_embedding(asr_text: str, text_extractor: Any) -> np.ndarray:
    """Extract one global ASR-text embedding for repeat_global mode."""
    text = str(asr_text).strip() or "[EMPTY_ASR]"
    features = np.asarray(text_extractor.extract_batch([text]), dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != 1:
        raise ValueError(f"Expected one text embedding row, got shape {features.shape}.")
    if not np.isfinite(features).all():
        raise ValueError("ASR text embedding contains NaN or Inf values.")
    return features


def analyze_multimodal_emotion_curve(
    waveform: np.ndarray,
    sample_rate: int,
    asr_text: str,
    audio_extractor: Any,
    text_extractor: Any,
    fusion_model: Any,
    label_names: list[str],
    config: CurveConfig = CurveConfig(),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the dependency-injected multimodal curve pipeline."""
    windows = list(
        iter_sliding_windows(
            waveform=waveform,
            sample_rate=sample_rate,
            window_seconds=config.window_seconds,
            step_seconds=config.step_seconds,
        )
    )
    audio_embeddings = extract_window_audio_embeddings(windows, audio_extractor, sample_rate)
    text_embeddings = extract_asr_text_embedding(asr_text, text_extractor)
    curve = build_multimodal_curve(
        windows=windows,
        audio_embeddings=audio_embeddings,
        text_embeddings=text_embeddings,
        fusion_model=fusion_model,
        label_names=label_names,
        text_mode=config.text_mode,
    )
    summary = build_curve_summary(
        curve=curve,
        label_names=label_names,
        high_emotion_threshold=config.high_emotion_threshold,
        change_delta=config.change_delta,
    )
    return curve, summary


def build_curve_summary(
    curve: pd.DataFrame,
    label_names: list[str],
    high_emotion_threshold: float = 0.6,
    change_delta: float = 0.25,
) -> dict[str, Any]:
    """Summarize dominant emotion, volatility, changes, and high affect spans."""
    prob_columns = [f"prob_{name}" for name in label_names]
    missing = set(prob_columns + ["start_time", "end_time", "pred_emotion"]).difference(curve.columns)
    if missing:
        raise ValueError(f"Missing curve columns: {sorted(missing)}")
    probabilities = curve[prob_columns].to_numpy(dtype=np.float64)
    if probabilities.shape[0] == 0:
        raise ValueError("curve must not be empty.")

    mean_probs = probabilities.mean(axis=0)
    dominant_index = int(mean_probs.argmax())
    diffs = np.abs(np.diff(probabilities, axis=0))
    step_changes = diffs.sum(axis=1) if diffs.size else np.array([], dtype=np.float64)
    volatility = float(step_changes.mean()) if step_changes.size else 0.0

    change_points: list[dict[str, Any]] = []
    for idx, _delta in enumerate(step_changes, start=1):
        previous_emotion = str(curve.loc[idx - 1, "pred_emotion"])
        current_emotion = str(curve.loc[idx, "pred_emotion"])
        max_delta = float(np.max(diffs[idx - 1]))
        if previous_emotion != current_emotion or max_delta >= change_delta:
            change_points.append(
                {
                    "window_index": int(curve.loc[idx, "window_index"]),
                    "time": float(curve.loc[idx, "start_time"]),
                    "from_emotion": previous_emotion,
                    "to_emotion": current_emotion,
                    "delta": max_delta,
                }
            )

    high_segments: list[dict[str, Any]] = []
    for emotion in ["sad", "angry"]:
        column = f"prob_{emotion}"
        if column not in curve.columns:
            continue
        for _, row in curve[curve[column].ge(high_emotion_threshold)].iterrows():
            high_segments.append(
                {
                    "emotion": emotion,
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "probability": float(row[column]),
                }
            )

    return {
        "dominant_emotion": label_names[dominant_index],
        "dominant_probability": float(mean_probs[dominant_index]),
        "mean_probabilities": {label: float(mean_probs[idx]) for idx, label in enumerate(label_names)},
        "volatility_strength": volatility,
        "emotion_change_points": change_points,
        "high_emotion_segments": high_segments,
    }


def build_plot_annotations(summary: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic annotation layers for the curve figure."""
    dominant = str(summary.get("dominant_emotion", "unknown"))
    volatility = float(summary.get("volatility_strength", 0.0))
    change_times = [float(item["time"]) for item in summary.get("emotion_change_points", []) if "time" in item]
    high_segments = [
        {
            "emotion": str(item.get("emotion", "")),
            "start_time": float(item["start_time"]),
            "end_time": float(item["end_time"]),
        }
        for item in summary.get("high_emotion_segments", [])
        if "start_time" in item and "end_time" in item
    ]
    return {
        "info_text": f"dominant={dominant} | volatility={volatility:.3f}",
        "change_times": change_times,
        "high_segments": high_segments,
    }


def plot_curve(curve: pd.DataFrame, label_names: list[str], output_path: Path, summary: dict[str, Any] | None = None) -> None:
    """Save a probability curve figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    annotations = build_plot_annotations(summary or {})
    segment_colors = {"angry": "#f4a3a3", "sad": "#b7a2d8"}
    for segment in annotations["high_segments"]:
        emotion = segment["emotion"]
        ax.axvspan(
            segment["start_time"],
            segment["end_time"],
            color=segment_colors.get(emotion, "#d9d9d9"),
            alpha=0.18,
            linewidth=0,
        )
    x = curve["start_time"].to_numpy(dtype=float)
    for label_name in label_names:
        ax.plot(x, curve[f"prob_{label_name}"], marker="o", linewidth=1.8, label=label_name)
    for time in annotations["change_times"]:
        ax.axvline(time, color="#333333", linestyle="--", linewidth=1.1, alpha=0.55)
        ax.text(time, 0.98, "change", rotation=90, va="top", ha="right", fontsize=8, color="#333333")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Multimodal Emotion Curve")
    ax.text(
        0.01,
        0.02,
        annotations["info_text"],
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#bbbbbb"},
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_curve_outputs(
    curve: pd.DataFrame,
    summary: dict[str, Any],
    label_names: list[str],
    output_dir: Path,
    run_id: str,
    config: CurveConfig,
) -> dict[str, Path]:
    """Write curve CSV, summary JSON, config JSON, and figure artifacts."""
    curves_dir = output_dir / "curves"
    figures_dir = output_dir / "figures"
    curves_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    curve_csv = curves_dir / f"{run_id}_emotion_curve.csv"
    summary_json = curves_dir / f"{run_id}_summary.json"
    config_json = curves_dir / f"{run_id}_config.json"
    figure_png = figures_dir / f"{run_id}_emotion_curve.png"

    curve.to_csv(curve_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_json.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_curve(curve, label_names, figure_png, summary=summary)

    return {
        "curve_csv": curve_csv,
        "summary_json": summary_json,
        "config_json": config_json,
        "figure_png": figure_png,
    }
