"""
提取轻量传统声学特征，作为纯音频情绪识别的可解释 baseline。
Extract lightweight traditional acoustic features as an interpretable pure-audio emotion baseline.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.fftpack import dct
from scipy.io import wavfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs" / "metadata.csv"
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "CSEMOTIONS" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "modules" / "module2_audio_features" / "outputs" / "mfcc"
N_MFCC = 13

SUMMARY_FEATURE_NAMES = [
    "duration_sec",
    "rms_mean",
    "rms_std",
    "rms_max",
    "zcr_mean",
    "zcr_std",
    "spectral_centroid_mean_hz",
    "spectral_centroid_std_hz",
    "spectral_bandwidth_mean_hz",
    "spectral_bandwidth_std_hz",
    "spectral_rolloff85_mean_hz",
    "spectral_rolloff85_std_hz",
    "pitch_mean_hz",
    "pitch_std_hz",
    "voiced_ratio",
]
MFCC_MEAN_NAMES = [f"mfcc_{idx:02d}_mean" for idx in range(1, N_MFCC + 1)]
MFCC_STD_NAMES = [f"mfcc_{idx:02d}_std" for idx in range(1, N_MFCC + 1)]
FEATURE_NAMES = SUMMARY_FEATURE_NAMES + MFCC_MEAN_NAMES + MFCC_STD_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract audio MFCC baseline features.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of metadata rows to process after sampling.",
    )
    parser.add_argument(
        "--balanced-per-class",
        type=int,
        default=None,
        help="Optional number of rows per emotion class for a balanced trial run.",
    )
    return parser.parse_args()


def extract_audio_bytes(audio_value: Any) -> bytes:
    if isinstance(audio_value, bytes):
        return audio_value
    if isinstance(audio_value, bytearray):
        return bytes(audio_value)
    if isinstance(audio_value, dict):
        for key in ("bytes", "data", "audio"):
            value = audio_value.get(key)
            if isinstance(value, bytes):
                return value
            if isinstance(value, bytearray):
                return bytes(value)
    raise TypeError(f"Unsupported audio value type: {type(audio_value)!r}")


def read_waveform(audio_value: Any) -> tuple[np.ndarray, int]:
    audio_bytes = extract_audio_bytes(audio_value)
    sample_rate, waveform = wavfile.read(io.BytesIO(audio_bytes))

    waveform = np.asarray(waveform)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if np.issubdtype(waveform.dtype, np.integer):
        max_abs = float(np.iinfo(waveform.dtype).max)
        waveform = waveform.astype(np.float32) / max_abs
    else:
        waveform = waveform.astype(np.float32)

    waveform = np.nan_to_num(waveform, copy=False)
    return waveform, int(sample_rate)


def frame_signal(
    waveform: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    frame_length = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_ms / 1000.0)))

    if waveform.size < frame_length:
        waveform = np.pad(waveform, (0, frame_length - waveform.size))

    frame_count = 1 + int(np.floor((waveform.size - frame_length) / hop_length))
    strides = (waveform.strides[0] * hop_length, waveform.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(frame_count, frame_length),
        strides=strides,
        writeable=False,
    )
    return frames.copy()


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, fft_size: int, n_filters: int = 26) -> np.ndarray:
    low_mel = hz_to_mel(0.0)
    high_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, fft_size // 2)

    filters = np.zeros((n_filters, fft_size // 2 + 1), dtype=np.float32)
    for idx in range(1, n_filters + 1):
        left, center, right = bin_points[idx - 1], bin_points[idx], bin_points[idx + 1]
        if center > left:
            filters[idx - 1, left:center] = (
                np.arange(left, center) - left
            ) / float(center - left)
        if right > center:
            filters[idx - 1, center:right] = (
                right - np.arange(center, right)
            ) / float(right - center)
    return filters


def summarize(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std())


def estimate_pitch_values(
    frames: np.ndarray,
    sample_rate: int,
    min_pitch: float = 50.0,
    max_pitch: float = 500.0,
    max_frames: int = 80,
) -> np.ndarray:
    if frames.size == 0:
        return np.empty(0, dtype=np.float32)

    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = frames[indices]

    min_lag = max(1, int(sample_rate / max_pitch))
    max_lag = min(frames.shape[1] - 1, int(sample_rate / min_pitch))
    if max_lag <= min_lag:
        return np.empty(0, dtype=np.float32)

    pitches: list[float] = []
    for frame in frames:
        frame = frame.astype(np.float64)
        frame = frame - frame.mean()
        energy = float(np.dot(frame, frame))
        if energy < 1e-8:
            continue

        fft_size = 1 << int(np.ceil(np.log2(2 * len(frame) - 1)))
        spectrum = np.fft.rfft(frame, fft_size)
        corr = np.fft.irfft(spectrum * np.conj(spectrum), fft_size)[: len(frame)]
        search = corr[min_lag:max_lag]
        if search.size == 0:
            continue

        best_lag = int(np.argmax(search)) + min_lag
        confidence = float(corr[best_lag] / (corr[0] + 1e-8))
        if confidence >= 0.25:
            pitches.append(sample_rate / best_lag)

    return np.asarray(pitches, dtype=np.float32)


def compute_feature_vector(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    duration = waveform.size / float(sample_rate)
    emphasized = np.append(waveform[:1], waveform[1:] - 0.97 * waveform[:-1])
    frames = frame_signal(emphasized, sample_rate)
    window = np.hamming(frames.shape[1]).astype(np.float32)
    windowed = frames * window

    fft_size = 1 << int(np.ceil(np.log2(frames.shape[1])))
    spectrum = np.fft.rfft(windowed, n=fft_size, axis=1)
    power = (np.abs(spectrum) ** 2) / float(fft_size)
    magnitude = np.sqrt(power)
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

    rms = np.sqrt(np.mean(frames**2, axis=1))
    signs = np.signbit(frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)

    mag_sum = magnitude.sum(axis=1) + 1e-10
    centroid = (magnitude * freqs).sum(axis=1) / mag_sum
    bandwidth = np.sqrt(((freqs - centroid[:, None]) ** 2 * magnitude).sum(axis=1) / mag_sum)
    cumulative_power = np.cumsum(power, axis=1)
    rolloff_threshold = 0.85 * cumulative_power[:, -1:]
    rolloff_indices = np.argmax(cumulative_power >= rolloff_threshold, axis=1)
    rolloff = freqs[rolloff_indices]

    filters = mel_filterbank(sample_rate, fft_size)
    mel_energy = np.maximum(power @ filters.T, 1e-10)
    log_mel = np.log(mel_energy)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :N_MFCC]
    mfcc_mean = mfcc.mean(axis=0)
    mfcc_std = mfcc.std(axis=0)

    pitches = estimate_pitch_values(frames, sample_rate)
    pitch_mean, pitch_std = summarize(pitches)
    voiced_ratio = float(len(pitches) / min(len(frames), 80))

    values = [
        duration,
        *summarize(rms),
        float(rms.max()) if rms.size else 0.0,
        *summarize(zcr),
        *summarize(centroid),
        *summarize(bandwidth),
        *summarize(rolloff),
        pitch_mean,
        pitch_std,
        voiced_ratio,
        *mfcc_mean.tolist(),
        *mfcc_std.tolist(),
    ]
    return np.nan_to_num(np.asarray(values, dtype=np.float32))


def extract_features_from_audio_value(audio_value: Any) -> np.ndarray:
    waveform, sample_rate = read_waveform(audio_value)
    return compute_feature_vector(waveform, sample_rate)


def select_metadata_rows(
    metadata: pd.DataFrame,
    limit: int | None = None,
    balanced_per_class: int | None = None,
) -> pd.DataFrame:
    selected = metadata
    if balanced_per_class is not None:
        selected = (
            metadata.groupby("emotion", group_keys=False, sort=True)
            .head(balanced_per_class)
            .reset_index(drop=True)
        )
    if limit is not None:
        selected = selected.head(limit).reset_index(drop=True)
    return selected.reset_index(drop=True)


def read_audio_column(parquet_dir: Path, parquet_file: str) -> pd.Series:
    file_path = parquet_dir / parquet_file
    return pq.ParquetFile(file_path).read(columns=["audio"]).to_pandas()["audio"]


def extract_feature_dataset(
    metadata_path: Path,
    parquet_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    balanced_per_class: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    metadata = pd.read_csv(metadata_path)
    metadata = select_metadata_rows(metadata, limit=limit, balanced_per_class=balanced_per_class)

    feature_rows: list[np.ndarray | None] = [None] * len(metadata)
    id_rows: list[dict[str, Any] | None] = [None] * len(metadata)
    processed = 0

    for parquet_name, group in metadata.groupby("parquet_file", sort=False):
        audio_column = read_audio_column(parquet_dir, str(parquet_name))
        for idx, row in group.iterrows():
            audio_value = audio_column.iloc[int(row["row_id"])]
            feature_rows[idx] = extract_features_from_audio_value(audio_value)
            id_rows[idx] = {
                "sample_id": row["sample_id"],
                "emotion": row["emotion"],
                "label_id": int(row["label_id"]),
                "split": row["split"],
                "parquet_file": row["parquet_file"],
                "row_id": int(row["row_id"]),
            }
            processed += 1
            if processed % 50 == 0:
                print(f"Processed {processed}/{len(metadata)} samples")

    features = np.vstack([row for row in feature_rows if row is not None]).astype(np.float32)
    ids = pd.DataFrame([row for row in id_rows if row is not None])

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "audio_mfcc.npy", features)
    ids.to_csv(output_dir / "audio_ids.csv", index=False, encoding="utf-8-sig")
    (output_dir / "audio_feature_names.json").write_text(
        json.dumps(FEATURE_NAMES, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return features, ids


def main() -> int:
    args = parse_args()
    features, ids = extract_feature_dataset(
        metadata_path=args.metadata,
        parquet_dir=args.parquet_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        balanced_per_class=args.balanced_per_class,
    )
    print(f"Wrote features: {args.output_dir / 'audio_mfcc.npy'} {features.shape}")
    print(f"Wrote ids: {args.output_dir / 'audio_ids.csv'} {len(ids)} rows")
    print(f"Wrote feature names: {args.output_dir / 'audio_feature_names.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
