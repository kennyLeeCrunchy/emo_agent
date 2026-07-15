"""
测试 MFCC/传统声学特征提取是否生成固定维度且顺序对齐的输出。
Test whether MFCC/traditional acoustic extraction creates fixed-width, order-aligned outputs.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.module2_audio_features.mfcc_features import (
    FEATURE_NAMES,
    extract_features_from_audio_value,
    extract_feature_dataset,
)


def make_wav_bytes(frequency: float = 220.0, sr: int = 16000, seconds: float = 0.4) -> bytes:
    samples = np.arange(int(sr * seconds), dtype=np.float32) / sr
    waveform = 0.2 * np.sin(2 * np.pi * frequency * samples)
    buffer = io.BytesIO()
    wavfile.write(buffer, sr, waveform)
    return buffer.getvalue()


def test_extract_features_from_audio_value_returns_fixed_finite_vector() -> None:
    audio_value = {"bytes": make_wav_bytes()}

    features = extract_features_from_audio_value(audio_value)

    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()
    assert "mfcc_01_mean" in FEATURE_NAMES
    assert "pitch_mean_hz" in FEATURE_NAMES


def test_extract_feature_dataset_preserves_metadata_order(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "features"
    data_dir.mkdir()

    parquet_path = data_dir / "train-00000-of-00001.parquet"
    table = pa.table(
        {
            "audio": [
                {"bytes": make_wav_bytes(220.0), "path": None},
                {"bytes": make_wav_bytes(440.0), "path": None},
            ],
            "text": ["low tone", "high tone"],
            "emotion": ["neutral", "happy"],
            "speaker": ["spk1", "spk2"],
        }
    )
    pq.write_table(table, parquet_path)

    metadata = pd.DataFrame(
        {
            "sample_id": [
                "train-00000-of-00001:1",
                "train-00000-of-00001:0",
            ],
            "emotion": ["happy", "neutral"],
            "label_id": [1, 0],
            "parquet_file": [parquet_path.name, parquet_path.name],
            "row_id": [1, 0],
            "split": ["train", "train"],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    features, ids = extract_feature_dataset(
        metadata_path=metadata_path,
        parquet_dir=data_dir,
        output_dir=output_dir,
    )

    saved_features = np.load(output_dir / "audio_mfcc.npy")
    saved_ids = pd.read_csv(output_dir / "audio_ids.csv")

    assert features.shape == (2, len(FEATURE_NAMES))
    assert np.array_equal(features, saved_features)
    assert ids["sample_id"].tolist() == metadata["sample_id"].tolist()
    assert saved_ids["sample_id"].tolist() == metadata["sample_id"].tolist()
