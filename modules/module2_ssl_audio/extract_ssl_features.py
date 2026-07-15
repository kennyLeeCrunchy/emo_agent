"""
Extract lightweight HuBERT or Whisper utterance embeddings for Module 2 comparison.

This script reads the same sample ids used by the MFCC baseline, loads audio from
the CSEMOTIONS parquet shards, mean-pools the model encoder states into one vector
per utterance, and saves aligned .npy feature matrices plus id files.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

from modules.module2_audio_features.mfcc_features import read_audio_column, read_waveform


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IDS_PATH = PROJECT_ROOT / "modules" / "module2_audio_features" / "outputs" / "mfcc" / "audio_ids.csv"
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "CSEMOTIONS" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "modules" / "module2_ssl_audio" / "outputs"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "models" / "pretrained" / "huggingface"
DEFAULT_HUBERT_MODEL = "facebook/hubert-base-ls960"
DEFAULT_WHISPER_MODEL = "openai/whisper-tiny"


class EmbeddingExtractor(Protocol):
    feature_dim: int | None

    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return one fixed-width utterance embedding."""


@dataclass
class HubertExtractor:
    model_name: str = DEFAULT_HUBERT_MODEL
    cache_dir: Path | None = DEFAULT_CACHE_DIR
    device: str = "auto"
    local_files_only: bool = False

    def __post_init__(self) -> None:
        from transformers import AutoFeatureExtractor, AutoModel

        self.device_name = resolve_device(self.device)
        self.processor = AutoFeatureExtractor.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            local_files_only=self.local_files_only,
        ).to(self.device_name)
        self.model.eval()
        self.target_sample_rate = int(getattr(self.processor, "sampling_rate", 16000))
        self.feature_dim = int(getattr(self.model.config, "hidden_size", 0)) or None

    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        waveform = prepare_waveform(waveform, sample_rate, self.target_sample_rate)
        inputs = self.processor(
            waveform,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device_name) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding.detach().cpu().numpy().astype(np.float32)


@dataclass
class WhisperExtractor:
    model_name: str = DEFAULT_WHISPER_MODEL
    cache_dir: Path | None = DEFAULT_CACHE_DIR
    device: str = "auto"
    local_files_only: bool = False

    def __post_init__(self) -> None:
        from transformers import AutoProcessor, WhisperModel

        self.device_name = resolve_device(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            local_files_only=self.local_files_only,
        )
        self.model = WhisperModel.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            local_files_only=self.local_files_only,
        ).to(self.device_name)
        self.model.eval()
        self.target_sample_rate = int(getattr(self.processor.feature_extractor, "sampling_rate", 16000))
        self.feature_dim = int(getattr(self.model.config, "d_model", 0)) or None

    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        waveform = prepare_waveform(waveform, sample_rate, self.target_sample_rate)
        inputs = self.processor(
            waveform,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(self.device_name)
        with torch.inference_mode():
            outputs = self.model.encoder(input_features=input_features)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding.detach().cpu().numpy().astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HuBERT or Whisper utterance embeddings.")
    parser.add_argument("--feature-set", choices=["hubert", "whisper", "both"], default="both")
    parser.add_argument("--ids-path", type=Path, default=DEFAULT_IDS_PATH)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--hubert-model", default=DEFAULT_HUBERT_MODEL)
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N sample limit for smoke runs.")
    parser.add_argument("--local-files-only", action="store_true", help="Only use already downloaded model files.")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def prepare_waveform(waveform: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    if sample_rate != target_sample_rate:
        divisor = gcd(int(sample_rate), int(target_sample_rate))
        up = int(target_sample_rate) // divisor
        down = int(sample_rate) // divisor
        waveform = resample_poly(waveform, up=up, down=down).astype(np.float32)
    return np.nan_to_num(waveform, copy=False)


def load_aligned_ids(ids_path: Path, limit: int | None = None) -> pd.DataFrame:
    ids = pd.read_csv(ids_path)
    required_columns = {"sample_id", "emotion", "label_id", "split", "parquet_file", "row_id"}
    missing = required_columns.difference(ids.columns)
    if missing:
        raise ValueError(f"Missing id columns: {sorted(missing)}")
    ids = ids.reset_index(drop=True)
    if limit is not None:
        ids = ids.head(limit).reset_index(drop=True)
    return ids


def extract_feature_dataset(
    ids_path: Path,
    parquet_dir: Path,
    output_dir: Path,
    extractor: EmbeddingExtractor,
    feature_name: str,
    limit: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    ids = load_aligned_ids(ids_path, limit=limit)
    feature_rows: list[np.ndarray | None] = [None] * len(ids)
    output_subdir = output_dir / feature_name
    processed = 0

    for parquet_name, group in ids.groupby("parquet_file", sort=False):
        audio_column = read_audio_column(parquet_dir, str(parquet_name))
        for idx, row in group.iterrows():
            audio_value: Any = audio_column.iloc[int(row["row_id"])]
            waveform, sample_rate = read_waveform(audio_value)
            feature_rows[int(idx)] = extractor.extract(waveform, sample_rate)
            processed += 1
            if processed % 10 == 0 or processed == len(ids):
                print(f"[{feature_name}] processed {processed}/{len(ids)} samples")

    features = np.vstack([row for row in feature_rows if row is not None]).astype(np.float32)
    if features.shape[0] != len(ids):
        raise RuntimeError(f"Expected {len(ids)} feature rows, got {features.shape[0]}")

    output_subdir.mkdir(parents=True, exist_ok=True)
    np.save(output_subdir / f"audio_{feature_name}.npy", features)
    ids.to_csv(output_subdir / "audio_ids.csv", index=False, encoding="utf-8-sig")
    (output_subdir / "model_info.txt").write_text(
        f"feature_set={feature_name}\nfeature_dim={features.shape[1]}\n",
        encoding="utf-8",
    )
    return features, ids


def build_extractors(args: argparse.Namespace) -> list[tuple[str, EmbeddingExtractor]]:
    extractors: list[tuple[str, EmbeddingExtractor]] = []
    if args.feature_set in {"hubert", "both"}:
        extractors.append(
            (
                "hubert",
                HubertExtractor(
                    model_name=args.hubert_model,
                    cache_dir=args.cache_dir,
                    device=args.device,
                    local_files_only=args.local_files_only,
                ),
            )
        )
    if args.feature_set in {"whisper", "both"}:
        extractors.append(
            (
                "whisper",
                WhisperExtractor(
                    model_name=args.whisper_model,
                    cache_dir=args.cache_dir,
                    device=args.device,
                    local_files_only=args.local_files_only,
                ),
            )
        )
    return extractors


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {resolve_device(args.device)}")
    print(f"Using ids: {args.ids_path}")
    for feature_name, extractor in build_extractors(args):
        features, ids = extract_feature_dataset(
            ids_path=args.ids_path,
            parquet_dir=args.parquet_dir,
            output_dir=args.output_dir,
            extractor=extractor,
            feature_name=feature_name,
            limit=args.limit,
        )
        print(f"Wrote {feature_name}: {features.shape}, rows={len(ids)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
