"""Run a multimodal sliding-window emotion curve for one audio sample."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.module2_audio_features.mfcc_features import read_audio_column, read_waveform
from modules.module2_ssl_audio.extract_ssl_features import WhisperExtractor
from modules.module3_text_features.extract_text_features import TextEmbeddingExtractor
from modules.module3_text_features.extract_whisper_asr import WhisperAsrTranscriber
from modules.module5_emotion_curve.emotion_curve import (
    CurveConfig,
    analyze_multimodal_emotion_curve,
    save_curve_outputs,
)


DEFAULT_METADATA_PATH = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs_disjoint_split" / "02_text_unique" / "metadata.csv"
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "CSEMOTIONS" / "data"
DEFAULT_LABEL_MAP_PATH = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs" / "label_map.json"
DEFAULT_FUSION_MODEL_PATH = (
    PROJECT_ROOT
    / "modules"
    / "module4_fusion"
    / "outputs"
    / "fusion_asr_main"
    / "01_fusion_logistic_classifier"
    / "model"
    / "fusion_logistic_pipeline.joblib"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "models" / "pretrained" / "huggingface"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "modules" / "module5_emotion_curve" / "outputs" / "01_multimodal_curve_smoke"
DEFAULT_WHISPER_MODEL = "openai/whisper-tiny"
DEFAULT_TEXT_MODEL = str(PROJECT_ROOT / "models" / "pretrained" / "manual" / "chinese-roberta-wwm-ext")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Module 5 multimodal emotion curve for one sample.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP_PATH)
    parser.add_argument("--fusion-model-path", type=Path, default=DEFAULT_FUSION_MODEL_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--sample-split", default="test", help="Split to sample from when --sample-id is not provided.")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--high-emotion-threshold", type=float, default=0.6)
    parser.add_argument("--change-delta", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


def label_names_from_map(label_map_path: Path) -> list[str]:
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(label_map.items(), key=lambda item: int(item[1]))]


def select_sample(metadata: pd.DataFrame, sample_index: int, sample_id: str | None, sample_split: str | None = None) -> pd.Series:
    if sample_id is not None:
        matches = metadata[metadata["sample_id"].astype(str).eq(str(sample_id))]
        if matches.empty:
            raise ValueError(f"sample_id not found: {sample_id}")
        return matches.iloc[0]
    candidates = metadata
    if sample_split:
        if "split" not in metadata.columns:
            raise ValueError("sample_split requires metadata to contain a split column.")
        candidates = metadata[metadata["split"].astype(str).eq(str(sample_split))].reset_index(drop=True)
        if candidates.empty:
            raise ValueError(f"No rows found for split: {sample_split}")
    if sample_index < 0 or sample_index >= len(candidates):
        raise ValueError(f"sample_index out of range: {sample_index}")
    return candidates.iloc[int(sample_index)]


def load_sample_waveform(
    metadata_path: Path,
    parquet_dir: Path,
    sample_index: int,
    sample_id: str | None,
    sample_split: str | None,
) -> tuple[Any, Any, pd.Series]:
    metadata = pd.read_csv(metadata_path)
    required = {"sample_id", "parquet_file", "row_id", "emotion", "label_id"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    row = select_sample(metadata, sample_index=sample_index, sample_id=sample_id, sample_split=sample_split)
    audio_column = read_audio_column(parquet_dir, str(row["parquet_file"]))
    waveform, sample_rate = read_waveform(audio_column.iloc[int(row["row_id"])])
    return waveform, sample_rate, row


def write_run_metadata(output_dir: Path, run_id: str, sample_row: pd.Series, asr_text: str, args: argparse.Namespace) -> Path:
    path = output_dir / "curves" / f"{run_id}_run_metadata.json"
    payload = {
        "sample_id": str(sample_row["sample_id"]),
        "split": str(sample_row.get("split", "")),
        "emotion": str(sample_row.get("emotion", "")),
        "label_id": int(sample_row.get("label_id", -1)),
        "parquet_file": str(sample_row["parquet_file"]),
        "row_id": int(sample_row["row_id"]),
        "asr_text": asr_text,
        "fusion_model_path": str(args.fusion_model_path),
        "whisper_model": args.whisper_model,
        "text_model": args.text_model,
        "text_mode": "repeat_global",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def safe_run_id(value: object) -> str:
    text = str(value).strip() or "unknown"
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("._") or "unknown"


def main() -> int:
    args = parse_args()
    waveform, sample_rate, sample_row = load_sample_waveform(
        metadata_path=args.metadata_path,
        parquet_dir=args.parquet_dir,
        sample_index=args.sample_index,
        sample_id=args.sample_id,
        sample_split=args.sample_split,
    )
    label_names = label_names_from_map(args.label_map)
    fusion_model = joblib.load(args.fusion_model_path)

    audio_extractor = WhisperExtractor(
        model_name=args.whisper_model,
        cache_dir=args.cache_dir,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    transcriber = WhisperAsrTranscriber(
        model_name=args.whisper_model,
        cache_dir=args.cache_dir,
        device=args.device,
        local_files_only=args.local_files_only,
        max_new_tokens=args.max_new_tokens,
    )
    text_extractor = TextEmbeddingExtractor(
        model_name=args.text_model,
        cache_dir=args.cache_dir,
        device=args.device,
        local_files_only=args.local_files_only,
    )

    asr_text = transcriber.transcribe(waveform, sample_rate)
    config = CurveConfig(
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        text_mode="repeat_global",
        high_emotion_threshold=args.high_emotion_threshold,
        change_delta=args.change_delta,
    )
    curve, summary = analyze_multimodal_emotion_curve(
        waveform=waveform,
        sample_rate=sample_rate,
        asr_text=asr_text,
        audio_extractor=audio_extractor,
        text_extractor=text_extractor,
        fusion_model=fusion_model,
        label_names=label_names,
        config=config,
    )
    run_id = f"sample_{safe_run_id(sample_row['sample_id'])}"
    paths = save_curve_outputs(curve, summary, label_names, args.output_dir, run_id, config)
    metadata_path = write_run_metadata(args.output_dir, run_id, sample_row, asr_text, args)

    print(f"sample_id={sample_row['sample_id']}")
    print(f"asr_text={asr_text}")
    print(f"dominant_emotion={summary['dominant_emotion']}")
    print(f"curve_csv={paths['curve_csv']}")
    print(f"summary_json={paths['summary_json']}")
    print(f"figure_png={paths['figure_png']}")
    print(f"run_metadata_json={metadata_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
