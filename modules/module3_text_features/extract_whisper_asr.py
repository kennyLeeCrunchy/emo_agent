"""
Transcribe dataset audio with Whisper ASR for text-route experiments.

The output keeps the original sample ids and labels, stores the original text
as ``gold_text``, and replaces ``text`` with the ASR transcript so existing
RoBERTa embedding extraction can be reused unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.module2_audio_features.mfcc_features import read_audio_column, read_waveform
from modules.module2_ssl_audio.extract_ssl_features import prepare_waveform, resolve_device


DEFAULT_METADATA_PATH = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs_disjoint_split" / "02_text_unique" / "metadata.csv"
DEFAULT_PARQUET_DIR = PROJECT_ROOT / "CSEMOTIONS" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "modules" / "module3_text_features" / "outputs" / "text_unique_asr" / "00_whisper_asr"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "models" / "pretrained" / "huggingface"
DEFAULT_MODEL_NAME = "openai/whisper-tiny"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe dataset audio with Whisper ASR.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N sample limit for smoke runs.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


class WhisperAsrTranscriber:
    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        device: str,
        local_files_only: bool,
        max_new_tokens: int,
    ) -> None:
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        self.device_name = resolve_device(device)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
        ).to(self.device_name)
        self.model.eval()
        self.target_sample_rate = int(self.processor.feature_extractor.sampling_rate)
        self.max_new_tokens = max_new_tokens
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="zh",
            task="transcribe",
        )
        self.model.generation_config.forced_decoder_ids = self.forced_decoder_ids

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> str:
        waveform = prepare_waveform(waveform, sample_rate, self.target_sample_rate)
        inputs = self.processor(
            waveform,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(self.device_name)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_features,
                max_new_tokens=self.max_new_tokens,
            )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def load_metadata(metadata_path: Path, limit: int | None = None) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    required = {"sample_id", "text", "emotion", "label_id", "split", "parquet_file", "row_id"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    metadata = metadata.reset_index(drop=True)
    if limit is not None:
        metadata = metadata.head(limit).reset_index(drop=True)
    return metadata


def transcribe_dataset(
    metadata_path: Path,
    parquet_dir: Path,
    output_dir: Path,
    transcriber: WhisperAsrTranscriber,
    limit: int | None = None,
) -> pd.DataFrame:
    metadata = load_metadata(metadata_path, limit=limit)
    output = metadata.copy()
    output.insert(output.columns.get_loc("text") + 1, "gold_text", output["text"].astype(str))

    transcripts: list[str] = [""] * len(output)
    processed = 0
    for parquet_name, group in output.groupby("parquet_file", sort=False):
        audio_column = read_audio_column(parquet_dir, str(parquet_name))
        for idx, row in group.iterrows():
            audio_value: Any = audio_column.iloc[int(row["row_id"])]
            waveform, sample_rate = read_waveform(audio_value)
            transcripts[int(idx)] = transcriber.transcribe(waveform, sample_rate)
            processed += 1
            if processed % 25 == 0 or processed == len(output):
                print(f"[asr] processed {processed}/{len(output)} samples")

    output["asr_text"] = transcripts
    output["text"] = output["asr_text"].replace("", "[EMPTY_ASR]")
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "metadata_asr.csv", index=False, encoding="utf-8-sig")
    model_info = {
        "model_name": transcriber.model.config.name_or_path,
        "source_metadata_path": str(metadata_path),
        "rows": int(len(output)),
        "split_counts": output["split"].value_counts().to_dict(),
        "text_column": "asr_text copied into text",
        "gold_text_column": "gold_text",
        "max_new_tokens": transcriber.max_new_tokens,
    }
    (output_dir / "model_info.json").write_text(
        json.dumps(model_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    args = parse_args()
    transcriber = WhisperAsrTranscriber(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=args.device,
        local_files_only=args.local_files_only,
        max_new_tokens=args.max_new_tokens,
    )
    output = transcribe_dataset(
        metadata_path=args.metadata_path,
        parquet_dir=args.parquet_dir,
        output_dir=args.output_dir,
        transcriber=transcriber,
        limit=args.limit,
    )
    print(f"Wrote ASR metadata: rows={len(output)}")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
