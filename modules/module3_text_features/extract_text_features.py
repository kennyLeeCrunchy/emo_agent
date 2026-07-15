"""
Extract frozen Chinese BERT/RoBERTa sentence embeddings for Module 3.

The script reads the canonical Module 1 metadata, cleans text minimally,
mean-pools the final hidden states with the attention mask, and saves an
aligned feature matrix plus text id file for downstream Logistic Regression.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs" / "metadata.csv"
MODULE_OUTPUTS_DIR = PROJECT_ROOT / "modules" / "module3_text_features" / "outputs"
DEFAULT_OUTPUT_DIR = MODULE_OUTPUTS_DIR / "text_full" / "01_roberta_features_full_samples"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "models" / "pretrained" / "huggingface"
DEFAULT_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
TEXT_EMBEDDINGS_FILENAME = "text_embeddings.npy"
TEXT_IDS_FILENAME = "text_ids.csv"
MODEL_INFO_FILENAME = "model_info.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Chinese text embeddings.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N sample limit for smoke runs.")
    parser.add_argument("--local-files-only", action="store_true", help="Only use already downloaded model files.")
    return parser.parse_args()


def clean_text(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        raise ValueError("Text is empty after cleaning.")
    return text


def validate_text_ids(ids: pd.DataFrame) -> None:
    required = {"sample_id", "text", "emotion", "label_id", "split", "parquet_file", "row_id"}
    missing = required.difference(ids.columns)
    if missing:
        raise ValueError(f"Missing text id columns: {sorted(missing)}")
    if ids["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values found in text ids.")
    bad_splits = set(ids["split"].dropna().astype(str)).difference({"train", "val", "test"})
    if bad_splits:
        raise ValueError(f"Unexpected split values: {sorted(bad_splits)}")
    if ids["label_id"].isna().any():
        raise ValueError("label_id contains missing values.")
    empty_rows: list[int] = []
    for row_index, value in ids["text"].items():
        try:
            clean_text(value)
        except ValueError:
            empty_rows.append(int(row_index))
    if empty_rows:
        raise ValueError(f"Empty text rows after cleaning: {empty_rows[:10]}")


def load_text_ids(metadata_path: Path, limit: int | None = None) -> pd.DataFrame:
    ids = pd.read_csv(metadata_path)
    validate_text_ids(ids)
    ids = ids.reset_index(drop=True)
    ids["text"] = [clean_text(value) for value in ids["text"]]
    if limit is not None:
        ids = ids.head(limit).reset_index(drop=True)
    return ids


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def mean_pool_last_hidden_state(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    masked_hidden = last_hidden_state * mask
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    return masked_hidden.sum(dim=1) / token_counts


@dataclass
class TextEmbeddingExtractor:
    model_name: str = DEFAULT_MODEL_NAME
    cache_dir: Path | None = DEFAULT_CACHE_DIR
    device: str = "auto"
    max_length: int = 128
    local_files_only: bool = False

    def __post_init__(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device_name = resolve_device(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
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
        self.feature_dim = int(getattr(self.model.config, "hidden_size", 0))

    def extract_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device_name) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = self.model(**encoded)
        pooled = mean_pool_last_hidden_state(outputs.last_hidden_state, encoded["attention_mask"])
        return pooled.detach().cpu().numpy().astype(np.float32)


def extract_text_dataset(
    metadata_path: Path,
    output_dir: Path,
    extractor: TextEmbeddingExtractor,
    batch_size: int = 32,
    limit: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    ids = load_text_ids(metadata_path, limit=limit)
    feature_batches: list[np.ndarray] = []
    texts = ids["text"].tolist()
    total = len(texts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        feature_batches.append(extractor.extract_batch(texts[start:end]))
        print(f"[text] processed {end}/{total} samples")

    features = np.vstack(feature_batches).astype(np.float32)
    if features.shape[0] != len(ids):
        raise RuntimeError(f"Expected {len(ids)} feature rows, got {features.shape[0]}")
    if not np.isfinite(features).all():
        raise ValueError("Text feature matrix contains NaN or Inf values.")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / TEXT_EMBEDDINGS_FILENAME, features)
    ids.to_csv(output_dir / TEXT_IDS_FILENAME, index=False, encoding="utf-8-sig")
    model_info = {
        "model_name": extractor.model_name,
        "feature_dim": int(features.shape[1]),
        "feature_shape": list(features.shape),
        "pooling": "attention_mask_mean_pool_last_hidden_state",
        "max_length": extractor.max_length,
        "batch_size": batch_size,
        "split_counts": ids["split"].value_counts().to_dict(),
        "local_files_only": extractor.local_files_only,
    }
    (output_dir / MODEL_INFO_FILENAME).write_text(
        json.dumps(model_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return features, ids


def main() -> int:
    args = parse_args()
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    extractor = TextEmbeddingExtractor(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=args.device,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )
    features, ids = extract_text_dataset(
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        extractor=extractor,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(f"Wrote text embeddings: {features.shape}, rows={len(ids)}")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
