"""
Create leakage-aware text-disjoint splits for downstream reevaluation.

This keeps all rows with the same normalized text in one split for audio and
fusion evaluation, and also creates a one-row-per-text metadata table for the
pure text model (方案 A).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

from modules.module1_preprocess.preprocess import (
    SPLIT_NAMES,
    compute_class_distribution,
    compute_split_class_distribution,
    split_counts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs" / "metadata.csv"
DEFAULT_LABEL_MAP_PATH = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs" / "label_map.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs_disjoint_split"
OUTPUT_COLUMNS = [
    "sample_id",
    "text",
    "emotion",
    "speaker",
    "label_id",
    "parquet_file",
    "row_id",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create text-disjoint split metadata.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def validate_text_groups(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "text", "emotion", "label_id", "speaker", "parquet_file", "row_id"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    frame = metadata.copy()
    frame["normalized_text"] = frame["text"].map(normalize_text)
    empty = frame["normalized_text"].eq("")
    if empty.any():
        raise ValueError(f"Empty normalized_text rows: {frame.loc[empty, 'sample_id'].head(10).tolist()}")
    conflicts = (
        frame.groupby("normalized_text")["emotion"]
        .nunique()
        .reset_index(name="label_count")
        .query("label_count > 1")
    )
    if not conflicts.empty:
        raise ValueError(f"Texts with conflicting emotion labels: {len(conflicts)}")
    return frame


def assign_text_disjoint_splits(
    metadata: pd.DataFrame,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> pd.Series:
    frame = validate_text_groups(metadata)
    split_values = pd.Series(index=frame.index, dtype="object")
    rng = random.Random(seed)

    group_info = (
        frame.groupby("normalized_text")
        .agg(emotion=("emotion", "first"), rows=("sample_id", "count"))
        .reset_index()
    )
    for emotion, emotion_groups in group_info.groupby("emotion", sort=True):
        texts = sorted(emotion_groups["normalized_text"].tolist())
        rng.shuffle(texts)
        counts = split_counts(len(texts), split_ratios)
        start = 0
        for split_name in SPLIT_NAMES:
            stop = start + counts[split_name]
            selected = set(texts[start:stop])
            split_values.loc[frame["normalized_text"].isin(selected)] = split_name
            start = stop

    if split_values.isna().any():
        raise ValueError("Some rows were not assigned to a text-disjoint split.")
    return split_values


def validate_text_disjoint_splits(metadata: pd.DataFrame) -> None:
    frame = validate_text_groups(metadata)
    if "split" not in frame.columns:
        raise ValueError("Missing split column.")
    leakage = (
        frame.groupby("normalized_text")["split"]
        .nunique()
        .reset_index(name="split_count")
        .query("split_count > 1")
    )
    if not leakage.empty:
        raise ValueError(f"normalized_text values crossing splits: {len(leakage)}")


def build_deduplicated_text_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    frame = validate_text_groups(metadata)
    if "split" not in frame.columns:
        raise ValueError("Missing split column.")
    deduped = (
        frame.sort_values(["normalized_text", "sample_id"])
        .groupby("normalized_text", as_index=False)
        .first()
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    return deduped[OUTPUT_COLUMNS]


def write_split_outputs(metadata: pd.DataFrame, output_dir: Path, label_map: dict[str, int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    metadata[OUTPUT_COLUMNS].to_csv(output_dir / "metadata.csv", index=False, encoding="utf-8-sig")
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    compute_class_distribution(metadata).to_csv(
        output_dir / "class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    compute_split_class_distribution(metadata).to_csv(
        output_dir / "split_class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for split_name in SPLIT_NAMES:
        split_frame = metadata.loc[metadata["split"].eq(split_name), OUTPUT_COLUMNS].reset_index(drop=True)
        split_frame.to_csv(splits_dir / f"{split_name}.csv", index=False, encoding="utf-8-sig")


def write_quality_report(full_metadata: pd.DataFrame, deduped_metadata: pd.DataFrame, output_dir: Path) -> None:
    full_frame = validate_text_groups(full_metadata)
    group_sizes = full_frame.groupby("normalized_text").size()
    report = {
        "full_rows": int(len(full_metadata)),
        "unique_text_rows": int(len(deduped_metadata)),
        "duplicate_text_groups": int((group_sizes > 1).sum()),
        "duplicate_text_rows": int(group_sizes[group_sizes > 1].sum()),
        "split_counts_full": full_metadata["split"].value_counts().to_dict(),
        "split_counts_text_dedup": deduped_metadata["split"].value_counts().to_dict(),
        "speaker_overlap_full": {
            f"{left}_{right}": int(
                len(
                    set(full_metadata.loc[full_metadata["split"].eq(left), "speaker"])
                    & set(full_metadata.loc[full_metadata["split"].eq(right), "speaker"])
                )
            )
            for left, right in [("train", "val"), ("train", "test"), ("val", "test")]
        },
    }
    (output_dir / "split_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def create_text_disjoint_outputs(
    metadata_path: Path,
    label_map_path: Path,
    output_dir: Path,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(metadata_path)
    label_map = {str(key): int(value) for key, value in json.loads(label_map_path.read_text(encoding="utf-8")).items()}
    metadata = metadata.copy()
    metadata["split"] = assign_text_disjoint_splits(metadata, split_ratios=split_ratios, seed=seed)
    validate_text_disjoint_splits(metadata)
    full_metadata = metadata[OUTPUT_COLUMNS].reset_index(drop=True)
    text_dedup_metadata = build_deduplicated_text_metadata(full_metadata)

    full_dir = output_dir / "01_full_samples"
    dedup_dir = output_dir / "02_text_unique"
    write_split_outputs(full_metadata, full_dir, label_map)
    write_split_outputs(text_dedup_metadata, dedup_dir, label_map)
    write_quality_report(full_metadata, text_dedup_metadata, output_dir)
    return full_metadata, text_dedup_metadata


def main() -> int:
    args = parse_args()
    full, dedup = create_text_disjoint_outputs(
        metadata_path=args.metadata_path,
        label_map_path=args.label_map,
        output_dir=args.output_dir,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
    )
    print(f"Wrote full text-disjoint rows: {len(full)}")
    print(f"Wrote unique-text rows: {len(dedup)}")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
