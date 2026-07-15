"""
构建 CSEMOTIONS 数据集的元数据索引、标签映射和 train/val/test 划分。
Build metadata indexes, label mappings, and train/val/test splits for CSEMOTIONS.

运行示例：
Run from the project root:
    python -m modules.module1_preprocess.preprocess --input CSEMOTIONS/data
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "CSEMOTIONS" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs"
REQUIRED_COLUMNS = ["text", "emotion", "speaker"]
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reusable metadata, label map, and dataset splits."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Parquet file or a directory containing Parquet shards. "
            f"Default: {DEFAULT_INPUT}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--split-by",
        choices=["speaker", "sample"],
        default="sample",
        help=(
            "Split by speaker identity or stratified individual sample. "
            "Default: sample"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_parquet_files(input_path: Path) -> list[Path]:
    input_path = input_path.expanduser().resolve()

    if input_path.is_file():
        if input_path.suffix.lower() != ".parquet":
            raise ValueError(f"Input file is not a .parquet file: {input_path}")
        return [input_path]

    if input_path.is_dir():
        parquet_files = sorted(input_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in: {input_path}")
        return parquet_files

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def validate_schema(parquet_file: pq.ParquetFile, file_path: Path) -> None:
    columns = set(parquet_file.schema_arrow.names)
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"{file_path} is missing required columns: {missing}")


def read_metadata_frame(file_path: Path) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(file_path)
    validate_schema(parquet_file, file_path)

    frame = parquet_file.read(columns=REQUIRED_COLUMNS).to_pandas()
    frame.insert(0, "sample_id", [f"{file_path.stem}:{idx}" for idx in range(len(frame))])
    frame["parquet_file"] = file_path.name
    frame["row_id"] = range(len(frame))
    return frame


def build_metadata(parquet_files: list[Path]) -> pd.DataFrame:
    frames = [read_metadata_frame(file_path) for file_path in parquet_files]
    if not frames:
        raise ValueError("No metadata could be read from the Parquet input.")

    metadata = pd.concat(frames, ignore_index=True)
    for column in REQUIRED_COLUMNS:
        metadata[column] = metadata[column].fillna("").astype(str)
    return metadata


def create_label_map(emotions: pd.Series) -> dict[str, int]:
    labels = sorted(label for label in emotions.fillna("").astype(str).unique() if label)
    if not labels:
        raise ValueError("No emotion labels were found.")
    return {label: idx for idx, label in enumerate(labels)}


def validate_split_ratios(split_ratios: tuple[float, float, float]) -> None:
    if any(ratio < 0 for ratio in split_ratios):
        raise ValueError(f"Split ratios must be non-negative: {split_ratios}")

    total = sum(split_ratios)
    if total <= 0:
        raise ValueError(f"At least one split ratio must be positive: {split_ratios}")


def split_counts(total: int, split_ratios: tuple[float, float, float]) -> dict[str, int]:
    validate_split_ratios(split_ratios)
    if total == 0:
        return {name: 0 for name in SPLIT_NAMES}

    ratio_total = sum(split_ratios)
    normalized = [ratio / ratio_total for ratio in split_ratios]
    counts = [int(total * ratio) for ratio in normalized]

    remainder = total - sum(counts)
    fractional_order = sorted(
        range(len(normalized)),
        key=lambda idx: (total * normalized[idx]) - counts[idx],
        reverse=True,
    )
    for idx in fractional_order[:remainder]:
        counts[idx] += 1

    if total >= len(SPLIT_NAMES):
        for idx, ratio in enumerate(normalized):
            if ratio > 0 and counts[idx] == 0:
                donor = max(range(len(counts)), key=lambda donor_idx: counts[donor_idx])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[idx] += 1

    return dict(zip(SPLIT_NAMES, counts))


def assign_splits(
    metadata: pd.DataFrame,
    split_by: str,
    split_ratios: tuple[float, float, float],
    seed: int,
) -> pd.Series:
    if split_by == "speaker":
        units = sorted(metadata["speaker"].unique())
        return assign_unit_splits(metadata, units, "speaker", split_ratios, seed)

    if split_by == "sample":
        return assign_stratified_sample_splits(metadata, split_ratios, seed)

    raise ValueError(f"Unsupported split_by value: {split_by}")


def assign_unit_splits(
    metadata: pd.DataFrame,
    units: list[str],
    key_column: str,
    split_ratios: tuple[float, float, float],
    seed: int,
) -> pd.Series:
    rng = random.Random(seed)
    shuffled_units = list(units)
    rng.shuffle(shuffled_units)

    counts = split_counts(len(shuffled_units), split_ratios)
    unit_to_split: dict[str, str] = {}
    start = 0
    for split_name in SPLIT_NAMES:
        stop = start + counts[split_name]
        for unit in shuffled_units[start:stop]:
            unit_to_split[unit] = split_name
        start = stop

    return metadata[key_column].map(unit_to_split)


def assign_stratified_sample_splits(
    metadata: pd.DataFrame,
    split_ratios: tuple[float, float, float],
    seed: int,
) -> pd.Series:
    split_values = pd.Series(index=metadata.index, dtype="object")
    rng = random.Random(seed)

    for _, group in metadata.groupby("emotion", sort=True):
        sample_ids = sorted(group["sample_id"].tolist())
        rng.shuffle(sample_ids)

        counts = split_counts(len(sample_ids), split_ratios)
        start = 0
        for split_name in SPLIT_NAMES:
            stop = start + counts[split_name]
            selected = sample_ids[start:stop]
            split_values.loc[metadata["sample_id"].isin(selected)] = split_name
            start = stop

    return split_values


def compute_class_distribution(metadata: pd.DataFrame) -> pd.DataFrame:
    counts = metadata.groupby(["emotion", "label_id"]).size().reset_index(name="count")
    counts = counts.sort_values(["label_id", "emotion"]).reset_index(drop=True)
    counts["ratio"] = counts["count"] / len(metadata)
    return counts[["emotion", "label_id", "count", "ratio"]]


def compute_split_class_distribution(metadata: pd.DataFrame) -> pd.DataFrame:
    counts = (
        metadata.groupby(["split", "emotion", "label_id"])
        .size()
        .reset_index(name="count")
    )
    split_totals = counts.groupby("split")["count"].transform("sum")
    counts["ratio"] = counts["count"] / split_totals
    counts = counts.sort_values(["split", "label_id", "emotion"]).reset_index(drop=True)
    return counts[["split", "emotion", "label_id", "count", "ratio"]]


def write_outputs(metadata: pd.DataFrame, label_map: dict[str, int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(output_dir / "metadata.csv", index=False, encoding="utf-8-sig")
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    compute_class_distribution(metadata).to_csv(
        output_dir / "class_distribution.csv", index=False, encoding="utf-8-sig"
    )
    compute_split_class_distribution(metadata).to_csv(
        output_dir / "split_class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for split_name in SPLIT_NAMES:
        split_frame = metadata[metadata["split"] == split_name].reset_index(drop=True)
        split_frame.to_csv(
            splits_dir / f"{split_name}.csv", index=False, encoding="utf-8-sig"
        )


def preprocess_dataset(
    input_path: Path,
    output_dir: Path,
    split_by: str = "speaker",
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> pd.DataFrame:
    parquet_files = resolve_parquet_files(input_path)
    metadata = build_metadata(parquet_files)

    label_map = create_label_map(metadata["emotion"])
    metadata["label_id"] = metadata["emotion"].map(label_map)
    metadata["split"] = assign_splits(metadata, split_by, split_ratios, seed)
    if metadata["split"].isna().any():
        raise ValueError("Some rows were not assigned to a split.")

    output_columns = [
        "sample_id",
        "text",
        "emotion",
        "speaker",
        "label_id",
        "parquet_file",
        "row_id",
        "split",
    ]
    metadata = metadata[output_columns]
    write_outputs(metadata, label_map, output_dir)
    return metadata


def main() -> int:
    args = parse_args()
    split_ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    metadata = preprocess_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        split_by=args.split_by,
        split_ratios=split_ratios,
        seed=args.seed,
    )

    print(f"Wrote metadata rows: {len(metadata)}")
    print(f"Metadata: {args.output_dir / 'metadata.csv'}")
    print(f"Label map: {args.output_dir / 'label_map.json'}")
    print(f"Splits: {args.output_dir / 'splits'}")
    print(f"Class distribution: {args.output_dir / 'class_distribution.csv'}")
    print(
        f"Split class distribution: "
        f"{args.output_dir / 'split_class_distribution.csv'}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
