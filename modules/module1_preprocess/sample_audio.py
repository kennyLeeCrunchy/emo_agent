"""
Export small audio samples for manual listening.

Run from the project root:
    python -m modules.module1_preprocess.sample_audio --input CSEMOTIONS/data
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "CSEMOTIONS" / "data"
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "modules" / "module1_preprocess" / "sample"
REQUIRED_COLUMNS = ["audio", "text", "emotion", "speaker"]
MANIFEST_COLUMNS = [
    "audio_file",
    "emotion",
    "speaker",
    "text",
    "parquet_file",
    "row_id",
    "row_group",
    "bytes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export wav samples by emotion, speaker, or emotion-speaker pairs."
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
        default=DEFAULT_SAMPLE_DIR,
        help=f"Directory for exported samples. Default: {DEFAULT_SAMPLE_DIR}",
    )
    parser.add_argument(
        "--group-by",
        choices=["emotion_speaker", "speaker", "emotion"],
        default="emotion_speaker",
        help=(
            "Sampling layout. emotion_speaker writes emotion folders with one "
            "sample per speaker. Default: emotion_speaker"
        ),
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=1,
        help="Number of wav files to export for each group. Default: 1",
    )
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


def extract_audio_bytes(audio_value: Any) -> bytes | None:
    if isinstance(audio_value, bytes):
        return audio_value

    if isinstance(audio_value, bytearray):
        return bytes(audio_value)

    if isinstance(audio_value, dict):
        for key in ["bytes", "data", "audio"]:
            value = audio_value.get(key)
            if isinstance(value, bytes):
                return value
            if isinstance(value, bytearray):
                return bytes(value)

    return None


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "unknown"


def collect_sample_rows(
    parquet_files: list[Path],
    group_by: str,
    samples_per_group: int,
) -> dict[str, list[dict[str, Any]]]:
    if group_by not in {"speaker", "emotion"}:
        raise ValueError(f"Unsupported group_by value: {group_by}")

    if samples_per_group <= 0:
        raise ValueError("samples_per_group must be positive.")

    selected: dict[str, list[dict[str, Any]]] = {}

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        validate_schema(parquet_file, parquet_path)
        row_offset = 0

        for row_group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(
                row_group, columns=["text", "emotion", "speaker"]
            )
            frame = table.to_pandas()

            for local_idx, row in frame.iterrows():
                group_value = str(row[group_by])
                bucket = selected.setdefault(group_value, [])
                if len(bucket) >= samples_per_group:
                    continue

                bucket.append(
                    {
                        "parquet_path": parquet_path,
                        "parquet_file": parquet_path.name,
                        "row_group": row_group,
                        "row_id": row_offset + int(local_idx),
                        "local_idx": int(local_idx),
                        "text": str(row["text"]),
                        "emotion": str(row["emotion"]),
                        "speaker": str(row["speaker"]),
                    }
                )

            row_offset += parquet_file.metadata.row_group(row_group).num_rows

    missing = {
        group_value: len(rows)
        for group_value, rows in selected.items()
        if len(rows) < samples_per_group
    }
    if missing:
        raise ValueError(
            f"Not enough samples for every {group_by}; requested "
            f"{samples_per_group}, found {missing}"
        )

    return selected


def collect_emotion_speaker_rows(
    parquet_files: list[Path],
    samples_per_speaker: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    if samples_per_speaker <= 0:
        raise ValueError("samples_per_speaker must be positive.")

    selected: dict[str, list[dict[str, Any]]] = {}
    seen_counts: dict[tuple[str, str], int] = {}

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        validate_schema(parquet_file, parquet_path)
        row_offset = 0

        for row_group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(
                row_group, columns=["text", "emotion", "speaker"]
            )
            frame = table.to_pandas()

            for local_idx, row in frame.iterrows():
                emotion = str(row["emotion"])
                speaker = str(row["speaker"])
                key = (emotion, speaker)
                if seen_counts.get(key, 0) >= samples_per_speaker:
                    continue

                selected.setdefault(emotion, []).append(
                    {
                        "parquet_path": parquet_path,
                        "parquet_file": parquet_path.name,
                        "row_group": row_group,
                        "row_id": row_offset + int(local_idx),
                        "local_idx": int(local_idx),
                        "text": str(row["text"]),
                        "emotion": emotion,
                        "speaker": speaker,
                    }
                )
                seen_counts[key] = seen_counts.get(key, 0) + 1

            row_offset += parquet_file.metadata.row_group(row_group).num_rows

    return selected


def write_selected_samples(
    selected: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    group_by: str,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, int] = {}

    for group_value in sorted(selected):
        group_dir = output_dir / safe_name(group_value)
        group_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict[str, Any]] = []

        for sample_idx, item in enumerate(selected[group_value], start=1):
            parquet_file = pq.ParquetFile(item["parquet_path"])
            table = parquet_file.read_row_group(item["row_group"], columns=["audio"])
            audio_value = table.to_pandas().iloc[item["local_idx"]]["audio"]
            audio_bytes = extract_audio_bytes(audio_value)
            if audio_bytes is None:
                raise ValueError(
                    f"No audio bytes found for {item['parquet_file']} "
                    f"row {item['row_id']}"
                )

            prefix = safe_name(item["speaker"]) if group_by == "emotion_speaker" else safe_name(group_value)
            filename = f"{prefix}_{sample_idx:02d}_{item['parquet_path'].stem}_row{item['row_id']}.wav"
            audio_path = group_dir / filename
            audio_path.write_bytes(audio_bytes)

            manifest_rows.append(
                {
                    "audio_file": audio_path.as_posix(),
                    "emotion": item["emotion"],
                    "speaker": item["speaker"],
                    "text": item["text"],
                    "parquet_file": item["parquet_file"],
                    "row_id": item["row_id"],
                    "row_group": item["row_group"],
                    "bytes": len(audio_bytes),
                }
            )

        manifest_path = group_dir / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(manifest_rows)

        exported[group_value] = len(manifest_rows)

    return exported


def export_audio_samples_by_group(
    input_path: Path,
    output_dir: Path,
    group_by: str,
    samples_per_group: int,
) -> dict[str, int]:
    parquet_files = resolve_parquet_files(input_path)
    if group_by == "emotion_speaker":
        selected = collect_emotion_speaker_rows(parquet_files, samples_per_group)
        return write_selected_samples(selected, output_dir, group_by)

    selected = collect_sample_rows(parquet_files, group_by, samples_per_group)
    return write_selected_samples(selected, output_dir, group_by)


def export_emotion_audio_samples(
    input_path: Path,
    output_dir: Path,
    samples_per_emotion: int = 5,
) -> dict[str, int]:
    return export_audio_samples_by_group(
        input_path=input_path,
        output_dir=output_dir,
        group_by="emotion",
        samples_per_group=samples_per_emotion,
    )


def export_speaker_audio_samples(
    input_path: Path,
    output_dir: Path,
    samples_per_speaker: int = 1,
) -> dict[str, int]:
    return export_audio_samples_by_group(
        input_path=input_path,
        output_dir=output_dir,
        group_by="speaker",
        samples_per_group=samples_per_speaker,
    )


def export_emotion_speaker_audio_samples(
    input_path: Path,
    output_dir: Path,
    samples_per_speaker: int = 1,
) -> dict[str, int]:
    return export_audio_samples_by_group(
        input_path=input_path,
        output_dir=output_dir,
        group_by="emotion_speaker",
        samples_per_group=samples_per_speaker,
    )


def main() -> int:
    args = parse_args()
    exported = export_audio_samples_by_group(
        input_path=args.input,
        output_dir=args.output_dir,
        group_by=args.group_by,
        samples_per_group=args.samples_per_group,
    )

    print(f"Exported samples to: {args.output_dir}")
    for emotion, count in sorted(exported.items()):
        print(f"  {emotion}: {count}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
