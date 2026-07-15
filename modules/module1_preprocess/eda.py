"""
查看中文语音情绪数据集的字段、样本音频、类别分布和说话人分布。
Inspect fields, sample audio, emotion distribution, and speaker distribution for the Chinese speech emotion dataset.

运行示例：
Run from the project root:
    python -m modules.module1_preprocess.eda --input CSEMOTIONS/data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "CSEMOTIONS" / "data"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "modules" / "module1_preprocess" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore a local Parquet speech emotion dataset."
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
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=f"Directory for saved figures. Default: {DEFAULT_FIGURES_DIR}",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="Number of rows to print from the first Parquet file.",
    )
    parser.add_argument(
        "--top-speakers",
        type=int,
        default=30,
        help="Number of speakers to show in the speaker distribution figure.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    """Configure common Chinese fonts for Windows and other local systems."""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


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


def print_dataset_overview(parquet_files: list[Path]) -> tuple[int, list[str]]:
    print("\n========== Dataset Overview ==========")
    print(f"Parquet files: {len(parquet_files)}")
    for file_path in parquet_files:
        print(f"  - {file_path.name}")

    total_rows = 0
    columns: list[str] = []

    for idx, file_path in enumerate(parquet_files):
        parquet_file = pq.ParquetFile(file_path)
        total_rows += parquet_file.metadata.num_rows
        if idx == 0:
            columns = parquet_file.schema_arrow.names
            print("\nColumns:")
            for column in columns:
                print(f"  - {column}")

            print("\nParquet schema:")
            print(parquet_file.schema)

    print(f"\nTotal rows: {total_rows}")
    return total_rows, columns


def read_sample(parquet_files: list[Path], sample_rows: int) -> pd.DataFrame:
    """Read a small sample including audio without loading every shard."""
    first_file = pq.ParquetFile(parquet_files[0])
    table = first_file.read_row_group(0).slice(0, max(sample_rows, 1))
    return table.to_pandas()


def print_sample(sample_df: pd.DataFrame) -> None:
    print("\n========== Field Types ==========")
    print(sample_df.dtypes)

    print("\n========== Head Samples ==========")
    preview_columns = [col for col in ["text", "emotion", "speaker"] if col in sample_df]
    if preview_columns:
        print(sample_df[preview_columns].head().to_string(index=False))
    else:
        print(sample_df.head().to_string(index=False))


def read_label_columns(parquet_files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for file_path in parquet_files:
        parquet_file = pq.ParquetFile(file_path)
        available_columns = set(parquet_file.schema_arrow.names)
        needed_columns = [
            column for column in ["emotion", "speaker"] if column in available_columns
        ]

        if not needed_columns:
            continue

        frame = pd.read_parquet(file_path, columns=needed_columns)
        frames.append(frame)

    if not frames:
        raise ValueError("Neither 'emotion' nor 'speaker' column was found.")

    return pd.concat(frames, ignore_index=True)


def print_distribution(name: str, series: pd.Series) -> pd.Series:
    print(f"\n========== {name} Distribution ==========")
    counts = series.fillna("<NA>").astype(str).value_counts()
    print(counts.to_string())
    return counts


def describe_audio_value(audio_value: Any, indent: int = 0) -> None:
    prefix = " " * indent
    print(f"{prefix}type: {type(audio_value)}")

    if isinstance(audio_value, dict):
        print(f"{prefix}keys: {list(audio_value.keys())}")
        for key, value in audio_value.items():
            if isinstance(value, (bytes, bytearray)):
                print(f"{prefix}- {key}: {type(value).__name__}, {len(value)} bytes")
            else:
                print(f"{prefix}- {key}: {type(value).__name__}, {repr(value)[:120]}")
        return

    if isinstance(audio_value, (bytes, bytearray)):
        print(f"{prefix}bytes length: {len(audio_value)}")
        return

    if hasattr(audio_value, "shape"):
        print(f"{prefix}shape: {getattr(audio_value, 'shape', None)}")
        print(f"{prefix}dtype: {getattr(audio_value, 'dtype', None)}")
        return

    print(f"{prefix}repr: {repr(audio_value)[:300]}")


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


def inspect_audio_sample(sample_df: pd.DataFrame) -> None:
    print("\n========== Audio Sample ==========")
    if "audio" not in sample_df.columns or sample_df.empty:
        print("No audio column found or sample dataframe is empty.")
        return

    audio_value = sample_df["audio"].iloc[0]
    describe_audio_value(audio_value)

    audio_bytes = extract_audio_bytes(audio_value)
    if audio_bytes is None:
        print("No bytes found in the first audio sample.")
        return

    print(f"audio bytes length: {len(audio_bytes)}")


def plot_bar(
    counts: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    rotate_xticks: bool = True,
) -> None:
    fig_width = max(8, min(18, len(counts) * 0.55))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    counts.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    if rotate_xticks:
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    else:
        ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def save_figures(
    emotion_counts: pd.Series,
    speaker_counts: pd.Series,
    figures_dir: Path,
    top_speakers: int,
) -> None:
    print("\n========== Saving Figures ==========")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_bar(
        emotion_counts,
        title="情绪类别分布",
        xlabel="情绪类别",
        ylabel="样本数量",
        output_path=figures_dir / "emotion_distribution.png",
        rotate_xticks=False,
    )

    speaker_plot_counts = speaker_counts.head(max(top_speakers, 1))
    speaker_title = "Speaker 样本数量分布"
    if len(speaker_counts) > len(speaker_plot_counts):
        speaker_title += f" Top {len(speaker_plot_counts)}"

    plot_bar(
        speaker_plot_counts,
        title=speaker_title,
        xlabel="Speaker",
        ylabel="样本数量",
        output_path=figures_dir / "speaker_distribution.png",
        rotate_xticks=True,
    )


def main() -> int:
    args = parse_args()
    configure_matplotlib()

    parquet_files = resolve_parquet_files(args.input)
    _, columns = print_dataset_overview(parquet_files)

    if not {"emotion", "speaker"}.intersection(columns):
        print("Warning: neither 'emotion' nor 'speaker' was found in the schema.")

    sample_df = read_sample(parquet_files, args.sample_rows)
    print_sample(sample_df)
    inspect_audio_sample(sample_df)

    label_df = read_label_columns(parquet_files)
    emotion_counts = print_distribution("Emotion", label_df["emotion"])
    speaker_counts = print_distribution("Speaker", label_df["speaker"])

    save_figures(
        emotion_counts=emotion_counts,
        speaker_counts=speaker_counts,
        figures_dir=args.figures_dir,
        top_speakers=args.top_speakers,
    )

    print("\nEDA finished.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
