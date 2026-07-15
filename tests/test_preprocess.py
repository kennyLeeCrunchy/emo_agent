"""
测试模块1预处理是否能生成稳定的数据索引、标签映射和数据划分。
Test whether Module 1 preprocessing creates stable indexes, label mappings, and data splits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.module1_preprocess.preprocess import preprocess_dataset
from modules.module1_preprocess.text_disjoint_split import (
    assign_text_disjoint_splits,
    build_deduplicated_text_metadata,
    validate_text_disjoint_splits,
)


def write_shard(path: Path, rows: list[dict[str, str]]) -> None:
    table = pa.table(
        {
            "audio": [b"fake-audio-bytes" for _ in rows],
            "text": [row["text"] for row in rows],
            "emotion": [row["emotion"] for row in rows],
            "speaker": [row["speaker"] for row in rows],
        }
    )
    pq.write_table(table, path)


def test_preprocess_dataset_writes_metadata_labels_and_splits(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    data_dir.mkdir()

    write_shard(
        data_dir / "train-00000-of-00002.parquet",
        [
            {"text": "angry one", "emotion": "angry", "speaker": "spk1"},
            {"text": "happy one", "emotion": "happy", "speaker": "spk2"},
            {"text": "sad one", "emotion": "sad", "speaker": "spk3"},
        ],
    )
    write_shard(
        data_dir / "train-00001-of-00002.parquet",
        [
            {"text": "neutral one", "emotion": "neutral", "speaker": "spk4"},
            {"text": "angry two", "emotion": "angry", "speaker": "spk1"},
        ],
    )

    preprocess_dataset(
        input_path=data_dir,
        output_dir=output_dir,
        split_by="speaker",
        split_ratios=(0.5, 0.25, 0.25),
        seed=123,
    )

    metadata = pd.read_csv(output_dir / "metadata.csv")
    label_map = json.loads((output_dir / "label_map.json").read_text(encoding="utf-8"))
    train = pd.read_csv(output_dir / "splits" / "train.csv")
    val = pd.read_csv(output_dir / "splits" / "val.csv")
    test = pd.read_csv(output_dir / "splits" / "test.csv")
    class_distribution = pd.read_csv(output_dir / "class_distribution.csv")
    split_distribution = pd.read_csv(output_dir / "split_class_distribution.csv")

    assert "audio" not in metadata.columns
    assert set(
        [
            "sample_id",
            "text",
            "emotion",
            "speaker",
            "label_id",
            "parquet_file",
            "row_id",
            "split",
        ]
    ).issubset(metadata.columns)
    assert metadata["sample_id"].is_unique
    assert label_map == {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}

    split_sample_ids = set(train["sample_id"]) | set(val["sample_id"]) | set(test["sample_id"])
    assert split_sample_ids == set(metadata["sample_id"])
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    assert set(class_distribution.columns) == {
        "emotion",
        "label_id",
        "count",
        "ratio",
    }
    assert class_distribution["count"].sum() == len(metadata)
    assert split_distribution["count"].sum() == len(metadata)


def test_sample_split_preserves_each_emotion_in_each_split(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    data_dir.mkdir()

    rows: list[dict[str, str]] = []
    for emotion in ["angry", "happy", "neutral"]:
        for idx in range(6):
            rows.append(
                {
                    "text": f"{emotion} {idx}",
                    "emotion": emotion,
                    "speaker": f"spk{idx % 3}",
                }
            )
    write_shard(data_dir / "train-00000-of-00001.parquet", rows)

    preprocess_dataset(
        input_path=data_dir,
        output_dir=output_dir,
        split_by="sample",
        split_ratios=(0.5, 0.25, 0.25),
        seed=123,
    )

    metadata = pd.read_csv(output_dir / "metadata.csv")
    split_distribution = pd.read_csv(output_dir / "split_class_distribution.csv")

    assert set(split_distribution["split"]) == {"train", "val", "test"}
    for split_name in ["train", "val", "test"]:
        emotions = set(metadata.loc[metadata["split"] == split_name, "emotion"])
        assert emotions == {"angry", "happy", "neutral"}


def test_text_disjoint_split_keeps_duplicate_texts_in_one_split() -> None:
    metadata = pd.DataFrame(
        {
            "sample_id": [f"id-{idx}" for idx in range(18)],
            "text": [
                "angry text a",
                "angry text a",
                "angry text b",
                "angry text b",
                "angry text c",
                "angry text c",
                "happy text a",
                "happy text a",
                "happy text b",
                "happy text b",
                "happy text c",
                "happy text c",
                "sad text a",
                "sad text a",
                "sad text b",
                "sad text b",
                "sad text c",
                "sad text c",
            ],
            "emotion": ["angry"] * 6 + ["happy"] * 6 + ["sad"] * 6,
            "speaker": [f"spk{idx % 3}" for idx in range(18)],
            "label_id": [0] * 6 + [1] * 6 + [2] * 6,
            "parquet_file": ["part.parquet"] * 18,
            "row_id": list(range(18)),
        }
    )

    split_values = assign_text_disjoint_splits(
        metadata,
        split_ratios=(1 / 3, 1 / 3, 1 / 3),
        seed=7,
    )
    result = metadata.assign(split=split_values)

    validate_text_disjoint_splits(result)
    assert set(result["split"]) == {"train", "val", "test"}
    for _, group in result.groupby("text"):
        assert group["split"].nunique() == 1


def test_deduplicated_text_metadata_keeps_one_representative_per_text() -> None:
    metadata = pd.DataFrame(
        {
            "sample_id": ["b", "a", "c", "d"],
            "text": ["same text", "same text", "other text", "other text"],
            "emotion": ["angry", "angry", "happy", "happy"],
            "speaker": ["spk2", "spk1", "spk3", "spk4"],
            "label_id": [0, 0, 1, 1],
            "parquet_file": ["part.parquet"] * 4,
            "row_id": [1, 0, 2, 3],
            "split": ["train", "train", "test", "test"],
        }
    )

    deduped = build_deduplicated_text_metadata(metadata)

    assert len(deduped) == 2
    assert set(deduped["text"]) == {"same text", "other text"}
    assert set(deduped["sample_id"]) == {"a", "c"}
    assert deduped["sample_id"].is_unique
