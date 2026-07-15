from __future__ import annotations

import csv
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.module1_preprocess.sample_audio import (
    export_emotion_audio_samples,
    export_emotion_speaker_audio_samples,
    export_speaker_audio_samples,
)


def write_audio_shard(path: Path, rows: list[dict[str, str]]) -> None:
    table = pa.table(
        {
            "audio": [row["audio"] for row in rows],
            "text": [row["text"] for row in rows],
            "emotion": [row["emotion"] for row in rows],
            "speaker": [row["speaker"] for row in rows],
        }
    )
    pq.write_table(table, path, row_group_size=2)


def test_export_emotion_audio_samples_writes_n_wavs_per_emotion(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    sample_dir = tmp_path / "sample"
    data_dir.mkdir()
    write_audio_shard(
        data_dir / "train-00000-of-00001.parquet",
        [
            {
                "audio": f"{emotion}-{idx}".encode(),
                "text": f"{emotion} text {idx}",
                "emotion": emotion,
                "speaker": f"speaker{idx}",
            }
            for emotion in ["angry", "happy"]
            for idx in range(3)
        ],
    )

    exported = export_emotion_audio_samples(
        input_path=data_dir,
        output_dir=sample_dir,
        samples_per_emotion=2,
    )

    assert exported == {"angry": 2, "happy": 2}
    for emotion in ["angry", "happy"]:
        wav_files = sorted((sample_dir / emotion).glob("*.wav"))
        assert len(wav_files) == 2
        assert all(path.read_bytes().startswith(emotion.encode()) for path in wav_files)

        manifest_path = sample_dir / emotion / "manifest.csv"
        rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig")))
        assert len(rows) == 2
        assert {row["emotion"] for row in rows} == {emotion}
        assert all((sample_dir / emotion / Path(row["audio_file"]).name).exists() for row in rows)


def test_export_speaker_audio_samples_writes_one_wav_per_speaker(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    sample_dir = tmp_path / "sample"
    data_dir.mkdir()
    write_audio_shard(
        data_dir / "train-00000-of-00001.parquet",
        [
            {
                "audio": f"{speaker}-{idx}".encode(),
                "text": f"{speaker} text {idx}",
                "emotion": "happy",
                "speaker": speaker,
            }
            for speaker in ["female001", "male001", "male002"]
            for idx in range(2)
        ],
    )

    exported = export_speaker_audio_samples(
        input_path=data_dir,
        output_dir=sample_dir,
        samples_per_speaker=1,
    )

    assert exported == {"female001": 1, "male001": 1, "male002": 1}
    for speaker in ["female001", "male001", "male002"]:
        wav_files = sorted((sample_dir / speaker).glob("*.wav"))
        assert len(wav_files) == 1
        assert wav_files[0].read_bytes().startswith(speaker.encode())

        manifest_path = sample_dir / speaker / "manifest.csv"
        rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig")))
        assert len(rows) == 1
        assert rows[0]["speaker"] == speaker


def test_export_emotion_speaker_audio_samples_writes_one_wav_per_speaker_per_emotion(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    sample_dir = tmp_path / "sample"
    data_dir.mkdir()
    write_audio_shard(
        data_dir / "train-00000-of-00001.parquet",
        [
            {
                "audio": f"{emotion}-{speaker}-{idx}".encode(),
                "text": f"{emotion} {speaker} text {idx}",
                "emotion": emotion,
                "speaker": speaker,
            }
            for emotion in ["angry", "happy"]
            for speaker in ["female001", "male001"]
            for idx in range(2)
        ],
    )

    exported = export_emotion_speaker_audio_samples(
        input_path=data_dir,
        output_dir=sample_dir,
    )

    assert exported == {"angry": 2, "happy": 2}
    for emotion in ["angry", "happy"]:
        wav_files = sorted((sample_dir / emotion).glob("*.wav"))
        assert len(wav_files) == 2
        assert all(path.read_bytes().startswith(emotion.encode()) for path in wav_files)

        manifest_path = sample_dir / emotion / "manifest.csv"
        rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig")))
        assert len(rows) == 2
        assert {row["emotion"] for row in rows} == {emotion}
        assert {row["speaker"] for row in rows} == {"female001", "male001"}
