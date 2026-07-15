from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.module1_preprocess.eda import inspect_audio_sample


def test_inspect_audio_sample_does_not_write_sample_audio(tmp_path: Path) -> None:
    sample_df = pd.DataFrame(
        {
            "audio": [b"fake-wav-bytes"],
            "text": ["sample text"],
            "emotion": ["surprise"],
            "speaker": ["female001"],
        }
    )

    inspect_audio_sample(sample_df)

    assert not (tmp_path / "sample_audio.wav").exists()
