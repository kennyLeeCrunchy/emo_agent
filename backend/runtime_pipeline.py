"""Runtime audio-to-Agent evidence pipeline for the local backend."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import soundfile as sf

from modules.module2_ssl_audio.extract_ssl_features import WhisperExtractor
from modules.module3_text_features.extract_text_features import TextEmbeddingExtractor
from modules.module3_text_features.extract_whisper_asr import WhisperAsrTranscriber
from modules.module4_fusion.train_fusion_logistic import build_fusion_features, label_names_from_map
from modules.module5_emotion_curve.emotion_curve import CurveConfig, analyze_multimodal_emotion_curve
from modules.module6_keywords.keyword_analysis import KeywordAnalysisInput, analyze_keywords_and_reasons


# runtime_pipeline.py lives directly under <project>/backend, so parents[1]
# is the repository root. Using parents[2] silently pointed one directory too
# high and made every default model/artifact path invalid at runtime.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "models" / "pretrained" / "huggingface"
DEFAULT_ROBERTA_MODEL = PROJECT_ROOT / "models" / "pretrained" / "manual" / "chinese-roberta-wwm-ext"
DEFAULT_LABEL_MAP_PATH = PROJECT_ROOT / "modules" / "module1_preprocess" / "outputs" / "label_map.json"
DEFAULT_AUDIO_PIPELINE_PATH = (
    PROJECT_ROOT
    / "modules"
    / "module2_ssl_audio"
    / "outputs"
    / "whisper_text_unique"
    / "audio_logistic"
    / "model"
    / "whisper_logistic_pipeline.joblib"
)
DEFAULT_TEXT_PIPELINE_PATH = (
    PROJECT_ROOT
    / "modules"
    / "module3_text_features"
    / "outputs"
    / "text_asr_main"
    / "03_text_logistic_classifier"
    / "model"
    / "text_logistic_pipeline.joblib"
)
DEFAULT_FUSION_PIPELINE_PATH = (
    PROJECT_ROOT
    / "modules"
    / "module4_fusion"
    / "outputs"
    / "fusion_asr_main"
    / "01_fusion_logistic_classifier"
    / "model"
    / "fusion_logistic_pipeline.joblib"
)


@dataclass(frozen=True)
class PredictionResult:
    prediction: str
    confidence: float
    probabilities: dict[str, float]


class AudioBytesDecoder:
    supported_extensions = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".webm"}

    def __init__(self, target_sample_rate: int = 16000, ffmpeg_path: str | None = None) -> None:
        self.target_sample_rate = target_sample_rate
        self.ffmpeg_path = ffmpeg_path or find_ffmpeg_path()

    def decode(self, audio_bytes: bytes, filename: str | None = None, content_type: str | None = None) -> tuple[np.ndarray, int]:
        if not audio_bytes:
            raise ValueError("audio_bytes must not be empty.")

        suffix = _suffix_from_filename(filename)
        if suffix in {"", ".wav", ".flac"}:
            try:
                return self._decode_with_soundfile(audio_bytes)
            except Exception:
                if suffix in {"", ".wav", ".flac"} and not self.ffmpeg_path:
                    raise
        if not self.ffmpeg_path:
            raise RuntimeError("ffmpeg is required for compressed or browser-recorded audio formats.")
        return self._decode_with_ffmpeg(audio_bytes, suffix=suffix)

    def _decode_with_soundfile(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        import io

        waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        waveform = _mono(waveform)
        return waveform.astype(np.float32), int(sample_rate)

    def _decode_with_ffmpeg(self, audio_bytes: bytes, suffix: str) -> tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix=suffix or ".audio", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)
        try:
            command = [
                str(self.ffmpeg_path),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(tmp_path),
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                str(self.target_sample_rate),
                "pipe:1",
            ]
            result = subprocess.run(command, capture_output=True, check=False)
        finally:
            _remove_temp_file(tmp_path)

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to decode audio: {stderr}")
        samples = np.frombuffer(result.stdout, dtype=np.float32).copy()
        if samples.size == 0:
            raise RuntimeError("ffmpeg decoded an empty audio stream.")
        return samples, self.target_sample_rate


class EmotionBackendRuntime:
    def __init__(
        self,
        audio_extractor: Any,
        transcriber: Any,
        text_extractor: Any,
        audio_classifier: Any,
        text_classifier: Any,
        fusion_classifier: Any,
        label_names: list[str],
    ) -> None:
        self.audio_extractor = audio_extractor
        self.transcriber = transcriber
        self.text_extractor = text_extractor
        self.audio_classifier = audio_classifier
        self.text_classifier = text_classifier
        self.fusion_classifier = fusion_classifier
        self.label_names = label_names

    def analyze_waveform(self, waveform: Sequence[float], sample_rate: int) -> dict[str, Any]:
        waveform_array = np.asarray(waveform, dtype=np.float32).reshape(-1)
        audio_embedding = np.asarray(self.audio_extractor.extract(waveform_array, sample_rate), dtype=np.float32)
        asr_text = self.transcriber.transcribe(waveform_array, sample_rate)
        text_embedding = np.asarray(self.text_extractor.extract_batch([asr_text]), dtype=np.float32)[0]

        audio_result = predict_with_pipeline(self.audio_classifier, audio_embedding, self.label_names)
        text_result = predict_with_pipeline(self.text_classifier, text_embedding, self.label_names)
        fusion_features = build_fusion_features(audio_embedding.reshape(1, -1), text_embedding.reshape(1, -1))
        fusion_result = predict_with_pipeline(self.fusion_classifier, fusion_features[0], self.label_names)
        emotion_curve, curve_summary = self._analyze_emotion_curve(waveform_array, sample_rate, asr_text)
        emotion_change_points = curve_summary.get("emotion_change_points", [])

        keyword_result = analyze_keywords_and_reasons(
            KeywordAnalysisInput(
                asr_text=asr_text,
                text_prediction=text_result.prediction,
                fusion_prediction=fusion_result.prediction,
                fusion_confidence=fusion_result.confidence,
                emotion_change_points=emotion_change_points,
                segments=[],
            )
        )

        return {
            "asr_text": asr_text,
            "audio_prediction": audio_result.prediction,
            "text_prediction": text_result.prediction,
            "fusion_prediction": fusion_result.prediction,
            "audio_confidence": audio_result.confidence,
            "text_confidence": text_result.confidence,
            "fusion_confidence": fusion_result.confidence,
            "audio_probabilities": audio_result.probabilities,
            "text_probabilities": text_result.probabilities,
            "fusion_probabilities": fusion_result.probabilities,
            "emotion_curve": emotion_curve,
            "emotion_curve_summary": curve_summary,
            "keywords": keyword_result["keywords"],
            "source_snippets": keyword_result["source_snippets"],
            "possible_reasons": keyword_result["possible_reasons"],
            "emotion_change_points": emotion_change_points,
        }

    def _analyze_emotion_curve(self, waveform: np.ndarray, sample_rate: int, asr_text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        curve, summary = analyze_multimodal_emotion_curve(
            waveform=waveform,
            sample_rate=sample_rate,
            asr_text=asr_text,
            audio_extractor=self.audio_extractor,
            text_extractor=self.text_extractor,
            fusion_model=self.fusion_classifier,
            label_names=self.label_names,
            config=CurveConfig(window_seconds=2.0, step_seconds=2.0, text_mode="repeat_global"),
        )
        return _records_from_curve(curve), _json_ready(summary)


class FullAudioAnalyzer:
    def __init__(
        self,
        runtime: EmotionBackendRuntime,
        decoder: AudioBytesDecoder | None = None,
    ) -> None:
        self.runtime = runtime
        self.decoder = decoder or AudioBytesDecoder()

    def analyze(self, audio_bytes: bytes, filename: str | None = None, content_type: str | None = None) -> dict[str, Any]:
        waveform, sample_rate = self.decoder.decode(audio_bytes, filename=filename, content_type=content_type)
        result = self.runtime.analyze_waveform(waveform, sample_rate)
        result["metadata"] = {
            "filename": filename,
            "content_type": content_type,
            "sample_rate": sample_rate,
            "duration_seconds": float(len(waveform) / sample_rate) if sample_rate else None,
            "audio_bytes": len(audio_bytes),
        }
        return result


class LazyFullAudioAnalyzer:
    def __init__(self, device: str = "auto", local_files_only: bool = True) -> None:
        self.device = device
        self.local_files_only = local_files_only
        self._delegate: FullAudioAnalyzer | None = None

    def analyze(self, audio_bytes: bytes, filename: str | None = None, content_type: str | None = None) -> dict[str, Any]:
        if self._delegate is None:
            self._delegate = FullAudioAnalyzer(load_default_runtime(self.device, self.local_files_only))
        return self._delegate.analyze(audio_bytes, filename=filename, content_type=content_type)


def load_default_runtime(device: str = "auto", local_files_only: bool = True) -> EmotionBackendRuntime:
    import json

    label_map = json.loads(DEFAULT_LABEL_MAP_PATH.read_text(encoding="utf-8"))
    label_names = label_names_from_map(label_map)
    return EmotionBackendRuntime(
        audio_extractor=WhisperExtractor(
            model_name="openai/whisper-tiny",
            cache_dir=DEFAULT_CACHE_DIR,
            device=device,
            local_files_only=local_files_only,
        ),
        transcriber=WhisperAsrTranscriber(
            model_name="openai/whisper-tiny",
            cache_dir=DEFAULT_CACHE_DIR,
            device=device,
            local_files_only=local_files_only,
            max_new_tokens=96,
        ),
        text_extractor=TextEmbeddingExtractor(
            model_name=str(DEFAULT_ROBERTA_MODEL),
            cache_dir=None,
            device=device,
            max_length=128,
            local_files_only=True,
        ),
        audio_classifier=joblib.load(DEFAULT_AUDIO_PIPELINE_PATH),
        text_classifier=joblib.load(DEFAULT_TEXT_PIPELINE_PATH),
        fusion_classifier=joblib.load(DEFAULT_FUSION_PIPELINE_PATH),
        label_names=label_names,
    )


def predict_with_pipeline(pipeline: Any, feature: Sequence[float] | np.ndarray, label_names: list[str]) -> PredictionResult:
    feature_array = np.asarray(feature, dtype=np.float32)
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(1, -1)
    probabilities = np.asarray(pipeline.predict_proba(feature_array), dtype=np.float64)
    if probabilities.shape != (1, len(label_names)):
        raise ValueError(f"predict_proba shape mismatch: expected {(1, len(label_names))}, got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("predicted probabilities contain NaN or Inf values.")
    pred_index = int(probabilities[0].argmax())
    probs = {label: float(probabilities[0, idx]) for idx, label in enumerate(label_names)}
    return PredictionResult(
        prediction=label_names[pred_index],
        confidence=float(probabilities[0, pred_index]),
        probabilities=probs,
    )


def find_ffmpeg_path() -> str | None:
    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _records_from_curve(curve: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in curve.to_dict(orient="records"):
        records.append(_json_ready(row))
    return records


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    return value


def _suffix_from_filename(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def _remove_temp_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        pass
    except OSError as exc:
        if getattr(exc, "winerror", None) != 32:
            raise


def _mono(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 1:
        return waveform
    return waveform.mean(axis=1).astype(np.float32)
