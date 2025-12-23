"""文字起こしエンジンモジュール

stable-ts + faster-whisper を使用した音声文字起こしを行う。
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voice_log.config import Config, WhisperConfig
from voice_log.hallucination import HallucinationDetector
from voice_log.logger import get_logger

logger = get_logger("transcribe")


@dataclass
class TranscriptionSegment:
    """文字起こしセグメント"""

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """文字起こし結果"""

    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str = "ja"
    duration_sec: float = 0.0
    processing_time_sec: float = 0.0
    device: str = ""
    compute_type: str = ""
    hallucination_issues: int = 0


def _detect_device(config: WhisperConfig) -> tuple[str, str]:
    """デバイスと計算タイプを判定する

    Args:
        config: Whisper設定

    Returns:
        tuple[str, str]: (device, compute_type)
    """
    device = config.device
    compute_type = config.compute_type

    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                logger.info("CUDA利用可能: GPUを使用します")
            else:
                device = "cpu"
                logger.info("CUDA利用不可: CPUを使用します")
        except ImportError:
            device = "cpu"
            logger.warning("torchがインポートできません: CPUを使用します")

    if compute_type == "auto":
        if device == "cuda":
            compute_type = "float16"
        else:
            compute_type = "int8"

    return device, compute_type


class TranscriptionEngine:
    """文字起こしエンジン"""

    def __init__(self, config: Config):
        """
        Args:
            config: アプリケーション設定
        """
        self.config = config
        self.whisper_config = config.whisper
        self.vad_config = config.vad
        self.stable_ts_config = config.stable_ts
        self.hallucination_config = config.hallucination

        self._model = None
        self._device = ""
        self._compute_type = ""

    def _load_model(self) -> Any:
        """モデルをロードする"""
        if self._model is not None:
            return self._model

        self._device, self._compute_type = _detect_device(self.whisper_config)

        logger.info(
            f"モデルをロード中: {self.whisper_config.model} "
            f"(device={self._device}, compute_type={self._compute_type})"
        )

        if self.stable_ts_config.enabled:
            import stable_whisper

            self._model = stable_whisper.load_faster_whisper(
                self.whisper_config.model,
                device=self._device,
                compute_type=self._compute_type,
            )
        else:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.whisper_config.model,
                device=self._device,
                compute_type=self._compute_type,
            )

        logger.info("モデルのロードが完了しました")
        return self._model

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """音声を文字起こしする

        Args:
            audio_path: 音声ファイルパス

        Returns:
            TranscriptionResult: 文字起こし結果
        """
        start_time = time.time()

        model = self._load_model()

        # 音声長を取得
        from voice_log.audio_io import get_audio_duration

        duration_sec = get_audio_duration(audio_path)

        logger.info(f"文字起こし開始: {audio_path} ({duration_sec:.1f}秒)")

        segments_list = []

        if self.stable_ts_config.enabled:
            result = self._transcribe_with_stable_ts(model, audio_path)
        else:
            result = self._transcribe_with_faster_whisper(model, audio_path)

        text = result["text"]
        segments_list = result["segments"]

        # ハルシネーション検知・処理
        hallucination_issues = 0
        if self.hallucination_config.enabled:
            detector = HallucinationDetector(
                max_token_repeat=self.hallucination_config.max_consecutive_token_repeat,
                max_line_repeat=self.hallucination_config.max_consecutive_line_repeat,
                ngram_size=self.hallucination_config.ngram_size,
                ngram_threshold=self.hallucination_config.ngram_repeat_threshold,
                action=self.hallucination_config.action,
            )

            analysis = detector.analyze(text)
            if analysis.has_issues:
                hallucination_issues = len(analysis.issues)
                for issue in analysis.issues:
                    logger.warning(f"反復検知: {issue.details}")

                if self.hallucination_config.action == "trim":
                    text = detector.process(text)
                    logger.info("反復をトリムしました")

        processing_time = time.time() - start_time
        rtf = processing_time / duration_sec if duration_sec > 0 else 0

        logger.info(f"文字起こし完了: {processing_time:.1f}秒 (RTF: {rtf:.2f})")

        return TranscriptionResult(
            text=text,
            segments=segments_list,
            language=self.whisper_config.language,
            duration_sec=duration_sec,
            processing_time_sec=processing_time,
            device=self._device,
            compute_type=self._compute_type,
            hallucination_issues=hallucination_issues,
        )

    def _transcribe_with_stable_ts(
        self, model: Any, audio_path: Path
    ) -> dict[str, Any]:
        """stable-tsで文字起こし"""
        result = model.transcribe(
            str(audio_path),
            language=self.whisper_config.language,
            beam_size=self.whisper_config.beam_size,
            temperature=self.whisper_config.temperature,
            condition_on_previous_text=self.whisper_config.condition_on_previous_text,
            vad_filter=self.vad_config.enabled,
            suppress_silence=self.stable_ts_config.suppress_silence,
        )

        segments = []
        for segment in result.segments:
            segments.append(
                TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                )
            )

        return {
            "text": result.text,
            "segments": segments,
        }

    def _transcribe_with_faster_whisper(
        self, model: Any, audio_path: Path
    ) -> dict[str, Any]:
        """faster-whisperで文字起こし"""
        segments_gen, info = model.transcribe(
            str(audio_path),
            language=self.whisper_config.language,
            beam_size=self.whisper_config.beam_size,
            temperature=self.whisper_config.temperature,
            condition_on_previous_text=self.whisper_config.condition_on_previous_text,
            vad_filter=self.vad_config.enabled,
        )

        segments = []
        text_parts = []

        for segment in segments_gen:
            segments.append(
                TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                )
            )
            text_parts.append(segment.text.strip())

        return {
            "text": " ".join(text_parts),
            "segments": segments,
        }


def transcribe_audio(audio_path: Path, config: Config) -> TranscriptionResult:
    """音声を文字起こしする（簡易関数）

    Args:
        audio_path: 音声ファイルパス
        config: 設定

    Returns:
        TranscriptionResult: 文字起こし結果
    """
    engine = TranscriptionEngine(config)
    return engine.transcribe(audio_path)
