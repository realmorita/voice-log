"""設定管理モジュール

config.yaml の読み込み・検証・デフォルト値管理を行う。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioConfig:
    """音声入力設定"""

    sample_rate: int = 16000
    channels: int = 1
    device_id: int | None = None
    dtype: str = "float32"


@dataclass
class InputConfig:
    """入力設定"""

    tmp_dir: str = ".tmp"
    accept_ext: list[str] = field(default_factory=lambda: ["wav", "mp3", "m4a", "flac"])


@dataclass
class VadConfig:
    """VAD設定"""

    enabled: bool = True
    min_silence_duration_ms: int = 500


@dataclass
class WhisperConfig:
    """Whisper設定"""

    model: str = "large-v3"
    language: str = "ja"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 3
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    word_timestamps_internal: bool = False
    batch_size: int = 0


@dataclass
class StableTsConfig:
    """stable-ts設定"""

    enabled: bool = True
    suppress_silence: bool = True
    regroup: bool = True


@dataclass
class HallucinationConfig:
    """反復検知設定"""

    enabled: bool = True
    max_consecutive_token_repeat: int = 8
    max_consecutive_line_repeat: int = 3
    ngram_size: int = 4
    ngram_repeat_threshold: int = 6
    action: str = "trim"


@dataclass
class LlmConfig:
    """LLM設定"""

    enabled: bool = True
    provider: str = "openai_compatible"
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen2.5:14b"
    temperature: float = 0.2
    max_output_tokens: int | None = None
    timeout_sec: int = 120
    prompt_mode: str = "minutes"


@dataclass
class OutputConfig:
    """出力設定"""

    out_dir: str = "outputs"
    formats_transcript: list[str] = field(default_factory=lambda: ["md", "txt"])
    formats_summary: list[str] = field(default_factory=lambda: ["md", "txt"])
    naming: str = "{date}_{time}_{stem}"


@dataclass
class LoggingConfig:
    """ロギング設定"""

    level: str = "INFO"
    file: str = "logs/app.log"
    faster_whisper_debug: bool = False


@dataclass
class Config:
    """アプリケーション設定"""

    audio: AudioConfig = field(default_factory=AudioConfig)
    input: InputConfig = field(default_factory=InputConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    stable_ts: StableTsConfig = field(default_factory=StableTsConfig)
    hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def reload(self, config_path: Path) -> None:
        """設定ファイルを再読み込みする"""
        new_config = load_config(config_path)
        self.audio = new_config.audio
        self.input = new_config.input
        self.vad = new_config.vad
        self.whisper = new_config.whisper
        self.stable_ts = new_config.stable_ts
        self.hallucination = new_config.hallucination
        self.llm = new_config.llm
        self.output = new_config.output
        self.logging = new_config.logging


def _merge_dict_to_dataclass(dataclass_instance: Any, data: dict) -> None:
    """辞書データをdataclassにマージする"""
    for key, value in data.items():
        if hasattr(dataclass_instance, key):
            setattr(dataclass_instance, key, value)


def load_config(config_path: Path) -> Config:
    """設定ファイルを読み込む

    Args:
        config_path: 設定ファイルのパス

    Returns:
        Config: 設定オブジェクト

    Raises:
        yaml.YAMLError: YAMLパースエラー時
    """
    config = Config()

    if not config_path.exists():
        return config

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return config

    # 各セクションをマージ
    if "audio" in data:
        _merge_dict_to_dataclass(config.audio, data["audio"])
    if "input" in data:
        _merge_dict_to_dataclass(config.input, data["input"])
    if "vad" in data:
        _merge_dict_to_dataclass(config.vad, data["vad"])
    if "whisper" in data:
        _merge_dict_to_dataclass(config.whisper, data["whisper"])
    if "stable_ts" in data:
        _merge_dict_to_dataclass(config.stable_ts, data["stable_ts"])
    if "hallucination" in data:
        _merge_dict_to_dataclass(config.hallucination, data["hallucination"])
    if "llm" in data:
        _merge_dict_to_dataclass(config.llm, data["llm"])
    if "output" in data:
        _merge_dict_to_dataclass(config.output, data["output"])
    if "logging" in data:
        _merge_dict_to_dataclass(config.logging, data["logging"])

    return config


def save_default_config(config_path: Path) -> None:
    """デフォルト設定をファイルに保存する

    Args:
        config_path: 保存先パス
    """
    config = Config()
    data = _serialize_config(config)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, allow_unicode=True, default_flow_style=False, sort_keys=False
        )

def save_config(config: Config, config_path: Path) -> None:
    """設定をファイルに保存する

    Args:
        config: 設定オブジェクト
        config_path: 保存先パス
    """
    data = _serialize_config(config)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, allow_unicode=True, default_flow_style=False, sort_keys=False
        )


def _serialize_config(config: Config) -> dict:
    """設定をYAML用の辞書に変換する"""
    return {
        "audio": {
            "sample_rate": config.audio.sample_rate,
            "channels": config.audio.channels,
            "device_id": config.audio.device_id,
            "dtype": config.audio.dtype,
        },
        "input": {
            "tmp_dir": config.input.tmp_dir,
            "accept_ext": config.input.accept_ext,
        },
        "vad": {
            "enabled": config.vad.enabled,
            "min_silence_duration_ms": config.vad.min_silence_duration_ms,
        },
        "whisper": {
            "model": config.whisper.model,
            "language": config.whisper.language,
            "device": config.whisper.device,
            "compute_type": config.whisper.compute_type,
            "beam_size": config.whisper.beam_size,
            "temperature": config.whisper.temperature,
            "condition_on_previous_text": config.whisper.condition_on_previous_text,
            "word_timestamps_internal": config.whisper.word_timestamps_internal,
            "batch_size": config.whisper.batch_size,
        },
        "stable_ts": {
            "enabled": config.stable_ts.enabled,
            "suppress_silence": config.stable_ts.suppress_silence,
            "regroup": config.stable_ts.regroup,
        },
        "hallucination": {
            "enabled": config.hallucination.enabled,
            "max_consecutive_token_repeat": config.hallucination.max_consecutive_token_repeat,
            "max_consecutive_line_repeat": config.hallucination.max_consecutive_line_repeat,
            "ngram_size": config.hallucination.ngram_size,
            "ngram_repeat_threshold": config.hallucination.ngram_repeat_threshold,
            "action": config.hallucination.action,
        },
        "llm": {
            "enabled": config.llm.enabled,
            "provider": config.llm.provider,
            "base_url": config.llm.base_url,
            "model": config.llm.model,
            "temperature": config.llm.temperature,
            "max_output_tokens": config.llm.max_output_tokens,
            "timeout_sec": config.llm.timeout_sec,
            "prompt_mode": config.llm.prompt_mode,
        },
        "output": {
            "out_dir": config.output.out_dir,
            "formats_transcript": config.output.formats_transcript,
            "formats_summary": config.output.formats_summary,
            "naming": config.output.naming,
        },
        "logging": {
            "level": config.logging.level,
            "file": config.logging.file,
            "faster_whisper_debug": config.logging.faster_whisper_debug,
        },
    }
