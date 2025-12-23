"""音声入出力モジュール

録音デバイス管理、マイク録音、音声ファイル変換を行う。
"""

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from voice_log.logger import get_logger

logger = get_logger("audio_io")


def list_audio_devices() -> list[dict]:
    """録音デバイス一覧を取得する

    Returns:
        list[dict]: デバイス情報のリスト
    """
    devices = sd.query_devices()
    input_devices = []

    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            input_devices.append(
                {
                    "id": i,
                    "name": device["name"],
                    "channels": device["max_input_channels"],
                    "sample_rate": device["default_samplerate"],
                }
            )

    return input_devices


def get_default_input_device() -> int | None:
    """デフォルト入力デバイスIDを取得する

    Returns:
        int | None: デバイスID、見つからない場合はNone
    """
    try:
        return sd.default.device[0]
    except Exception:
        return None


class AudioRecorder:
    """音声録音クラス"""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device_id: int | None = None,
        dtype: str = "float32",
    ):
        """
        Args:
            sample_rate: サンプルレート
            channels: チャンネル数
            device_id: デバイスID（None=デフォルト）
            dtype: データ型
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_id = device_id
        self.dtype = dtype

        self._frames: list[np.ndarray] = []
        self._recording = False
        self._stream = None

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """録音コールバック"""
        if status:
            logger.warning(f"録音ステータス: {status}")
        self._frames.append(indata.copy())

    def start(self) -> None:
        """録音を開始する"""
        self._frames = []
        self._recording = True

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device_id,
            dtype=self.dtype,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("録音を開始しました")

    def stop(self) -> np.ndarray:
        """録音を停止し、録音データを返す

        Returns:
            np.ndarray: 録音データ
        """
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._recording = False
        logger.info("録音を停止しました")

        if not self._frames:
            return np.array([])

        return np.concatenate(self._frames, axis=0)

    def is_recording(self) -> bool:
        """録音中かどうか

        Returns:
            bool: 録音中ならTrue
        """
        return self._recording

    def save(self, audio_data: np.ndarray, path: Path) -> Path:
        """録音データをWAVファイルに保存する

        Args:
            audio_data: 録音データ
            path: 保存先パス

        Returns:
            Path: 保存したファイルパス
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio_data, self.sample_rate)
        logger.info(f"録音を保存しました: {path}")
        return path


def record_with_enter_control(
    recorder: AudioRecorder,
    output_path: Path,
    on_start: Callable[[], None] | None = None,
    on_stop: Callable[[], None] | None = None,
) -> Path:
    """Enterキーで録音を開始/停止する

    Args:
        recorder: AudioRecorderインスタンス
        output_path: 出力パス
        on_start: 録音開始時コールバック
        on_stop: 録音停止時コールバック

    Returns:
        Path: 保存したファイルパス
    """
    if on_start:
        on_start()

    recorder.start()
    input()  # Enterで停止

    audio_data = recorder.stop()

    if on_stop:
        on_stop()

    return recorder.save(audio_data, output_path)


def convert_to_wav(
    input_path: Path,
    output_path: Path | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """音声ファイルをWAV形式に変換する（FFmpeg使用）

    Args:
        input_path: 入力ファイルパス
        output_path: 出力パス（None=一時ファイル）
        sample_rate: サンプルレート
        channels: チャンネル数

    Returns:
        Path: 変換後のファイルパス

    Raises:
        RuntimeError: FFmpegが見つからない、または変換失敗
    """
    if output_path is None:
        tmp_dir = tempfile.mkdtemp()
        output_path = Path(tmp_dir) / "converted.wav"

    # 既にWAVで正しいフォーマットなら変換不要
    if input_path.suffix.lower() == ".wav":
        try:
            info = sf.info(input_path)
            if info.samplerate == sample_rate and info.channels == channels:
                logger.info(f"変換不要: {input_path}")
                return input_path
        except Exception:
            pass

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        str(output_path),
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"変換完了: {input_path} -> {output_path}")
        return output_path
    except FileNotFoundError as err:
        raise RuntimeError(
            "FFmpegが見つかりません。FFmpegをインストールしてください。"
        ) from err
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"音声ファイルの変換に失敗しました: {e.stderr}") from e


def get_audio_duration(path: Path) -> float:
    """音声ファイルの長さを取得する

    Args:
        path: 音声ファイルパス

    Returns:
        float: 音声長（秒）
    """
    info = sf.info(path)
    return info.duration
