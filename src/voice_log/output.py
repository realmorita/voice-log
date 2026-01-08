"""出力管理モジュール

文字起こし・要約のファイル保存を行う。
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voice_log.transcribe import TranscriptionSegment


def format_srt_timestamp(seconds: float) -> str:
    """秒数をSRT形式のタイムスタンプに変換する

    Args:
        seconds: 秒数

    Returns:
        str: SRT形式のタイムスタンプ (HH:MM:SS,mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list["TranscriptionSegment"]) -> str:
    """セグメントリストをSRT形式のテキストに変換する

    Args:
        segments: 文字起こしセグメントのリスト

    Returns:
        str: SRT形式のテキスト
    """
    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        start_ts = format_srt_timestamp(seg.start)
        end_ts = format_srt_timestamp(seg.end)

        srt_lines.append(str(i))
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(seg.text)
        srt_lines.append("")  # 空行（セグメント間の区切り）

    return "\n".join(srt_lines)


def generate_filename(
    pattern: str,
    stem: str,
    date: datetime | None = None,
) -> str:
    """出力ファイル名を生成する

    Args:
        pattern: ファイル名パターン（{date}, {time}, {stem}）
        stem: 元ファイル名（拡張子なし）
        date: 日時（デフォルト: 現在時刻）

    Returns:
        str: 生成されたファイル名（拡張子なし）
    """
    if date is None:
        date = datetime.now()

    date_str = date.strftime("%Y-%m-%d")
    time_str = date.strftime("%H%M%S")

    result = pattern
    result = result.replace("{date}", date_str)
    result = result.replace("{time}", time_str)
    result = result.replace("{stem}", stem)

    return result


class OutputManager:
    """出力管理クラス"""

    def __init__(
        self,
        out_dir: Path,
        naming: str = "{date}_{time}_{stem}",
    ):
        """
        Args:
            out_dir: 出力ディレクトリ
            naming: ファイル命名パターン
        """
        self.out_dir = out_dir
        self.naming = naming

    def save_transcript(
        self,
        transcript: str,
        stem: str,
        formats: list[str] | None = None,
        segments: list["TranscriptionSegment"] | None = None,
    ) -> dict[str, Path]:
        """文字起こしを保存する

        Args:
            transcript: 文字起こしテキスト
            stem: 元ファイル名
            formats: 出力フォーマット（デフォルト: ["md", "txt"]）
            segments: 文字起こしセグメント（SRT出力用、オプション）

        Returns:
            dict[str, Path]: フォーマットごとの出力パス
        """
        if formats is None:
            formats = ["md", "txt"]

        self.out_dir.mkdir(parents=True, exist_ok=True)

        base_name = generate_filename(self.naming, stem)
        paths = {}

        for fmt in formats:
            file_path = self.out_dir / f"{base_name}_transcript.{fmt}"

            if fmt == "srt":
                # SRT形式はセグメントから生成
                if segments:
                    content = segments_to_srt(segments)
                else:
                    # セグメントがない場合はスキップ
                    continue
            else:
                content = transcript

            file_path.write_text(content, encoding="utf-8")
            paths[fmt] = file_path

        return paths

    def save_summary(
        self,
        summary: str,
        stem: str,
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """要約を保存する

        Args:
            summary: 要約テキスト
            stem: 元ファイル名
            formats: 出力フォーマット（デフォルト: ["md"]）

        Returns:
            dict[str, Path]: フォーマットごとの出力パス
        """
        if formats is None:
            formats = ["md"]

        self.out_dir.mkdir(parents=True, exist_ok=True)

        base_name = generate_filename(self.naming, stem)
        paths = {}

        for fmt in formats:
            file_path = self.out_dir / f"{base_name}_summary.{fmt}"
            content = summary

            file_path.write_text(content, encoding="utf-8")
            paths[fmt] = file_path

        return paths
