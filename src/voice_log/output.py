"""出力管理モジュール

文字起こし・要約のファイル保存を行う。
"""

from datetime import datetime
from pathlib import Path
from typing import Any


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


def generate_meta_footer(
    model: str = "",
    device: str = "",
    compute_type: str = "",
    audio_duration_sec: float = 0.0,
    processing_time_sec: float = 0.0,
    vad_enabled: bool = False,
    hallucination_trimmed: int = 0,
    **kwargs: Any,
) -> str:
    """メタ情報フッターを生成する

    Args:
        model: モデル名
        device: デバイス
        compute_type: 計算タイプ
        audio_duration_sec: 音声長（秒）
        processing_time_sec: 処理時間（秒）
        vad_enabled: VAD有効フラグ
        hallucination_trimmed: 反復トリム件数

    Returns:
        str: フッターテキスト
    """
    lines = [
        "",
        "---",
        "",
        "## メタ情報",
        "",
    ]

    if model:
        lines.append(f"- **モデル**: {model}")
    if device:
        lines.append(f"- **デバイス**: {device}")
    if compute_type:
        lines.append(f"- **compute_type**: {compute_type}")
    if audio_duration_sec > 0:
        lines.append(f"- **音声長**: {audio_duration_sec:.1f}秒")
    if processing_time_sec > 0:
        rtf = processing_time_sec / audio_duration_sec if audio_duration_sec > 0 else 0
        lines.append(f"- **処理時間**: {processing_time_sec:.1f}秒 (RTF: {rtf:.2f})")
    lines.append(f"- **VAD**: {'有効' if vad_enabled else '無効'}")
    if hallucination_trimmed > 0:
        lines.append(f"- **反復トリム**: {hallucination_trimmed}件")

    lines.append(f"- **生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(lines)


class OutputManager:
    """出力管理クラス"""

    def __init__(
        self,
        out_dir: Path,
        naming: str = "{date}_{time}_{stem}",
        meta_footer: bool = True,
    ):
        """
        Args:
            out_dir: 出力ディレクトリ
            naming: ファイル命名パターン
            meta_footer: メタ情報フッターを追加するかどうか
        """
        self.out_dir = out_dir
        self.naming = naming
        self.meta_footer = meta_footer

    def save_transcript(
        self,
        transcript: str,
        stem: str,
        formats: list[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """文字起こしを保存する

        Args:
            transcript: 文字起こしテキスト
            stem: 元ファイル名
            formats: 出力フォーマット（デフォルト: ["md", "txt"]）
            meta: メタ情報（フッター用）

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
            content = transcript

            if self.meta_footer and meta and fmt == "md":
                content += generate_meta_footer(**meta)

            file_path.write_text(content, encoding="utf-8")
            paths[fmt] = file_path

        return paths

    def save_summary(
        self,
        summary: str,
        stem: str,
        formats: list[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """要約を保存する

        Args:
            summary: 要約テキスト
            stem: 元ファイル名
            formats: 出力フォーマット（デフォルト: ["md"]）
            meta: メタ情報（フッター用）

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

            if self.meta_footer and meta and fmt == "md":
                content += generate_meta_footer(**meta)

            file_path.write_text(content, encoding="utf-8")
            paths[fmt] = file_path

        return paths
