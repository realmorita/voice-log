"""プロンプト管理モジュール

prompts/ 配下のテンプレート読み込みとプレースホルダ置換を行う。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PromptModeInfo:
    """プロンプトモードの表示情報"""

    mode_id: str
    display_name: str


def _parse_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """YAMLフロントマターを解析し、本文を返す

    Args:
        content: プロンプトファイルの内容

    Returns:
        tuple[dict[str, Any], str]: メタデータと本文
    """
    if not content.startswith("---"):
        return {}, content

    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, content

    end_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break

    if end_index is None:
        return {}, content

    metadata_block = "".join(lines[1:end_index])
    body = "".join(lines[end_index + 1 :])

    try:
        metadata = yaml.safe_load(metadata_block)
    except yaml.YAMLError:
        metadata = None

    if not isinstance(metadata, dict):
        metadata = {}

    return metadata, body


def list_prompt_modes(prompts_dir: Path) -> list[PromptModeInfo]:
    """利用可能なプロンプトモード一覧を取得する

    Args:
        prompts_dir: プロンプトディレクトリのパス

    Returns:
        list[PromptModeInfo]: モード情報のリスト
    """
    if not prompts_dir.exists():
        return []

    modes: list[PromptModeInfo] = []
    for file in prompts_dir.glob("*.md"):
        content = file.read_text(encoding="utf-8")
        metadata, _ = _parse_front_matter(content)
        mode_name = metadata.get("mode_name")
        display_name = (
            mode_name if isinstance(mode_name, str) and mode_name else file.stem
        )
        modes.append(PromptModeInfo(mode_id=file.stem, display_name=display_name))

    return sorted(modes, key=lambda mode: mode.mode_id)


def load_prompt(prompts_dir: Path, mode: str) -> str:
    """プロンプトファイルを読み込む

    Args:
        prompts_dir: プロンプトディレクトリのパス
        mode: プロンプトモード名

    Returns:
        str: プロンプトテンプレートの内容

    Raises:
        FileNotFoundError: 指定モードのファイルが存在しない場合
    """
    prompt_path = prompts_dir / f"{mode}.md"

    if not prompt_path.exists():
        raise FileNotFoundError(f"プロンプトファイルが見つかりません: {prompt_path}")

    content = prompt_path.read_text(encoding="utf-8")
    _, body = _parse_front_matter(content)
    return body


def render_prompt(
    template: str,
    transcript: str,
    title: str = "",
    date: str = "",
    lang: str = "ja",
) -> str:
    """プロンプトテンプレートのプレースホルダを置換する

    Args:
        template: プロンプトテンプレート
        transcript: 文字起こしテキスト
        title: タイトル（オプション）
        date: 日付（オプション）
        lang: 言語コード（デフォルト: ja）

    Returns:
        str: プレースホルダ置換後のプロンプト
    """
    result = template
    result = result.replace("{{TRANSCRIPT}}", transcript)
    result = result.replace("{{TITLE}}", title)
    result = result.replace("{{DATE}}", date)
    result = result.replace("{{LANG}}", lang)

    return result


class PromptManager:
    """プロンプト管理クラス"""

    def __init__(self, prompts_dir: Path):
        """
        Args:
            prompts_dir: プロンプトディレクトリのパス
        """
        self.prompts_dir = prompts_dir

    def list_modes(self) -> list[PromptModeInfo]:
        """利用可能なモード一覧を取得する

        Returns:
            list[PromptModeInfo]: モード情報のリスト
        """
        return list_prompt_modes(self.prompts_dir)

    def load(self, mode: str) -> str:
        """プロンプトを読み込む

        Args:
            mode: モード名

        Returns:
            str: プロンプトテンプレート
        """
        return load_prompt(self.prompts_dir, mode)

    def render(
        self,
        mode: str,
        transcript: str,
        title: str = "",
        date: str = "",
        lang: str = "ja",
    ) -> str:
        """プロンプトを読み込んでレンダリングする

        Args:
            mode: モード名
            transcript: 文字起こしテキスト
            title: タイトル（オプション）
            date: 日付（オプション）
            lang: 言語コード（デフォルト: ja）

        Returns:
            str: レンダリング後のプロンプト
        """
        template = self.load(mode)
        return render_prompt(template, transcript, title, date, lang)
