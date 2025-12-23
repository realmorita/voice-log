"""プロンプト管理モジュール

prompts/ 配下のテンプレート読み込みとプレースホルダ置換を行う。
"""

from pathlib import Path


def list_prompt_modes(prompts_dir: Path) -> list[str]:
    """利用可能なプロンプトモード一覧を取得する

    Args:
        prompts_dir: プロンプトディレクトリのパス

    Returns:
        list[str]: モード名（ファイル名から拡張子を除いたもの）のリスト
    """
    if not prompts_dir.exists():
        return []

    modes = []
    for file in prompts_dir.glob("*.md"):
        modes.append(file.stem)

    return sorted(modes)


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

    return prompt_path.read_text(encoding="utf-8")


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

    def list_modes(self) -> list[str]:
        """利用可能なモード一覧を取得する

        Returns:
            list[str]: モード名のリスト
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
