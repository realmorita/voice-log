"""Tests for prompts module.

# TODO: テストリスト
# - [x] 正常系: 利用可能なプロンプトモード一覧を取得
# - [x] 正常系: プロンプトファイルを読み込む
# - [x] 正常系: プレースホルダを置換する
# - [x] 境界値: 存在しないプロンプトモードでエラー
# - [x] 正常系: 日付・タイトル・言語を置換
"""

import tempfile
from pathlib import Path

import pytest

from voice_log.prompts import (
    PromptManager,
    PromptModeInfo,
    list_prompt_modes,
    load_prompt,
    render_prompt,
)


class TestListPromptModes:
    """list_prompt_modes のテスト"""

    def test_lists_available_modes(self):
        """利用可能なプロンプトモード一覧を取得する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()

            (prompts_dir / "minutes.md").write_text("# 議事録")
            (prompts_dir / "todo.md").write_text("# TODO")
            (prompts_dir / "summary_3lines.md").write_text("# 要約")

            modes = list_prompt_modes(prompts_dir)

            mode_ids = [mode.mode_id for mode in modes]
            assert "minutes" in mode_ids
            assert "todo" in mode_ids
            assert "summary_3lines" in mode_ids

    def test_uses_mode_name_from_front_matter(self):
        """フロントマターのモード名を表示名に使う"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()

            content = """---
mode_name: 議事録まとめ
---
# 議事録
"""
            (prompts_dir / "minutes.md").write_text(content, encoding="utf-8")

            modes = list_prompt_modes(prompts_dir)

            assert modes == [
                PromptModeInfo(mode_id="minutes", display_name="議事録まとめ")
            ]

    def test_returns_empty_for_missing_dir(self):
        """存在しないディレクトリでは空を返す"""
        modes = list_prompt_modes(Path("/nonexistent/prompts"))
        assert modes == []


class TestLoadPrompt:
    """load_prompt のテスト"""

    def test_loads_prompt_file(self):
        """プロンプトファイルを読み込む"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()

            content = """# 議事録要約プロンプト

以下の文字起こしを議事録形式で要約してください。

{{TRANSCRIPT}}
"""
            (prompts_dir / "minutes.md").write_text(content, encoding="utf-8")

            result = load_prompt(prompts_dir, "minutes")

            assert "議事録要約プロンプト" in result
            assert "{{TRANSCRIPT}}" in result

    def test_loads_prompt_file_without_front_matter(self):
        """フロントマターを本文に含めない"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()

            content = """---
mode_name: 議事録まとめ
---
# 議事録要約プロンプト

{{TRANSCRIPT}}
"""
            (prompts_dir / "minutes.md").write_text(content, encoding="utf-8")

            result = load_prompt(prompts_dir, "minutes")

            assert "mode_name" not in result
            assert "議事録要約プロンプト" in result

    def test_raises_for_missing_mode(self):
        """存在しないモードでエラーを返す"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()

            with pytest.raises(FileNotFoundError):
                load_prompt(prompts_dir, "nonexistent")


class TestRenderPrompt:
    """render_prompt のテスト"""

    def test_replaces_transcript_placeholder(self):
        """{{TRANSCRIPT}} プレースホルダを置換する"""
        template = "要約してください: {{TRANSCRIPT}}"
        result = render_prompt(template, transcript="これはテストです。")

        assert result == "要約してください: これはテストです。"

    def test_replaces_all_placeholders(self):
        """全てのプレースホルダを置換する"""
        template = """# {{TITLE}}

日付: {{DATE}}
言語: {{LANG}}

内容:
{{TRANSCRIPT}}
"""
        result = render_prompt(
            template,
            transcript="会議の内容です。",
            title="定例会議",
            date="2025-12-23",
            lang="ja",
        )

        assert "定例会議" in result
        assert "2025-12-23" in result
        assert "ja" in result
        assert "会議の内容です。" in result

    def test_handles_missing_optional_placeholders(self):
        """オプションのプレースホルダが未指定でも動作する"""
        template = "{{TITLE}}: {{TRANSCRIPT}}"
        result = render_prompt(template, transcript="テスト")

        assert "{{TITLE}}" not in result  # 空文字に置換される
        assert "テスト" in result


class TestPromptManager:
    """PromptManager クラスのテスト"""

    def test_manages_prompts(self):
        """プロンプトの管理ができる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()

            content = "要約: {{TRANSCRIPT}}"
            (prompts_dir / "minutes.md").write_text(content)

            manager = PromptManager(prompts_dir)

            assert manager.list_modes() == [
                PromptModeInfo(mode_id="minutes", display_name="minutes")
            ]

            rendered = manager.render("minutes", transcript="テスト内容")
            assert rendered == "要約: テスト内容"
