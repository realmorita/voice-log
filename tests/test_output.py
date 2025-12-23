"""Tests for output module.

# TODO: テストリスト
# - [x] 正常系: 出力ファイル名を生成
# - [x] 正常系: 文字起こしをファイルに保存
# - [x] 正常系: 要約をファイルに保存
# - [x] 正常系: メタ情報フッターを生成
"""

import tempfile
from datetime import datetime
from pathlib import Path

from voice_log.output import OutputManager, generate_filename, generate_meta_footer


class TestGenerateFilename:
    """generate_filename のテスト"""

    def test_generates_filename_with_pattern(self):
        """パターンに従ってファイル名を生成する"""
        result = generate_filename(
            pattern="{date}_{time}_{stem}",
            stem="meeting",
            date=datetime(2025, 12, 23, 14, 30, 0),
        )

        assert result == "2025-12-23_143000_meeting"

    def test_generates_filename_with_different_pattern(self):
        """異なるパターンでもファイル名を生成する"""
        result = generate_filename(
            pattern="{stem}_{date}",
            stem="interview",
            date=datetime(2025, 1, 5, 9, 0, 0),
        )

        assert result == "interview_2025-01-05"


class TestGenerateMetaFooter:
    """generate_meta_footer のテスト"""

    def test_generates_meta_footer(self):
        """メタ情報フッターを生成する"""
        result = generate_meta_footer(
            model="large-v3",
            device="cuda",
            compute_type="float16",
            audio_duration_sec=120.5,
            processing_time_sec=45.2,
            vad_enabled=True,
            hallucination_trimmed=2,
        )

        assert "large-v3" in result
        assert "cuda" in result
        assert "float16" in result
        assert "120.5" in result or "2:00" in result


class TestOutputManager:
    """OutputManager クラスのテスト"""

    def test_saves_transcript(self):
        """文字起こしをファイルに保存する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "outputs"
            manager = OutputManager(out_dir, naming="{date}_{time}_{stem}")

            transcript = "これはテストの文字起こしです。"
            paths = manager.save_transcript(
                transcript,
                stem="test",
                formats=["md", "txt"],
            )

            assert len(paths) == 2
            assert paths["md"].exists()
            assert paths["txt"].exists()

            assert transcript in paths["md"].read_text(encoding="utf-8")

    def test_saves_summary(self):
        """要約をファイルに保存する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "outputs"
            manager = OutputManager(out_dir, naming="{date}_{time}_{stem}")

            summary = "# 要約\n\nこれはテストの要約です。"
            paths = manager.save_summary(
                summary,
                stem="test",
                formats=["md"],
            )

            assert len(paths) == 1
            assert paths["md"].exists()
            assert summary in paths["md"].read_text(encoding="utf-8")

    def test_adds_meta_footer(self):
        """メタ情報フッターを追加する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "outputs"
            manager = OutputManager(out_dir, naming="{stem}", meta_footer=True)

            transcript = "テスト内容"
            paths = manager.save_transcript(
                transcript,
                stem="test",
                formats=["md"],
                meta={
                    "model": "large-v3",
                    "device": "cuda",
                },
            )

            content = paths["md"].read_text(encoding="utf-8")
            assert "large-v3" in content
