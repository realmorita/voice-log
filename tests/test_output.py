"""Tests for output module.

# TODO: テストリスト
# - [x] 正常系: 出力ファイル名を生成
# - [x] 正常系: 文字起こしをファイルに保存
# - [x] 正常系: 要約をファイルに保存
"""

import tempfile
from datetime import datetime
from pathlib import Path

from voice_log.output import OutputManager, generate_filename


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

class TestFormatSrtTimestamp:
    """format_srt_timestamp のテスト"""

    def test_formats_zero_seconds(self):
        """0秒を正しくフォーマットする"""
        from voice_log.output import format_srt_timestamp

        result = format_srt_timestamp(0.0)
        assert result == "00:00:00,000"

    def test_formats_simple_seconds(self):
        """秒数を正しくフォーマットする"""
        from voice_log.output import format_srt_timestamp

        result = format_srt_timestamp(5.32)
        assert result == "00:00:05,320"

    def test_formats_minutes_and_seconds(self):
        """分と秒を正しくフォーマットする"""
        from voice_log.output import format_srt_timestamp

        result = format_srt_timestamp(125.5)  # 2分5.5秒
        assert result == "00:02:05,500"

    def test_formats_hours(self):
        """時間を正しくフォーマットする"""
        from voice_log.output import format_srt_timestamp

        result = format_srt_timestamp(3661.123)  # 1時間1分1.123秒
        assert result == "01:01:01,123"


class TestSegmentsToSrt:
    """segments_to_srt のテスト"""

    def test_converts_segments_to_srt(self):
        """セグメントをSRT形式に変換する"""
        from dataclasses import dataclass

        from voice_log.output import segments_to_srt

        @dataclass
        class MockSegment:
            start: float
            end: float
            text: str

        segments = [
            MockSegment(start=0.0, end=2.5, text="こんにちは"),
            MockSegment(start=2.5, end=5.0, text="世界"),
        ]

        result = segments_to_srt(segments)

        assert "1\n" in result
        assert "00:00:00,000 --> 00:00:02,500" in result
        assert "こんにちは" in result
        assert "2\n" in result
        assert "00:00:02,500 --> 00:00:05,000" in result
        assert "世界" in result

    def test_returns_empty_for_no_segments(self):
        """セグメントがない場合は空文字を返す"""
        from voice_log.output import segments_to_srt

        result = segments_to_srt([])
        assert result == ""


class TestSrtOutput:
    """SRT出力のテスト"""

    def test_saves_transcript_with_srt_format(self):
        """SRT形式で文字起こしを保存する"""
        from dataclasses import dataclass

        @dataclass
        class MockSegment:
            start: float
            end: float
            text: str

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "outputs"
            manager = OutputManager(out_dir, naming="{stem}")

            transcript = "こんにちは 世界"
            segments = [
                MockSegment(start=0.0, end=2.5, text="こんにちは"),
                MockSegment(start=2.5, end=5.0, text="世界"),
            ]

            paths = manager.save_transcript(
                transcript,
                stem="test",
                formats=["srt"],
                segments=segments,
            )

            assert "srt" in paths
            assert paths["srt"].exists()

            content = paths["srt"].read_text(encoding="utf-8")
            assert "00:00:00,000 --> 00:00:02,500" in content
            assert "こんにちは" in content

    def test_skips_srt_without_segments(self):
        """セグメントがない場合はSRT出力をスキップする"""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "outputs"
            manager = OutputManager(out_dir, naming="{stem}")

            paths = manager.save_transcript(
                "テスト",
                stem="test",
                formats=["srt", "txt"],
            )

            # txtは生成されるがsrtはスキップされる
            assert "txt" in paths
            assert "srt" not in paths
