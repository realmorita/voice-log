"""Tests for hallucination detection module.

# TODO: テストリスト
# - [x] 正常系: 同一トークン連続を検知
# - [x] 正常系: 同一行連続を検知
# - [x] 正常系: n-gram反復を検知
# - [x] 正常系: 反復をトリム
# - [x] 境界値: 閾値ぎりぎりのケース
# - [x] 正常系: 反復なしの場合は何もしない
"""

from voice_log.hallucination import (
    HallucinationDetector,
    detect_line_repetition,
    detect_ngram_repetition,
    detect_token_repetition,
    trim_repetitions,
)


class TestDetectTokenRepetition:
    """同一トークン連続検知のテスト"""

    def test_detects_token_repetition(self):
        """同一トークンの連続を検知する"""
        text = "これは は は は は は は は は は テストです"
        result = detect_token_repetition(text, max_repeat=8)

        assert result.has_issue is True
        assert "は" in result.details

    def test_no_detection_below_threshold(self):
        """閾値以下では検知しない"""
        text = "これは は は テストです"
        result = detect_token_repetition(text, max_repeat=8)

        assert result.has_issue is False

    def test_handles_empty_text(self):
        """空のテキストを処理できる"""
        result = detect_token_repetition("", max_repeat=8)
        assert result.has_issue is False


class TestDetectLineRepetition:
    """同一行連続検知のテスト"""

    def test_detects_line_repetition(self):
        """同一行の連続を検知する"""
        text = """これはテストです。
これはテストです。
これはテストです。
これはテストです。
異なる行です。"""
        result = detect_line_repetition(text, max_repeat=3)

        assert result.has_issue is True

    def test_no_detection_below_threshold(self):
        """閾値以下では検知しない"""
        text = """これはテストです。
これはテストです。
異なる行です。"""
        result = detect_line_repetition(text, max_repeat=3)

        assert result.has_issue is False


class TestDetectNgramRepetition:
    """n-gram反復検知のテスト"""

    def test_detects_ngram_repetition(self):
        """n-gramの過剰反復を検知する"""
        # 同じ4-gramが6回以上繰り返される
        text = "会議の内容 会議の内容 会議の内容 会議の内容 会議の内容 会議の内容 会議の内容"
        result = detect_ngram_repetition(text, ngram_size=4, threshold=6)

        assert result.has_issue is True

    def test_no_detection_below_threshold(self):
        """閾値以下では検知しない"""
        text = "会議の内容 会議の内容 異なる内容"
        result = detect_ngram_repetition(text, ngram_size=4, threshold=6)

        assert result.has_issue is False


class TestTrimRepetitions:
    """反復トリムのテスト"""

    def test_trims_token_repetitions(self):
        """トークン反復をトリムする"""
        text = (
            "これは テスト テスト テスト テスト テスト テスト テスト テスト テスト です"
        )
        result = trim_repetitions(text, max_token_repeat=3)

        # 反復が削減されている
        assert result.count("テスト") < 9

    def test_trims_line_repetitions(self):
        """行反復をトリムする"""
        text = """これはテストです。
これはテストです。
これはテストです。
これはテストです。
これはテストです。
異なる行です。"""
        result = trim_repetitions(text, max_line_repeat=2)

        lines = [line for line in result.split("\n") if line.strip()]
        assert lines.count("これはテストです。") <= 2

    def test_preserves_text_without_repetitions(self):
        """反復なしのテキストは変更しない"""
        text = "これは正常なテキストです。問題ありません。"
        result = trim_repetitions(text)

        assert result == text


class TestHallucinationDetector:
    """HallucinationDetector クラスのテスト"""

    def test_detects_all_issues(self):
        """全ての反復問題を検知する"""
        detector = HallucinationDetector(
            max_token_repeat=5,
            max_line_repeat=3,
            ngram_size=4,
            ngram_threshold=5,
        )

        text = "テスト テスト テスト テスト テスト テスト テスト"
        result = detector.analyze(text)

        assert result.has_issues is True
        assert len(result.issues) > 0

    def test_no_issues_for_normal_text(self):
        """正常なテキストでは問題を検知しない"""
        detector = HallucinationDetector()

        text = "これは正常な日本語のテキストです。特に問題はありません。"
        result = detector.analyze(text)

        assert result.has_issues is False

    def test_apply_trim_action(self):
        """トリムアクションを適用する"""
        detector = HallucinationDetector(
            max_token_repeat=3,
            action="trim",
        )

        text = "これは テスト テスト テスト テスト テスト テスト テスト です"
        result = detector.process(text)

        assert result.count("テスト") < 7
