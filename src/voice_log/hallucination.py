"""反復（ハルシネーション）検知・抑制モジュール

同一トークン・同一行・n-gram反復を検知し、必要に応じてトリムする。
"""

import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class DetectionResult:
    """検知結果"""

    has_issue: bool = False
    details: str = ""
    positions: list[int] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """分析結果"""

    has_issues: bool = False
    issues: list[DetectionResult] = field(default_factory=list)
    original_text: str = ""


def _tokenize(text: str) -> list[str]:
    """テキストをトークンに分割する

    日本語の文字と空白で分割する簡易トークナイザ
    """
    # 空白で分割し、空文字を除去
    tokens = text.split()
    if not tokens:
        # 空白がない場合は文字単位で分割（日本語対応）
        tokens = list(text)
    return [t for t in tokens if t.strip()]


def detect_token_repetition(text: str, max_repeat: int = 8) -> DetectionResult:
    """同一トークンの連続を検知する

    Args:
        text: 検査対象のテキスト
        max_repeat: 許容する最大連続回数

    Returns:
        DetectionResult: 検知結果
    """
    if not text.strip():
        return DetectionResult(has_issue=False)

    tokens = _tokenize(text)
    if len(tokens) < 2:
        return DetectionResult(has_issue=False)

    # 連続する同一トークンを検知
    max_consecutive = 1
    current_consecutive = 1
    repeated_token = ""

    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            current_consecutive += 1
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
                repeated_token = tokens[i]
        else:
            current_consecutive = 1

    if max_consecutive > max_repeat:
        return DetectionResult(
            has_issue=True,
            details=f"トークン「{repeated_token}」が{max_consecutive}回連続しています",
        )

    return DetectionResult(has_issue=False)


def detect_line_repetition(text: str, max_repeat: int = 3) -> DetectionResult:
    """同一行の連続を検知する

    Args:
        text: 検査対象のテキスト
        max_repeat: 許容する最大連続回数

    Returns:
        DetectionResult: 検知結果
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if len(lines) < 2:
        return DetectionResult(has_issue=False)

    max_consecutive = 1
    current_consecutive = 1
    repeated_line = ""

    for i in range(1, len(lines)):
        if lines[i] == lines[i - 1]:
            current_consecutive += 1
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
                repeated_line = lines[i]
        else:
            current_consecutive = 1

    if max_consecutive > max_repeat:
        return DetectionResult(
            has_issue=True,
            details=f"行が{max_consecutive}回連続しています: {repeated_line[:30]}...",
        )

    return DetectionResult(has_issue=False)


def detect_ngram_repetition(
    text: str, ngram_size: int = 4, threshold: int = 6
) -> DetectionResult:
    """n-gramの過剰反復を検知する

    Args:
        text: 検査対象のテキスト
        ngram_size: n-gramのサイズ（文字数）
        threshold: 反復閾値

    Returns:
        DetectionResult: 検知結果
    """
    # 空白を正規化
    normalized = re.sub(r"\s+", " ", text.strip())

    if len(normalized) < ngram_size:
        return DetectionResult(has_issue=False)

    # n-gramを抽出してカウント
    ngrams = []
    for i in range(len(normalized) - ngram_size + 1):
        ngram = normalized[i : i + ngram_size]
        ngrams.append(ngram)

    counter = Counter(ngrams)

    # 閾値を超える反復を探す
    for ngram, count in counter.most_common(10):
        if count >= threshold and ngram.strip():
            return DetectionResult(
                has_issue=True,
                details=f"n-gram「{ngram}」が{count}回出現しています",
            )

    return DetectionResult(has_issue=False)


def trim_repetitions(
    text: str,
    max_token_repeat: int = 3,
    max_line_repeat: int = 2,
) -> str:
    """反復をトリムする

    Args:
        text: 対象テキスト
        max_token_repeat: トークン反復の上限
        max_line_repeat: 行反復の上限

    Returns:
        str: トリム後のテキスト
    """
    result = text

    # 行反復をトリム
    lines = result.split("\n")
    trimmed_lines = []
    consecutive_count = 0
    prev_line = None

    for line in lines:
        stripped = line.strip()
        if stripped == prev_line and stripped:
            consecutive_count += 1
            if consecutive_count < max_line_repeat:
                trimmed_lines.append(line)
        else:
            trimmed_lines.append(line)
            consecutive_count = 0
            prev_line = stripped

    result = "\n".join(trimmed_lines)

    # トークン反復をトリム
    tokens = result.split()
    if len(tokens) > 1:
        trimmed_tokens = []
        consecutive_count = 0
        prev_token = None

        for token in tokens:
            if token == prev_token:
                consecutive_count += 1
                if consecutive_count < max_token_repeat:
                    trimmed_tokens.append(token)
            else:
                trimmed_tokens.append(token)
                consecutive_count = 0
                prev_token = token

        result = " ".join(trimmed_tokens)

    return result


class HallucinationDetector:
    """反復検知・処理クラス"""

    def __init__(
        self,
        max_token_repeat: int = 8,
        max_line_repeat: int = 3,
        ngram_size: int = 4,
        ngram_threshold: int = 6,
        action: str = "trim",
    ):
        """
        Args:
            max_token_repeat: トークン連続の閾値
            max_line_repeat: 行連続の閾値
            ngram_size: n-gramサイズ
            ngram_threshold: n-gram反復閾値
            action: 検知時のアクション（"trim", "warn_only"）
        """
        self.max_token_repeat = max_token_repeat
        self.max_line_repeat = max_line_repeat
        self.ngram_size = ngram_size
        self.ngram_threshold = ngram_threshold
        self.action = action

    def analyze(self, text: str) -> AnalysisResult:
        """テキストを分析して反復問題を検知する

        Args:
            text: 分析対象のテキスト

        Returns:
            AnalysisResult: 分析結果
        """
        issues = []

        token_result = detect_token_repetition(text, self.max_token_repeat)
        if token_result.has_issue:
            issues.append(token_result)

        line_result = detect_line_repetition(text, self.max_line_repeat)
        if line_result.has_issue:
            issues.append(line_result)

        ngram_result = detect_ngram_repetition(
            text, self.ngram_size, self.ngram_threshold
        )
        if ngram_result.has_issue:
            issues.append(ngram_result)

        return AnalysisResult(
            has_issues=len(issues) > 0,
            issues=issues,
            original_text=text,
        )

    def process(self, text: str) -> str:
        """テキストを処理して反復を除去する

        Args:
            text: 処理対象のテキスト

        Returns:
            str: 処理後のテキスト
        """
        if self.action == "warn_only":
            return text

        return trim_repetitions(
            text,
            max_token_repeat=self.max_token_repeat,
            max_line_repeat=self.max_line_repeat,
        )
