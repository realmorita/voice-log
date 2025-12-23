"""要約エンジンモジュール

Ollama（OpenAI互換API）経由でローカルLLMによる要約を行う。
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI

from voice_log.config import LlmConfig
from voice_log.logger import get_logger, log_to_file_only
from voice_log.prompts import PromptManager

logger = get_logger("summarize")


def replace_surrogate_codepoints(
    text: str,
    replacement: str = "\uFFFD",
) -> tuple[str, list[tuple[int, int]]]:
    """サロゲート文字を置換し、位置とコードポイントを返す

    Args:
        text: 対象文字列
        replacement: 置換文字

    Returns:
        tuple[str, list[tuple[int, int]]]: 置換後文字列と(位置,コードポイント)一覧
    """
    if text == "":
        return text, []

    occurrences: list[tuple[int, int]] = []
    replaced_characters: list[str] = []

    for index, character in enumerate(text):
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            occurrences.append((index, codepoint))
            replaced_characters.append(replacement)
        else:
            replaced_characters.append(character)

    if not occurrences:
        return text, []

    return "".join(replaced_characters), occurrences


def format_surrogate_codepoints(
    occurrences: list[tuple[int, int]],
    max_items: int = 5,
) -> str:
    """サロゲート文字の概要をログ向けに整形する

    Args:
        occurrences: (位置,コードポイント)一覧
        max_items: 表示する最大件数

    Returns:
        str: 概要文字列
    """
    if not occurrences:
        return "count=0"

    limited = occurrences[:max_items]
    position_items = ", ".join(
        f"{index}:{codepoint:04X}" for index, codepoint in limited
    )
    remaining = len(occurrences) - len(limited)
    suffix = "" if remaining <= 0 else f", ... +{remaining}"

    return f"count={len(occurrences)}, positions={position_items}{suffix}"


@dataclass
class SummaryResult:
    """要約結果"""

    text: str
    success: bool = True
    error: str = ""
    model: str = ""
    tokens_used: int = 0


def check_ollama_connection(base_url: str) -> tuple[bool, str]:
    """Ollamaへの接続を確認する

    Args:
        base_url: OllamaのベースURL

    Returns:
        tuple[bool, str]: (接続成功, メッセージ)
    """
    try:
        # 将来的なモデルプロバイダーの拡張を踏まえ、base_urlを切り替えるだけで複数のプロバイダーをサポートするOpenAI SDKを使用する
        client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollamaでは使用されない
            timeout=10.0,
        )

        models = client.models.list()
        model_count = len(list(models))
        return True, f"接続成功: {model_count}個のモデルが利用可能"

    except APIConnectionError as e:
        return False, f"接続エラー: Ollamaが起動していない可能性があります ({e})"
    except Exception as e:
        return False, f"エラー: {e}"


def list_ollama_models(base_url: str) -> list[dict[str, Any]]:
    """利用可能なOllamaモデル一覧を取得する

    Args:
        base_url: OllamaのベースURL

    Returns:
        list[dict]: モデル情報のリスト
    """
    try:
        client = OpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=10.0,
        )

        models = client.models.list()
        result = []

        for model in models:
            result.append(
                {
                    "id": model.id,
                    "owned_by": getattr(model, "owned_by", "ollama"),
                }
            )

        return result

    except Exception as e:
        logger.error(f"モデル一覧取得エラー: {e}")
        return []


class SummaryEngine:
    """要約エンジン"""

    def __init__(
        self,
        config: LlmConfig,
        prompts_dir: Path | None = None,
    ):
        """
        Args:
            config: LLM設定
            prompts_dir: プロンプトディレクトリ
        """
        self.config = config
        self.prompts_dir = prompts_dir or Path("prompts")
        self.prompt_manager = PromptManager(self.prompts_dir)

        self._client = OpenAI(
            base_url=config.base_url,
            api_key="ollama",
            timeout=float(config.timeout_sec),
            max_retries=0,
        )

    def generate(
        self,
        transcript: str,
        prompt_mode: str | None = None,
        title: str = "",
        date: str = "",
    ) -> SummaryResult:
        """要約を生成する

        Args:
            transcript: 文字起こしテキスト
            prompt_mode: プロンプトモード（None=設定値を使用）
            title: タイトル
            date: 日付

        Returns:
            SummaryResult: 要約結果
        """
        mode = prompt_mode or self.config.prompt_mode

        sanitized_transcript, transcript_surrogates = replace_surrogate_codepoints(
            transcript
        )
        if transcript_surrogates:
            log_to_file_only(
                logger,
                logging.INFO,
                "文字起こしテキストのサロゲート文字を検出して置換: %s, replacement=U+FFFD",
                format_surrogate_codepoints(transcript_surrogates),
            )

        sanitized_title, title_surrogates = replace_surrogate_codepoints(title)
        if title_surrogates:
            log_to_file_only(
                logger,
                logging.INFO,
                "タイトルのサロゲート文字を検出して置換: %s, replacement=U+FFFD",
                format_surrogate_codepoints(title_surrogates),
            )

        sanitized_date, date_surrogates = replace_surrogate_codepoints(date)
        if date_surrogates:
            log_to_file_only(
                logger,
                logging.INFO,
                "日付のサロゲート文字を検出して置換: %s, replacement=U+FFFD",
                format_surrogate_codepoints(date_surrogates),
            )

        try:
            # プロンプトをレンダリング
            prompt = self.prompt_manager.render(
                mode=mode,
                transcript=sanitized_transcript,
                title=sanitized_title,
                date=sanitized_date,
            )
        except FileNotFoundError as e:
            logger.error(f"プロンプトファイルが見つかりません: {e}")
            return SummaryResult(
                text="",
                success=False,
                error=str(e),
                model=self.config.model,
            )

        sanitized_prompt, prompt_surrogates = replace_surrogate_codepoints(prompt)
        if prompt_surrogates:
            log_to_file_only(
                logger,
                logging.INFO,
                "要約プロンプトのサロゲート文字を検出して置換: %s, replacement=U+FFFD",
                format_surrogate_codepoints(prompt_surrogates),
            )
            prompt = sanitized_prompt

        logger.info(f"要約生成中: モデル={self.config.model}, モード={mode}")

        try:
            # チャット補完のパラメータを準備
            completion_kwargs = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "あなたは文字起こしを要約するアシスタントです。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.config.temperature,
            }

            completion = self._client.chat.completions.create(**completion_kwargs)

            summary_text = completion.choices[0].message.content or ""
            tokens_used = getattr(completion.usage, "total_tokens", 0)

            logger.info(f"要約生成完了: {len(summary_text)}文字")

            return SummaryResult(
                text=summary_text,
                success=True,
                model=self.config.model,
                tokens_used=tokens_used,
            )

        except APIConnectionError as e:
            error_msg = f"Ollamaに接続できません: {e}"
            logger.error(error_msg)
            return SummaryResult(
                text="",
                success=False,
                error=error_msg,
                model=self.config.model,
            )

        except APITimeoutError as e:
            error_msg = f"要約生成がタイムアウトしました: {e}"
            logger.error(error_msg)
            return SummaryResult(
                text="",
                success=False,
                error=error_msg,
                model=self.config.model,
            )

        except Exception as e:
            error_msg = f"要約生成エラー: {e}"
            logger.error(error_msg)
            return SummaryResult(
                text="",
                success=False,
                error=error_msg,
                model=self.config.model,
            )


def generate_summary(
    transcript: str,
    config: LlmConfig,
    prompts_dir: Path | None = None,
    prompt_mode: str | None = None,
    title: str = "",
    date: str = "",
) -> SummaryResult:
    """要約を生成する（簡易関数）

    Args:
        transcript: 文字起こしテキスト
        config: LLM設定
        prompts_dir: プロンプトディレクトリ
        prompt_mode: プロンプトモード
        title: タイトル
        date: 日付

    Returns:
        SummaryResult: 要約結果
    """
    engine = SummaryEngine(config, prompts_dir)
    return engine.generate(transcript, prompt_mode, title, date)
