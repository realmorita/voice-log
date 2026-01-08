"""ロギング設定モジュール

アプリケーション全体のログ設定を管理する。
"""

import logging
import sys
from pathlib import Path

from voice_log.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """ロギングを設定する

    Args:
        config: ロギング設定

    Returns:
        logging.Logger: 設定済みのルートロガー
    """
    # ログファイルのディレクトリを作成
    log_path = Path(config.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # ログレベルを設定
    level = getattr(logging, config.level.upper(), logging.INFO)

    # ルートロガーを取得
    logger = logging.getLogger("voice_log")
    logger.setLevel(level)

    # 既存のハンドラをクリア
    logger.handlers.clear()

    # フォーマッタを作成
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ファイルハンドラを追加
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # コンソールハンドラを追加
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # faster-whisperのデバッグログ設定
    if config.faster_whisper_debug:
        fw_logger = logging.getLogger("faster_whisper")
        fw_logger.setLevel(logging.DEBUG)
    else:
        fw_logger = logging.getLogger("faster_whisper")
        fw_logger.setLevel(logging.WARNING)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """ロガーを取得する

    Args:
        name: ロガー名（voice_log.xxx形式）

    Returns:
        logging.Logger: ロガー
    """
    if name:
        return logging.getLogger(f"voice_log.{name}")
    return logging.getLogger("voice_log")


def log_to_file_only(
    logger: logging.Logger,
    level: int,
    message: str,
    *args: object,
) -> None:
    """ログファイルのみに出力する

    Args:
        logger: ロガー
        level: ログレベル
        message: メッセージ
        *args: フォーマット引数
    """
    root_logger = logging.getLogger("voice_log")
    if not root_logger.isEnabledFor(level):
        return

    file_handlers = [
        handler
        for handler in root_logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]
    if not file_handlers:
        logger.log(level, message, *args)
        return

    record = root_logger.makeRecord(
        name=logger.name,
        level=level,
        fn="",
        lno=0,
        msg=message,
        args=args,
        exc_info=None,
    )
    for handler in file_handlers:
        if level >= handler.level:
            handler.handle(record)
