"""
ロギングユーティリティ

構造化ログを提供します。
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    ロガーをセットアップする

    Args:
        name: ロガー名（通常は__name__）
        log_file: ログファイルのパス（Noneの場合はコンソールのみ）
        level: ログレベル（logging.DEBUG, INFO, WARNING, ERROR, CRITICAL）
        format_string: カスタムフォーマット文字列

    Returns:
        設定されたロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 既存のハンドラーをクリア（重複を防ぐ）
    if logger.handlers:
        logger.handlers.clear()

    # デフォルトフォーマット
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # コンソールハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ファイルハンドラー
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    既存のロガーを取得する（なければデフォルト設定で作成）

    Args:
        name: ロガー名

    Returns:
        ロガー
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
