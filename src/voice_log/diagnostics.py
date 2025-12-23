"""起動時診断モジュール

設定ファイル検証、Ollama疎通確認、CUDA可否判定を行う。
"""

from pathlib import Path

from voice_log.config import Config, load_config
from voice_log.logger import get_logger
from voice_log.summarize import check_ollama_connection, list_ollama_models

logger = get_logger("diagnostics")


class DiagnosticsResult:
    """診断結果"""

    def __init__(self):
        self.config_ok = False
        self.config_message = ""
        self.ollama_ok = False
        self.ollama_message = ""
        self.ollama_models: list[str] = []
        self.cuda_available = False
        self.device = "cpu"
        self.compute_type = "int8"

    @property
    def all_ok(self) -> bool:
        """全ての診断が成功したか"""
        return self.config_ok and self.ollama_ok


def check_cuda() -> tuple[bool, str, str]:
    """CUDA利用可否を確認する

    Returns:
        tuple[bool, str, str]: (利用可能, device, compute_type)
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, "cuda", f"float16 (GPU: {device_name})"
        else:
            return False, "cpu", "int8"
    except ImportError:
        return False, "cpu", "int8"
    except Exception as e:
        logger.warning(f"CUDA確認中にエラー: {e}")
        return False, "cpu", "int8"


def run_diagnostics(
    config_path: Path | None = None,
    config: Config | None = None,
) -> DiagnosticsResult:
    """起動時診断を実行する

    Args:
        config_path: 設定ファイルのパス
        config: 設定オブジェクト（渡された場合はこれを使用）

    Returns:
        DiagnosticsResult: 診断結果
    """
    result = DiagnosticsResult()

    # [1/3] 設定ファイル読み込み
    logger.info("[1/3] 設定ファイルを確認中...")

    if config is not None:
        result.config_ok = True
        result.config_message = "設定オブジェクトを使用"
    elif config_path and config_path.exists():
        try:
            config = load_config(config_path)
            result.config_ok = True
            result.config_message = f"設定ファイルを読み込みました: {config_path}"
        except Exception as e:
            result.config_ok = False
            result.config_message = f"設定ファイルの読み込みに失敗: {e}"
            config = Config()
    else:
        config = Config()
        result.config_ok = True
        result.config_message = "デフォルト設定を使用"

    logger.info(result.config_message)

    # [2/3] Ollama疎通確認
    logger.info("[2/3] Ollamaの接続を確認中...")

    if config.llm.enabled:
        ollama_ok, ollama_msg = check_ollama_connection(config.llm.base_url)
        result.ollama_ok = ollama_ok
        result.ollama_message = ollama_msg

        if ollama_ok:
            models = list_ollama_models(config.llm.base_url)
            result.ollama_models = [m["id"] for m in models]
    else:
        result.ollama_ok = True
        result.ollama_message = "LLM機能は無効化されています"

    logger.info(result.ollama_message)

    # [3/3] CUDA確認
    logger.info("[3/3] CUDA/GPUを確認中...")

    cuda_available, device, compute_type = check_cuda()
    result.cuda_available = cuda_available
    result.device = device
    result.compute_type = compute_type

    if cuda_available:
        logger.info(f"CUDA利用可能: {compute_type}")
    else:
        logger.info("CPUモードで動作します")

    return result
