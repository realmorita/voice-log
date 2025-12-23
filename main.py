"""Voice Log - ローカル音声文字起こし＋LLM要約ツール"""

import os
import sys

# cuDNN ライブラリをプリロード（ctranslate2/faster-whisper用）
# nvidia-cudnn-cu* パッケージのライブラリを事前にロードしておく
def _preload_cudnn_libraries() -> None:
    """cuDNN ライブラリを ctypes で事前にロード"""
    try:
        import ctypes
        import nvidia.cudnn
        # nvidia.cudnn は namespace パッケージなので __path__ を使用
        cudnn_base_path = nvidia.cudnn.__path__[0] if nvidia.cudnn.__path__ else None
        if cudnn_base_path is None:
            return
        cudnn_lib_path = os.path.join(cudnn_base_path, "lib")
        if os.path.isdir(cudnn_lib_path):
            # LD_LIBRARY_PATH も設定（子プロセス用）
            ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
            if cudnn_lib_path not in ld_library_path:
                os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib_path}:{ld_library_path}"

            # 主要な cuDNN ライブラリをプリロード
            cudnn_libs = [
                "libcudnn.so.9",
                "libcudnn_ops.so.9",
                "libcudnn_cnn.so.9",
                "libcudnn_adv.so.9",
                "libcudnn_graph.so.9",
                "libcudnn_heuristic.so.9",
            ]
            for lib_name in cudnn_libs:
                lib_path = os.path.join(cudnn_lib_path, lib_name)
                if os.path.exists(lib_path):
                    try:
                        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass  # ロード失敗は無視
    except ImportError:
        pass  # nvidia-cudnn がインストールされていない場合は何もしない

_preload_cudnn_libraries()
import tempfile
import logging
from datetime import datetime
from pathlib import Path

from voice_log.audio_io import (
    AudioRecorder,
    convert_to_wav,
    get_audio_duration,
    list_audio_devices,
    record_with_enter_control,
)
from voice_log.config import Config, load_config, save_config, save_default_config
from voice_log.diagnostics import run_diagnostics
from voice_log.logger import setup_logging, get_logger, log_to_file_only
from voice_log.output import OutputManager
from voice_log.prompts import PromptManager
from voice_log.summarize import (
    SummaryEngine,
    check_ollama_connection,
    list_ollama_models,
)
from voice_log.transcribe import TranscriptionEngine
from voice_log import ui


# デフォルトパス
DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_PROMPTS_DIR = Path("prompts")
DEFAULT_OUTPUT_DIR = Path("outputs")


def get_config() -> Config:
    """設定を読み込む"""
    if DEFAULT_CONFIG_PATH.exists():
        return load_config(DEFAULT_CONFIG_PATH)
    return Config()


def handle_record_and_transcribe(config: Config) -> None:
    """[1] 録音して文字起こし（＋要約）"""
    ui.show_info("録音開始するにはEnterを押してください。もう一度Enterで録音停止。")

    # 録音
    recorder = AudioRecorder(
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
        device_id=config.audio.device_id,
    )

    tmp_dir = Path(tempfile.mkdtemp())
    wav_path = tmp_dir / "recording.wav"

    try:
        ui.wait_for_enter("Enterで録音開始...")

        ui.show_info("🎙️ 録音中... (Enterで停止)")
        recorder.start()
        input()  # Enterで停止
        audio_data = recorder.stop()

        if len(audio_data) == 0:
            ui.show_error("録音データが空です")
            return

        recorder.save(audio_data, wav_path)
        duration = get_audio_duration(wav_path)
        ui.show_success(f"録音完了: {duration:.1f}秒")

        # 文字起こし
        _process_transcription(wav_path, config, "recording")

    except Exception as e:
        ui.show_error(f"録音エラー: {e}")


def handle_file_transcribe(config: Config) -> None:
    """[2] ファイルから文字起こし（＋要約）"""
    file_path = ui.ask_file_path("音声ファイルパスを入力")
    if not file_path:
        return

    # 対応形式チェック
    ext = file_path.suffix.lower().lstrip(".")
    if ext not in config.input.accept_ext:
        ui.show_error(
            f"非対応の形式です: {ext}\n"
            f"対応形式: {', '.join(config.input.accept_ext)}"
        )
        return

    # 必要に応じてWAV変換
    if ext != "wav":
        ui.show_info(f"WAV形式に変換中: {file_path.name}")
        try:
            wav_path = convert_to_wav(
                file_path,
                sample_rate=config.audio.sample_rate,
                channels=config.audio.channels,
            )
        except RuntimeError as e:
            ui.show_error(str(e))
            return
    else:
        wav_path = file_path

    _process_transcription(wav_path, config, file_path.stem)


def _process_transcription(wav_path: Path, config: Config, stem: str) -> None:
    """文字起こし＋要約の共通処理"""
    logger = get_logger("main")

    # 文字起こし
    ui.show_info("文字起こし中...")

    try:
        engine = TranscriptionEngine(config)
        result = engine.transcribe(wav_path)

        ui.show_success(
            f"文字起こし完了: {result.processing_time_sec:.1f}秒 "
            f"(RTF: {result.processing_time_sec / result.duration_sec:.2f})"
        )

        if result.hallucination_issues > 0:
            ui.show_warning(f"反復検知: {result.hallucination_issues}件")

    except Exception as e:
        ui.show_error(f"文字起こしエラー: {e}")
        logger.exception("文字起こしエラー")
        return

    # 出力保存
    output_manager = OutputManager(
        out_dir=Path(config.output.out_dir),
        naming=config.output.naming,
    )

    meta = {
        "model": config.whisper.model,
        "device": result.device,
        "compute_type": result.compute_type,
        "audio_duration_sec": result.duration_sec,
        "processing_time_sec": result.processing_time_sec,
        "vad_enabled": config.vad.enabled,
        "hallucination_trimmed": result.hallucination_issues,
    }
    log_to_file_only(logger, logging.INFO, "文字起こしメタ情報: %s", meta)

    paths = output_manager.save_transcript(
        result.text,
        stem=stem,
        formats=config.output.formats_transcript,
        segments=result.segments,
    )
    if paths:
        ui.show_success(f"文字起こし保存: {list(paths.values())[0]}")
    else:
        ui.show_warning("文字起こしファイルは保存されませんでした (セグメント無しか設定の問題)")

    # 要約（LLM有効時）
    if config.llm.enabled:
        ui.show_info("要約生成中...")

        try:
            summary_engine = SummaryEngine(
                config=config.llm,
                prompts_dir=DEFAULT_PROMPTS_DIR,
            )

            date_str = datetime.now().strftime("%Y-%m-%d")
            summary_result = summary_engine.generate(
                transcript=result.text,
                date=date_str,
            )

            if summary_result.success:
                log_to_file_only(
                    logger,
                    logging.INFO,
                    "要約メタ情報: %s",
                    {"llm_model": summary_result.model},
                )
                summary_paths = output_manager.save_summary(
                    summary_result.text,
                    stem=stem,
                    formats=config.output.formats_summary,
                )
                ui.show_success(f"要約保存: {list(summary_paths.values())[0]}")
            else:
                ui.show_warning(f"要約スキップ: {summary_result.error}")

        except Exception as e:
            ui.show_warning(f"要約スキップ: {e}")
            logger.exception("要約エラー")


def handle_text_summary(config: Config) -> None:
    """[3] テキストから要約のみ"""
    if not config.llm.enabled:
        ui.show_error("LLM機能が無効化されています")
        return

    text = ui.ask_multiline_text("文字起こしテキストを入力 (空行で終了)")
    if not text.strip():
        ui.show_error("テキストが入力されていません")
        return

    ui.show_info("要約生成中...")

    try:
        summary_engine = SummaryEngine(
            config=config.llm,
            prompts_dir=DEFAULT_PROMPTS_DIR,
        )

        date_str = datetime.now().strftime("%Y-%m-%d")
        summary_result = summary_engine.generate(
            transcript=text,
            date=date_str,
        )

        if summary_result.success:
            ui.console.print("\n[bold]要約結果:[/bold]")
            ui.console.print(summary_result.text)

            # 保存確認
            save = ui.ask_input("ファイルに保存しますか? (y/n)", default="y")
            if save.lower() == "y":
                output_manager = OutputManager(
                    out_dir=Path(config.output.out_dir),
                    naming=config.output.naming,
                )
                paths = output_manager.save_summary(
                    summary_result.text,
                    stem="text_summary",
                    formats=["md"],
                )
                ui.show_success(f"保存: {list(paths.values())[0]}")
        else:
            ui.show_error(f"要約失敗: {summary_result.error}")

    except Exception as e:
        ui.show_error(f"要約エラー: {e}")


def handle_list_devices() -> None:
    """[4] 録音デバイス一覧"""
    devices = list_audio_devices()
    ui.show_devices(devices)


def handle_list_prompt_modes(config: Config) -> None:
    """[5] 要約モード選択"""
    manager = PromptManager(DEFAULT_PROMPTS_DIR)
    modes = manager.list_modes()

    selected_mode = ui.ask_summary_mode(modes, config.llm.prompt_mode)
    if selected_mode is None:
        ui.show_info("キャンセルしました")
        return

    if selected_mode == config.llm.prompt_mode:
        ui.show_info("モードは変更されていません")
        return

    config.llm.prompt_mode = selected_mode
    save_config(config, DEFAULT_CONFIG_PATH)
    ui.show_success(f"要約モードを更新しました: {selected_mode}")


def handle_select_summary_model(config: Config) -> None:
    """[6] 要約モデル選択"""
    if not config.llm.enabled:
        ui.show_error("LLM機能が無効化されています")
        return

    ok, message = check_ollama_connection(config.llm.base_url)
    if not ok:
        ui.show_error(message)
        return

    try:
        models = [model["id"] for model in list_ollama_models(config.llm.base_url)]
    except Exception as e:
        ui.show_error(f"モデル一覧の取得に失敗しました: {e}")
        return
    if not models:
        ui.show_warning("利用可能なモデルがありません")
        return

    selected_model = ui.ask_summary_model(models, config.llm.model)
    if selected_model is None:
        ui.show_info("キャンセルしました")
        return

    if selected_model == config.llm.model:
        ui.show_info("モデルは変更されていません")
        return

    config.llm.model = selected_model
    save_config(config, DEFAULT_CONFIG_PATH)
    ui.show_success(f"要約モデルを更新しました: {selected_model}")


def handle_init_config() -> None:
    """[7] 設定初期化"""
    if DEFAULT_CONFIG_PATH.exists():
        confirm = ui.ask_input(
            f"{DEFAULT_CONFIG_PATH} は既に存在します。上書きしますか? (y/n)",
            default="n",
        )
        if confirm.lower() != "y":
            ui.show_info("キャンセルしました")
            return

    save_default_config(DEFAULT_CONFIG_PATH)
    ui.show_success(f"設定ファイルを作成しました: {DEFAULT_CONFIG_PATH}")


def handle_reload_config() -> tuple[Config, bool]:
    """[8] 設定再読み込み"""
    if not DEFAULT_CONFIG_PATH.exists():
        ui.show_error(f"設定ファイルがありません: {DEFAULT_CONFIG_PATH}")
        return get_config(), False

    try:
        config = load_config(DEFAULT_CONFIG_PATH)
        ui.show_success("設定を再読み込みしました")
        return config, True
    except Exception as e:
        ui.show_error(f"設定の読み込みに失敗: {e}")
        return get_config(), False


def main() -> None:
    """メインエントリポイント"""
    # 設定読み込み
    config = get_config()

    # ロギング設定
    setup_logging(config.logging)
    logger = get_logger("main")
    logger.info("Voice Log 起動")

    # タイトル表示
    ui.show_title()

    # 起動診断
    diag_result = run_diagnostics(config=config)
    ui.show_diagnostics_result(diag_result)

    # モデル一覧表示
    if diag_result.ollama_ok and diag_result.ollama_models:
        ui.show_models(diag_result.ollama_models)

    # メインループ
    while True:
        try:
            choice = ui.show_menu()

            if choice == "0":
                ui.show_info("終了します")
                break
            elif choice == "1":
                handle_record_and_transcribe(config)
            elif choice == "2":
                handle_file_transcribe(config)
            elif choice == "3":
                handle_text_summary(config)
            elif choice == "4":
                handle_list_devices()
            elif choice == "5":
                handle_list_prompt_modes(config)
            elif choice == "6":
                handle_select_summary_model(config)
            elif choice == "7":
                handle_init_config()
            elif choice == "8":
                config, _ = handle_reload_config()
            else:
                ui.show_error("無効な選択です")

            ui.wait_for_enter()

        except KeyboardInterrupt:
            ui.show_info("\n中断しました")
            break
        except Exception as e:
            logger.exception("エラー")
            ui.show_error(f"エラー: {e}")
            ui.wait_for_enter()


if __name__ == "__main__":
    main()
