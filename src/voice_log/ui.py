"""CLI UIモジュール

Rich を使用したCLI表示、メニュー操作を行う。
"""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from voice_log.prompts import PromptModeInfo
from voice_log import __version__
from voice_log.diagnostics import DiagnosticsResult

console = Console()


def show_title() -> None:
    """タイトルを表示する"""
    title = f"""
[bold cyan]Voice Log[/bold cyan] v{__version__}
[dim]ローカル音声文字起こし＋LLM要約ツール[/dim]
"""
    console.print(Panel(title, border_style="cyan"))


def show_diagnostics_result(result: DiagnosticsResult) -> None:
    """診断結果を表示する

    Args:
        result: 診断結果
    """
    table = Table(title="起動診断", show_header=True, header_style="bold magenta")
    table.add_column("項目", style="cyan")
    table.add_column("状態", style="green")
    table.add_column("詳細")

    # 設定
    config_status = "✓ OK" if result.config_ok else "✗ 失敗"
    config_style = "green" if result.config_ok else "red"
    table.add_row(
        "[1/3] 設定",
        f"[{config_style}]{config_status}[/{config_style}]",
        result.config_message,
    )

    # Ollama
    ollama_status = "✓ OK" if result.ollama_ok else "✗ 失敗"
    ollama_style = "green" if result.ollama_ok else "red"
    table.add_row(
        "[2/3] Ollama",
        f"[{ollama_style}]{ollama_status}[/{ollama_style}]",
        result.ollama_message,
    )

    # CUDA
    cuda_status = "✓ CUDA" if result.cuda_available else "○ CPU"
    cuda_style = "green" if result.cuda_available else "yellow"
    table.add_row(
        "[3/3] デバイス",
        f"[{cuda_style}]{cuda_status}[/{cuda_style}]",
        f"{result.device} ({result.compute_type})",
    )

    console.print(table)
    console.print()


def show_models(models: list[str]) -> None:
    """利用可能なモデル一覧を表示する

    Args:
        models: モデル名のリスト
    """
    if not models:
        console.print("[yellow]利用可能なモデルがありません[/yellow]")
        return

    table = Table(title="利用可能なモデル", show_header=True)
    table.add_column("#", style="dim")
    table.add_column("モデル名", style="cyan")

    for i, model in enumerate(models, 1):
        table.add_row(str(i), model)

    console.print(table)
    console.print()


def show_menu() -> str:
    """メインメニューを表示して選択を取得する

    Returns:
        str: ユーザーの選択
    """
    menu = """
[bold]操作を選択してください:[/bold]

  [cyan][1][/cyan] 録音して文字起こし（＋要約）
  [cyan][2][/cyan] ファイルから文字起こし（＋要約）
  [cyan][3][/cyan] テキストから要約のみ

  [cyan][4][/cyan] 録音デバイス一覧
  [cyan][5][/cyan] 要約モード選択
  [cyan][6][/cyan] 要約モデル選択

  [cyan][7][/cyan] 設定初期化
  [cyan][8][/cyan] 設定再読み込み

  [cyan][0][/cyan] 終了
"""
    console.print(menu)
    return Prompt.ask("選択", default="0")


def show_devices(devices: list[dict]) -> None:
    """録音デバイス一覧を表示する

    Args:
        devices: デバイス情報のリスト
    """
    if not devices:
        console.print("[yellow]録音デバイスが見つかりません[/yellow]")
        return

    table = Table(title="録音デバイス一覧", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("デバイス名")
    table.add_column("チャンネル", justify="right")
    table.add_column("サンプルレート", justify="right")

    for device in devices:
        table.add_row(
            str(device["id"]),
            device["name"],
            str(device["channels"]),
            f"{device['sample_rate']:.0f} Hz",
        )

    console.print(table)
    console.print()


def show_prompt_modes(modes: list[PromptModeInfo]) -> None:
    """要約モード一覧を表示する

    Args:
        modes: モード情報のリスト
    """
    if not modes:
        console.print("[yellow]要約モードが見つかりません[/yellow]")
        console.print(
            "[dim]prompts/ ディレクトリに .md ファイルを作成してください[/dim]"
        )
        return

    table = Table(title="要約モード一覧", show_header=True)
    table.add_column("#", style="dim")
    table.add_column("モード名", style="cyan")
    table.add_column("ファイル")

    for i, mode in enumerate(modes, 1):
        table.add_row(str(i), mode.display_name, f"prompts/{mode.mode_id}.md")

    console.print(table)
    console.print()


def ask_summary_mode(
    modes: list[PromptModeInfo],
    current_mode: str,
) -> str | None:
    """要約モードの選択を取得する

    Args:
        modes: モード情報のリスト
        current_mode: 現在のモード名

    Returns:
        str | None: 選択されたモード名。キャンセル時はNone。
    """
    if not modes:
        console.print("[yellow]要約モードが見つかりません[/yellow]")
        console.print(
            "[dim]prompts/ ディレクトリに .md ファイルを作成してください[/dim]"
        )
        return None

    table = Table(title="要約モード選択", show_header=True)
    table.add_column("#", style="dim")
    table.add_column("モード名", style="cyan")
    table.add_column("ファイル")
    table.add_column("現在", justify="center")

    for i, mode in enumerate(modes, 1):
        mark = "✓" if mode.mode_id == current_mode else ""
        table.add_row(str(i), mode.display_name, f"prompts/{mode.mode_id}.md", mark)

    console.print(table)
    current_display_name = current_mode
    for mode in modes:
        if mode.mode_id == current_mode:
            current_display_name = mode.display_name
            break
    console.print(f"[dim]現在のモード: {current_display_name}[/dim]")
    console.print("[dim]番号を入力してください。0でキャンセル。[/dim]")

    choices = [str(i) for i in range(0, len(modes) + 1)]
    selected = Prompt.ask("選択", default="0", choices=choices)
    if selected == "0":
        return None

    return modes[int(selected) - 1].mode_id


def ask_summary_model(models: list[str], current_model: str) -> str | None:
    """要約モデルの選択を取得する

    Args:
        models: モデル名のリスト
        current_model: 現在のモデル名

    Returns:
        str | None: 選択されたモデル名。キャンセル時はNone。
    """
    if not models:
        console.print("[yellow]利用可能なモデルがありません[/yellow]")
        return None

    table = Table(title="要約モデル選択", show_header=True)
    table.add_column("#", style="dim")
    table.add_column("モデル名", style="cyan")
    table.add_column("現在", justify="center")

    for i, model in enumerate(models, 1):
        mark = "✓" if model == current_model else ""
        table.add_row(str(i), model, mark)

    console.print(table)
    console.print(f"[dim]現在のモデル: {current_model}[/dim]")
    console.print("[dim]番号を入力してください。0でキャンセル。[/dim]")

    choices = [str(i) for i in range(0, len(models) + 1)]
    selected = Prompt.ask("選択", default="0", choices=choices)
    if selected == "0":
        return None

    return models[int(selected) - 1]


def show_progress(message: str):
    """進捗スピナーを表示するコンテキストマネージャー

    Args:
        message: 表示メッセージ

    Returns:
        Progress: Richの Progress オブジェクト
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    )


def show_success(message: str) -> None:
    """成功メッセージを表示する"""
    console.print(f"[green]✓[/green] {message}")


def show_error(message: str) -> None:
    """エラーメッセージを表示する"""
    console.print(f"[red]✗[/red] {message}")


def show_warning(message: str) -> None:
    """警告メッセージを表示する"""
    console.print(f"[yellow]⚠[/yellow] {message}")


def show_info(message: str) -> None:
    """情報メッセージを表示する"""
    console.print(f"[blue]ℹ[/blue] {message}")


def ask_input(prompt: str, default: str = "") -> str:
    """ユーザー入力を取得する

    Args:
        prompt: プロンプト
        default: デフォルト値

    Returns:
        str: ユーザーの入力
    """
    return Prompt.ask(prompt, default=default)


def ask_file_path(prompt: str = "ファイルパスを入力") -> Path | None:
    """ファイルパスを取得する

    Args:
        prompt: プロンプト

    Returns:
        Path | None: ファイルパス、キャンセル時はNone
    """
    path_str = Prompt.ask(prompt)
    if not path_str:
        return None

    path = Path(path_str.strip().strip('"').strip("'"))
    if not path.exists():
        show_error(f"ファイルが見つかりません: {path}")
        return None

    return path


def ask_multiline_text(prompt: str = "テキストを入力 (空行で終了)") -> str:
    """複数行テキストを取得する

    Args:
        prompt: プロンプト

    Returns:
        str: 入力されたテキスト
    """
    console.print(f"[cyan]{prompt}[/cyan]")
    lines = []

    while True:
        try:
            line = input()
            if not line:
                break
            lines.append(line)
        except EOFError:
            break

    return "\n".join(lines)


def wait_for_enter(prompt: str = "Enterで続行...") -> None:
    """Enterキー待ち

    Args:
        prompt: プロンプト
    """
    console.print(f"\n[dim]{prompt}[/dim]")
    input()
