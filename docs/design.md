# 設計書：完全ローカル音声文字起こし＋ローカルLLM要約（Python CLI）

## 0. 概要
本ツールは **完全にローカル環境のみ**で、(1) 音声→文字起こし（CUDA対応で高速化、VADで無音スキップ、長時間対応）を行い、(2) 生成した文字起こしを **ローカルLLM（Ollama）** で要約します。  
要約プロンプトは `prompts/*.md` として外出しし、運用上の差し替え・拡張を容易にします。

## 1. 目的・非機能方針（要点）
- ローカル完結（ネット遮断でも完走）
- CUDAが使える場合は自動利用、不可ならCPUへフォールバック
- 長時間音声でも完走（チャンク処理・ログ・中間保存）
- 反復（ループ）検知・抑制を「既定」で有効化
- プロンプトはMarkdownでモード切替（Skills運用）

## 2. 採用ライブラリと責務（外部依存）
|区分|ライブラリ|責務|
|---|---|---|
|ASR/整列|stable-ts（`stable-ts[fw]`）|faster-whisperと統合した前処理（VAD等）＋後処理（regroup、timestamp調整、suppress_silence等）|
|ASR推論|faster-whisper|Whisper推論（CTranslate2で高速）|
|推論基盤|CTranslate2|Transformer推論エンジン|
|録音|python-sounddevice|録音デバイス列挙・録音（PortAudio）|
|WAV I/O|python-soundfile|録音データのWAV保存・読み込み|
|変換|FFmpeg（外部コマンド）|mp3/m4a等→wav変換（任意、stable-ts側でも必要になり得る）|
|要約|OpenAI Python SDK（`openai`）|Ollamaローカル推論（チャット/生成、モデル一覧）|
|設定|PyYAML|config.yamlのロード/ダンプ|
|CLI表示|Rich|スピナー/進捗/整形表示|

## 3. ディレクトリ構成（提案）
```
project/
  app.py
  config.yaml
  prompts/
    minutes.md
    todo.md
    summary_3lines.md
  outputs/
  logs/
  src/
    __init__.py
    config.py
    ui.py
    diagnostics.py
    audio_io.py
    transcribe.py
    hallucination.py
    prompts.py
    summarize.py
    output.py
```

## 4. 設定（config.yaml）スキーマ（提案）
```yaml
audio:
  sample_rate: 16000
  channels: 1
  device_id: null
  dtype: float32

input:
  tmp_dir: ".tmp"
  accept_ext: ["wav","mp3","m4a","flac"]

vad:
  enabled: true
  # faster-whisperのvad_filter相当。stable-ts側に渡す
  min_silence_duration_ms: 500

whisper:
  model: "large-v3"           # 例: "large-v3" / "turbo" / ローカルパス
  language: "ja"
  device: "auto"              # auto|cuda|cpu
  compute_type: "auto"        # auto|float16|int8_float16|int8
  beam_size: 3
  temperature: 0.0
  condition_on_previous_text: false
  word_timestamps_internal: false   # stable-ts機能を使う場合のみtrue推奨
  batch_size: 0               # 0なら通常transcribe、>0ならbatched

stable_ts:
  enabled: true
  suppress_silence: true
  regroup: true

hallucination:
  enabled: true
  max_consecutive_token_repeat: 8
  max_consecutive_line_repeat: 3
  ngram_size: 4
  ngram_repeat_threshold: 6
  action: "trim"              # trim|warn_only|redecode

llm:
  enabled: true
  provider: "openai_compatible"  # openai_compatible
  base_url: "http://127.0.0.1:11434"
  model: "qwen2.5:14b"
  temperature: 0.2
  max_output_tokens: 1024
  timeout_sec: 120
  prompt_mode: "minutes"      # prompts/<mode>.md

output:
  out_dir: "outputs"
  formats:
    transcript: ["md","txt"]
    summary: ["md","txt"]
  naming: "{date}_{time}_{stem}"
  meta_footer: true

logging:
  level: "INFO"
  file: "logs/app.log"
  faster_whisper_debug: false
```

## 5. 起動時自己診断（FR-1）
### 5.1 手順
1. `config.yaml` 読込・必須キー検証
2. Ollama疎通（OpenAI互換: `GET /v1/models` もしくは OpenAI SDK の models.list 相当）
3. CUDA可否判定（「使えるか」を実際にモデル初期化で判定し、失敗ならCPUへ）
4. 主要設定・利用デバイス・モデル名を表示

### 5.2 実装要点
- `device=auto` の場合：
  - まず `device="cuda"` でモデル初期化を試みる
  - 失敗（CUDA関連例外）なら `device="cpu"` にフォールバック
- `compute_type=auto` の場合：
  - cuda: `float16`（VRAM節約したいなら `int8_float16`）
  - cpu: `int8`

## 6. CLI（メニュー）設計（FR-2）
```
[1] 録音して文字起こし（＋要約）
[2] ファイルから文字起こし（＋要約）
[3] テキストから要約のみ
[4] 録音デバイス一覧
[5] 要約モード一覧
[6] 設定初期化
[7] 設定再読み込み
[0] 終了
```
- 入力は `input()` ベースで実装（依存を増やさない）
- 表示・進捗はRich（Status/Progress）で統一

## 7. 録音（FR-3）
### 7.1 方式
- `sounddevice.InputStream` を用いた「開始→停止」録音（Enterで開始、Enterで停止）
- コールバックでフレームをバッファし、停止後に1つのNumPy配列へ結合
- `soundfile.write()` で一時WAVへ保存

### 7.2 擬似コード
```python
frames = []
with sd.InputStream(samplerate=sr, channels=ch, device=device_id, callback=cb):
    wait_until_stop_key()
audio = np.concatenate(frames, axis=0)
sf.write(tmp_wav_path, audio, sr)
```

## 8. 音声ファイル入力（FR-4）
### 8.1 変換ポリシー
- 入力がwav以外の場合、FFmpegで `16kHz / mono / wav` に正規化してからASRへ渡す（再現性・性能・VAD品質を安定化）
- stable-ts側がURL/多形式入力を処理できる場合でも、ツール側で統一するとトラブルシュートが容易

### 8.2 FFmpeg変換コマンド例
```bash
ffmpeg -y -i "in.m4a" -ac 1 -ar 16000 "normalized.wav"
```

## 9. 文字起こし（FR-6）と stable-ts（FR-7）
### 9.1 エンジン選択
- `stable_ts.enabled=true` の場合：
  - `stable_whisper.load_faster_whisper(model)` でモデルロード
  - `model.transcribe()` を stable-ts 側の拡張引数込みで実行
- `stable_ts.enabled=false` の場合：
  - `faster_whisper.WhisperModel` を直接利用（最小依存）

### 9.2 推奨デフォルト（ハルシネーション抑制）
- `condition_on_previous_text=False`
- `temperature=0.0~0.2`
- VAD有効（無音部を削ることでループ誘発を低減）
- stable-ts `suppress_silence` 有効（環境によりオフ可能）

### 9.3 長時間対応
- stable-ts／faster-whisperの「チャンク処理（内部）」に加え、ツール側で中間結果を一定間隔で保存（例：5分ごとに追記）
- 途中失敗でもリカバリできるよう、`outputs/<session>/partial_*.json` を保持

## 10. 反復（ループ）検知・抑制（NFR-3）
### 10.1 検知対象
- 同一トークン（単語）連続
- 同一行連続
- n-gram（例：4-gram）の過剰反復

### 10.2 判定アルゴリズム（例）
- 文字起こし結果を正規化（全角/半角、空白圧縮）
- token列に対し、直近Wの窓で「同一tokenの最大連続長」を算出
- line列に対し、同一行の連続回数を算出
- n-gram辞書で末尾から連続反復回数を算出

### 10.3 アクション
- `trim`（既定）：末尾の反復ブロックをトリムし、警告ログ＋メタ情報に残す
- `warn_only`：警告のみ
- `redecode`（オプション）：反復が起きたタイムスタンプ範囲の音声を切り出し、パラメータ（beam/temperature）を変えて再推論し差し替え

## 11. 要約（FR-9）とプロンプト管理（FR-10）
### 11.1 実行フロー
1. `prompts/<mode>.md` を読み込み
2. `{{TRANSCRIPT}}` 等を置換して最終プロンプト生成
3. Ollama `client.chat()` または `ollama.generate()` で実行
4. 失敗時は「要約スキップ」で部分成功とする（文字起こしは保存）

### 11.2 プレースホルダ
- `{{TRANSCRIPT}}`（必須）
- `{{TITLE}}`（任意）
- `{{DATE}}`
- `{{LANG}}`

### 11.3 長文対策（オプション）
- コンテキスト超過が疑われる場合：
  - チャンク要約（例：3,000〜6,000文字単位）
  - チャンク要約群を統合要約（2段構え）

## 12. 出力（FR-8）
### 12.1 ファイル
- `*_transcript.md` / `*.txt`
- `*_summary.md` / `*.txt`
- `*.json`（セグメント、処理時間、警告、設定要約）

### 12.2 メタ情報（フッター例）
- ASRモデル名、device、compute_type、beam_size、temperature
- VAD/stable-ts有効/無効
- 処理時間、音声長、RTF
- 反復検知件数・トリム有無

## 13. ロギング／運用（NFR-5）
- `logs/app.log` にINFO/DEBUGを出力
- 例外は「どのステップで失敗したか」を必ず記録（例：[2/3] Ollama確認で失敗）
- faster-whisper内部ログは必要に応じてDEBUGへ

## 14. テスト観点（Acceptance Criteria対応）
- ネット遮断状態で：録音→文字起こし→要約まで完走
- `prompts/*.md` 修正後、要約のみ再生成で差分反映
- 2時間音声で完走（GPU/CPU）
- 反復しやすい音声で検知ログ＋抑制が働く
- [1]〜[7] メニューの成功・失敗復帰（部分成功）

## 15. 参考（公式ドキュメント）
- stable-ts: https://github.com/jianfch/stable-ts
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- Ollama OpenAI互換API: https://docs.ollama.com/api/openai-compatibility
- OpenAI Python SDK: https://github.com/openai/openai-python
- OpenAI API Reference: https://platform.openai.com/docs/api-reference/introduction
- python-sounddevice: https://python-sounddevice.readthedocs.io/
- python-soundfile: https://python-soundfile.readthedocs.io/
- Rich: https://rich.readthedocs.io/
- PyYAML: https://pyyaml.org/wiki/PyYAMLDocumentation
- FFmpeg: https://ffmpeg.org/ffmpeg.html


## 11.1 要約（OpenAI互換API経由：Ollamaを含む）
### 方針
- 要約生成は **OpenAI Python SDK（openai-python）** を使用し、`base_url` を切り替えることで **OpenAI互換サーバ**（当面はOllama）に接続する。
- Ollama側は OpenAI互換エンドポイント（例：`http://127.0.0.1:11434/v1`）を提供する。
- 互換APIは **/v1/responses**（推奨）および **/v1/chat/completions** を利用可能とし、設定で切替できる。

### 実装イメージ（Responses API）
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",  # 必須だがOllamaでは未使用
    timeout=120.0,
    max_retries=0,
)

resp = client.responses.create(
    model="llama3.2",
    instructions="あなたは議事録の要約アシスタントです。",
    input=final_prompt_text,
    temperature=0.2,
    max_output_tokens=1024,
)

summary_text = resp.output_text
```

### 実装イメージ（Chat Completions API）
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
)

completion = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "あなたは議事録の要約アシスタントです。"},
        {"role": "user", "content": final_prompt_text},
    ],
    temperature=0.2,
    max_tokens=1024,
)

summary_text = completion.choices[0].message.content
```

### 注意点（Ollama互換の制約）
- /v1/responses は **非ステートフル**（`previous_response_id` / `conversation` を用いた状態保持は非対応）であるため、ツール側は「毎回プロンプトを完結」させる。
- モデル一覧は `/v1/models` を第一選択とし、取得失敗時は「Ollamaのバージョンが古い」可能性をユーザーに提示する。
