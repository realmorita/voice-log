"""Tests for config module.

# TODO: テストリスト
# - [x] 正常系: デフォルト設定を生成できる
# - [x] 正常系: 設定ファイルを読み込める
# - [x] 正常系: 設定を再読み込みできる
# - [x] 境界値: 存在しないファイルでデフォルト設定を返す
# - [x] 異常系: 不正なYAMLでエラーを返す
# - [x] 正常系: 設定の特定キーにアクセスできる
# - [x] 正常系: 設定をファイルに保存できる
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from voice_log.config import Config, load_config, save_config, save_default_config


class TestConfig:
    """Config クラスのテスト"""

    def test_creates_default_config(self):
        """デフォルト設定を生成できる"""
        config = Config()

        assert config.audio.sample_rate == 16000
        assert config.audio.channels == 1
        assert config.whisper.model == "large-v3"
        assert config.whisper.device == "auto"
        assert config.whisper.condition_on_previous_text is False
        assert config.vad.enabled is True
        assert config.llm.enabled is True
        assert config.hallucination.enabled is True

    def test_loads_config_from_file(self):
        """設定ファイルを読み込める"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "audio": {"sample_rate": 48000},
                    "whisper": {"model": "turbo"},
                },
                f,
            )
            f.flush()

            config = load_config(Path(f.name))

            assert config.audio.sample_rate == 48000
            assert config.whisper.model == "turbo"
            # デフォルト値が維持される
            assert config.audio.channels == 1

    def test_returns_default_for_missing_file(self):
        """存在しないファイルでデフォルト設定を返す"""
        config = load_config(Path("/nonexistent/config.yaml"))

        assert config.audio.sample_rate == 16000
        assert config.whisper.model == "large-v3"

    def test_raises_error_for_invalid_yaml(self):
        """不正なYAMLでエラーを返す"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            with pytest.raises(yaml.YAMLError):
                load_config(Path(f.name))

    def test_reloads_config(self):
        """設定を再読み込みできる"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"audio": {"sample_rate": 16000}}, f)
            f.flush()

            config = load_config(Path(f.name))
            assert config.audio.sample_rate == 16000

            # ファイルを更新
            with open(f.name, "w") as f2:
                yaml.dump({"audio": {"sample_rate": 44100}}, f2)

            # 再読み込み
            config.reload(Path(f.name))
            assert config.audio.sample_rate == 44100


class TestSaveDefaultConfig:
    """save_default_config のテスト"""

    def test_saves_default_config_file(self):
        """デフォルト設定をファイルに保存できる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            save_default_config(config_path)

            assert config_path.exists()

            with open(config_path) as f:
                data = yaml.safe_load(f)

            assert data["audio"]["sample_rate"] == 16000
            assert data["whisper"]["model"] == "large-v3"


class TestSaveConfig:
    """save_config のテスト"""

    def test_saves_config_file(self):
        """設定をファイルに保存できる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = Config()
            config.llm.model = "test-model"

            save_config(config, config_path)

            with open(config_path) as f:
                data = yaml.safe_load(f)

            assert data["llm"]["model"] == "test-model"
