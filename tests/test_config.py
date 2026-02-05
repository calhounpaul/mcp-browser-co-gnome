"""Tests for configuration loading."""

import os
from pathlib import Path

import pytest
import yaml

from novnc_automation.config import (
    BrowserConfig,
    Config,
    DockerConfig,
    MLServicesConfig,
    RecordingConfig,
    TunnelConfig,
)


class TestBrowserConfig:
    def test_defaults(self):
        cfg = BrowserConfig()
        assert cfg.headless is False
        assert cfg.stealth_mode is True
        assert cfg.viewport_width == 1920
        assert cfg.viewport_height == 1080
        assert cfg.user_agent is None
        assert cfg.extra_args == []


class TestRecordingConfig:
    def test_defaults(self):
        cfg = RecordingConfig()
        assert cfg.record_video is True
        assert cfg.record_trace is True
        assert cfg.record_har is True
        assert cfg.record_actions is True
        assert cfg.recordings_dir == Path("tmp")
        assert cfg.sessions_dir == Path("tmp/sessions")


class TestMLServicesConfig:
    def test_defaults(self):
        cfg = MLServicesConfig()
        assert cfg.idle_timeout == 300
        assert cfg.always_on == []


class TestConfigFromEnv:
    def test_defaults_without_env(self, monkeypatch):
        # Clear all relevant env vars
        for var in [
            "HEADLESS", "STEALTH_MODE", "VIEWPORT_WIDTH", "VIEWPORT_HEIGHT",
            "USER_AGENT", "RECORD_VIDEO", "RECORD_TRACE", "RECORD_HAR",
            "RECORD_ACTIONS", "TMP_DIR", "RECORDINGS_DIR", "SESSIONS_DIR",
            "ENABLE_TUNNEL", "TUNNEL_PORT", "VNC_PASSWORD", "RESOLUTION",
            "NOVNC_PORT", "VNC_PORT", "CDP_PORT",
        ]:
            monkeypatch.delenv(var, raising=False)

        cfg = Config.from_env()
        assert cfg.browser.headless is False
        assert cfg.browser.stealth_mode is True
        assert cfg.docker.vnc_password == "secret"
        assert cfg.tunnel.enable_tunnel is False

    def test_headless_from_env(self, monkeypatch):
        monkeypatch.setenv("HEADLESS", "true")
        cfg = Config.from_env()
        assert cfg.browser.headless is True

    def test_viewport_from_env(self, monkeypatch):
        monkeypatch.setenv("VIEWPORT_WIDTH", "800")
        monkeypatch.setenv("VIEWPORT_HEIGHT", "600")
        cfg = Config.from_env()
        assert cfg.browser.viewport_width == 800
        assert cfg.browser.viewport_height == 600

    def test_tunnel_from_env(self, monkeypatch):
        monkeypatch.setenv("ENABLE_TUNNEL", "true")
        monkeypatch.setenv("TUNNEL_PORT", "7070")
        cfg = Config.from_env()
        assert cfg.tunnel.enable_tunnel is True
        assert cfg.tunnel.tunnel_port == 7070

    def test_docker_from_env(self, monkeypatch):
        monkeypatch.setenv("VNC_PASSWORD", "mypass")
        monkeypatch.setenv("RESOLUTION", "1280x720x16")
        cfg = Config.from_env()
        assert cfg.docker.vnc_password == "mypass"
        assert cfg.docker.resolution == "1280x720x16"

    def test_tmp_dir_overrides_recordings_dir(self, monkeypatch):
        monkeypatch.setenv("TMP_DIR", "/custom/tmp")
        cfg = Config.from_env()
        assert cfg.recording.recordings_dir == Path("/custom/tmp")


class TestConfigFromYaml:
    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = Config.from_yaml(tmp_path / "nonexistent.yml")
        assert cfg.browser.headless is False
        assert cfg.docker.vnc_password == "secret"

    def test_loads_yaml(self, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({
            "browser": {"headless": True, "viewport_width": 800},
            "docker": {"vnc_password": "test123"},
        }))
        cfg = Config.from_yaml(config_file)
        assert cfg.browser.headless is True
        assert cfg.browser.viewport_width == 800
        assert cfg.docker.vnc_password == "test123"
        # Defaults preserved for unset fields
        assert cfg.browser.stealth_mode is True

    def test_empty_yaml(self, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text("")
        cfg = Config.from_yaml(config_file)
        assert cfg.browser.headless is False


class TestConfigLoad:
    def test_env_overrides_yaml(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({
            "browser": {"headless": False, "viewport_width": 800},
        }))
        monkeypatch.setenv("HEADLESS", "true")
        cfg = Config.load(config_file)
        # Env should override yaml
        assert cfg.browser.headless is True
        # Non-overridden values preserved from yaml
        assert cfg.browser.viewport_width == 800


class TestConfigToYaml:
    def test_saves_file(self, tmp_path):
        cfg = Config(
            browser=BrowserConfig(headless=True, viewport_width=800),
            docker=DockerConfig(vnc_password="test"),
        )
        out = tmp_path / "out.yml"
        cfg.to_yaml(out)
        assert out.exists()
        # UnsafeLoader needed because Path objects serialize with python-specific tags
        content = yaml.load(out.read_text(), Loader=yaml.UnsafeLoader)
        assert content["browser"]["headless"] is True
        assert content["browser"]["viewport_width"] == 800
        assert content["docker"]["vnc_password"] == "test"

    def test_creates_parent_dirs(self, tmp_path):
        cfg = Config()
        out = tmp_path / "subdir" / "nested" / "config.yml"
        cfg.to_yaml(out)
        assert out.exists()
