"""Tests for recording management."""

import json
from pathlib import Path

import pytest

from novnc_automation.config import Config, RecordingConfig
from novnc_automation.models.session_state import ActionEntry
from novnc_automation.recording import ActionLogger, RecordingManager, get_global_action_log_path


@pytest.fixture
def tmp_recordings_dir(tmp_path):
    return tmp_path / "recordings"


@pytest.fixture
def recording_config(tmp_recordings_dir):
    return Config(recording=RecordingConfig(recordings_dir=tmp_recordings_dir))


@pytest.fixture
def recorder(recording_config):
    return RecordingManager(session_id="test-session", config=recording_config)


class TestRecordingManager:
    def test_directory_creation(self, recorder, tmp_recordings_dir):
        assert (tmp_recordings_dir / "traces").is_dir()
        assert (tmp_recordings_dir / "har").is_dir()
        assert (tmp_recordings_dir / "actions").is_dir()
        assert (tmp_recordings_dir / "videos").is_dir()

    def test_path_properties(self, recorder, tmp_recordings_dir):
        assert recorder.trace_path == tmp_recordings_dir / "traces" / "test-session.zip"
        assert recorder.har_path == tmp_recordings_dir / "har" / "test-session.har"
        assert recorder.actions_path == tmp_recordings_dir / "actions" / "test-session.jsonl"
        assert recorder.video_path == tmp_recordings_dir / "videos" / "test-session.mp4"

    def test_get_har_options_enabled(self, recorder):
        opts = recorder.get_har_options()
        assert opts is not None
        assert "record_har_path" in opts
        assert opts["record_har_content"] == "embed"

    def test_get_har_options_disabled(self, tmp_recordings_dir):
        config = Config(recording=RecordingConfig(
            recordings_dir=tmp_recordings_dir,
            record_har=False,
        ))
        rec = RecordingManager(session_id="s", config=config)
        assert rec.get_har_options() is None

    async def test_start_action_log(self, recorder):
        await recorder.start_action_log()
        log = await recorder.get_action_log()
        assert log is not None
        assert log.session_id == "test-session"
        assert log.actions == []

    async def test_log_action_writes_to_file(self, recorder, monkeypatch):
        # Patch global action log to use temp dir
        monkeypatch.setattr(
            "novnc_automation.recording.GLOBAL_ACTION_LOG_DIR",
            recorder.recordings_dir / "data",
        )
        await recorder.start_action_log()
        await recorder.log_action(
            action="click",
            selector="#btn",
            url="https://example.com",
            duration_ms=15.5,
        )

        log = await recorder.get_action_log()
        assert len(log.actions) == 1
        assert log.actions[0].action == "click"
        assert log.actions[0].selector == "#btn"

        # Check file was written
        assert recorder.actions_path.exists()
        content = recorder.actions_path.read_text().strip()
        entry = json.loads(content)
        assert entry["action"] == "click"

    async def test_log_action_without_start_only_logs_global(self, recorder, monkeypatch):
        monkeypatch.setattr(
            "novnc_automation.recording.GLOBAL_ACTION_LOG_DIR",
            recorder.recordings_dir / "data",
        )
        # Don't call start_action_log - should still log globally
        await recorder.log_action(action="goto", url="https://example.com")

        # Per-session log should NOT exist
        assert not recorder.actions_path.exists()

        # Global log should exist
        global_dir = recorder.recordings_dir / "data"
        assert global_dir.exists()
        log_files = list(global_dir.glob("actions_*.jsonl"))
        assert len(log_files) == 1

    async def test_load_action_log(self, recorder, monkeypatch):
        monkeypatch.setattr(
            "novnc_automation.recording.GLOBAL_ACTION_LOG_DIR",
            recorder.recordings_dir / "data",
        )
        await recorder.start_action_log()
        await recorder.log_action(action="click", selector="#a")
        await recorder.log_action(action="fill", selector="#b", value="hello")

        # Load from file
        loaded = await recorder.load_action_log()
        assert loaded is not None
        assert len(loaded.actions) == 2
        assert loaded.actions[0].action == "click"
        assert loaded.actions[1].value == "hello"

    async def test_load_action_log_nonexistent(self, recorder):
        result = await recorder.load_action_log()
        assert result is None

    def test_get_recording_paths_empty(self, recorder):
        paths = recorder.get_recording_paths()
        assert all(v is None for v in paths.values())


class TestActionLogger:
    async def test_logs_success(self, recorder, monkeypatch):
        monkeypatch.setattr(
            "novnc_automation.recording.GLOBAL_ACTION_LOG_DIR",
            recorder.recordings_dir / "data",
        )
        await recorder.start_action_log()

        async with ActionLogger(recorder, action="click", selector="#btn"):
            pass  # Simulate successful action

        log = await recorder.get_action_log()
        assert len(log.actions) == 1
        assert log.actions[0].success is True
        assert log.actions[0].duration_ms is not None

    async def test_logs_failure_on_exception(self, recorder, monkeypatch):
        monkeypatch.setattr(
            "novnc_automation.recording.GLOBAL_ACTION_LOG_DIR",
            recorder.recordings_dir / "data",
        )
        await recorder.start_action_log()

        with pytest.raises(ValueError):
            async with ActionLogger(recorder, action="click", selector="#btn"):
                raise ValueError("element not found")

        log = await recorder.get_action_log()
        assert len(log.actions) == 1
        assert log.actions[0].success is False
        assert "element not found" in log.actions[0].error

    async def test_manual_error(self, recorder, monkeypatch):
        monkeypatch.setattr(
            "novnc_automation.recording.GLOBAL_ACTION_LOG_DIR",
            recorder.recordings_dir / "data",
        )
        await recorder.start_action_log()

        async with ActionLogger(recorder, action="fill", selector="#input") as al:
            al.set_error("validation failed")

        log = await recorder.get_action_log()
        assert log.actions[0].success is False
        assert log.actions[0].error == "validation failed"
