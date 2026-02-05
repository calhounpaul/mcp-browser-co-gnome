"""Tests for Pydantic models."""

from datetime import datetime

from novnc_automation.models.session_state import ActionEntry, ActionLog, SessionState


class TestActionEntry:
    def test_defaults(self):
        entry = ActionEntry(action="click")
        assert entry.action == "click"
        assert entry.success is True
        assert entry.selector is None
        assert entry.error is None
        assert entry.metadata == {}
        assert isinstance(entry.timestamp, datetime)

    def test_full_entry(self):
        entry = ActionEntry(
            action="fill",
            selector="#email",
            value="test@example.com",
            url="https://example.com",
            duration_ms=42.5,
            success=True,
        )
        assert entry.selector == "#email"
        assert entry.value == "test@example.com"
        assert entry.duration_ms == 42.5

    def test_failed_entry(self):
        entry = ActionEntry(action="click", success=False, error="Element not found")
        assert entry.success is False
        assert entry.error == "Element not found"

    def test_serialization_roundtrip(self):
        entry = ActionEntry(action="goto", url="https://example.com")
        json_str = entry.model_dump_json()
        restored = ActionEntry.model_validate_json(json_str)
        assert restored.action == "goto"
        assert restored.url == "https://example.com"


class TestActionLog:
    def test_creation(self):
        log = ActionLog(session_id="test-session")
        assert log.session_id == "test-session"
        assert log.actions == []

    def test_add_action(self):
        log = ActionLog(session_id="test-session")
        entry = ActionEntry(action="click", selector="#btn")
        log.add_action(entry)
        assert len(log.actions) == 1
        assert log.actions[0].selector == "#btn"

    def test_multiple_actions(self):
        log = ActionLog(session_id="test-session")
        log.add_action(ActionEntry(action="goto", url="https://example.com"))
        log.add_action(ActionEntry(action="click", selector="#btn"))
        log.add_action(ActionEntry(action="fill", selector="#input", value="hello"))
        assert len(log.actions) == 3
        assert log.actions[0].action == "goto"
        assert log.actions[2].value == "hello"


class TestSessionState:
    def test_defaults(self):
        state = SessionState(session_id="sess-1")
        assert state.session_id == "sess-1"
        assert state.url is None
        assert state.viewport == {"width": 1920, "height": 1080}
        assert state.metadata == {}
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.updated_at, datetime)

    def test_touch_updates_timestamp(self):
        state = SessionState(session_id="sess-1")
        old_updated = state.updated_at
        # Force a small delay via manual override
        state.updated_at = datetime(2020, 1, 1)
        state.touch()
        assert state.updated_at > datetime(2020, 1, 1)

    def test_serialization_roundtrip(self):
        state = SessionState(
            session_id="sess-1",
            url="https://example.com",
            metadata={"key": "value"},
        )
        json_str = state.model_dump_json()
        restored = SessionState.model_validate_json(json_str)
        assert restored.session_id == "sess-1"
        assert restored.url == "https://example.com"
        assert restored.metadata == {"key": "value"}

    def test_with_recording_paths(self):
        state = SessionState(
            session_id="sess-1",
            trace_path="/tmp/trace.zip",
            har_path="/tmp/req.har",
            video_path="/tmp/video.mp4",
        )
        assert state.trace_path == "/tmp/trace.zip"
        assert state.har_path == "/tmp/req.har"
        assert state.video_path == "/tmp/video.mp4"
