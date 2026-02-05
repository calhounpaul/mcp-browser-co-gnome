"""Tests for session management."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from novnc_automation.config import Config, RecordingConfig
from novnc_automation.session import SessionManager


@pytest.fixture
def tmp_sessions_dir(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return sessions_dir


@pytest.fixture
def session_manager(tmp_sessions_dir):
    config = Config(recording=RecordingConfig(sessions_dir=tmp_sessions_dir))
    return SessionManager(config=config)


@pytest.fixture
def mock_context(tmp_path):
    """Mock Playwright BrowserContext that writes a storage state file."""
    ctx = AsyncMock()

    async def fake_storage_state(path=None):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps({
                "cookies": [{"name": "sid", "value": "abc123"}],
                "origins": [],
            }))
        return {"cookies": [], "origins": []}

    ctx.storage_state = fake_storage_state
    return ctx


class TestSessionManager:
    async def test_save_and_load_session(self, session_manager, mock_context):
        state = await session_manager.save_session(
            session_id="test-sess",
            context=mock_context,
            url="https://example.com",
            metadata={"page": "home"},
        )
        assert state.session_id == "test-sess"
        assert state.url == "https://example.com"
        assert state.metadata == {"page": "home"}

        loaded = await session_manager.load_session_state("test-sess")
        assert loaded is not None
        assert loaded.session_id == "test-sess"
        assert loaded.url == "https://example.com"

    async def test_load_nonexistent_session(self, session_manager):
        result = await session_manager.load_session_state("no-such-session")
        assert result is None

    async def test_save_preserves_created_at(self, session_manager, mock_context):
        # First save
        state1 = await session_manager.save_session(
            session_id="test-sess", context=mock_context, url="https://example.com"
        )
        created_at = state1.created_at

        # Second save (update)
        state2 = await session_manager.save_session(
            session_id="test-sess", context=mock_context, url="https://example.com/page2"
        )
        assert state2.created_at == created_at
        assert state2.url == "https://example.com/page2"

    async def test_list_sessions(self, session_manager, mock_context):
        await session_manager.save_session("sess-1", mock_context, url="https://a.com")
        await session_manager.save_session("sess-2", mock_context, url="https://b.com")

        sessions = await session_manager.list_sessions()
        assert len(sessions) == 2
        ids = {s.session_id for s in sessions}
        assert ids == {"sess-1", "sess-2"}

    async def test_list_sessions_empty(self, session_manager):
        sessions = await session_manager.list_sessions()
        assert sessions == []

    async def test_delete_session(self, session_manager, mock_context):
        await session_manager.save_session("to-delete", mock_context)

        assert await session_manager.delete_session("to-delete") is True
        assert await session_manager.load_session_state("to-delete") is None

    async def test_delete_nonexistent_session(self, session_manager):
        assert await session_manager.delete_session("nope") is False

    async def test_get_storage_state(self, session_manager, mock_context):
        await session_manager.save_session("test-sess", mock_context)

        storage = await session_manager.get_storage_state("test-sess")
        assert storage is not None
        assert "cookies" in storage

    async def test_get_storage_state_nonexistent(self, session_manager):
        result = await session_manager.get_storage_state("nope")
        assert result is None

    async def test_get_storage_state_path(self, session_manager, mock_context):
        await session_manager.save_session("test-sess", mock_context)
        path = session_manager.get_storage_state_path("test-sess")
        assert path is not None
        assert path.exists()

    async def test_get_storage_state_path_nonexistent(self, session_manager):
        path = session_manager.get_storage_state_path("nope")
        assert path is None

    async def test_update_session_metadata(self, session_manager, mock_context):
        await session_manager.save_session("test-sess", mock_context, url="https://a.com")

        updated = await session_manager.update_session_metadata(
            "test-sess", url="https://b.com"
        )
        assert updated is not None
        assert updated.url == "https://b.com"

        # Verify persisted
        loaded = await session_manager.load_session_state("test-sess")
        assert loaded.url == "https://b.com"

    async def test_update_nonexistent_session(self, session_manager):
        result = await session_manager.update_session_metadata("nope", url="https://b.com")
        assert result is None
