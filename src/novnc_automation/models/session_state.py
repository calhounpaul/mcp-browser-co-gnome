"""Pydantic models for session state serialization."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ActionEntry(BaseModel):
    """Single action logged during browser automation."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: str
    selector: str | None = None
    value: str | None = None
    url: str | None = None
    screenshot_path: str | None = None
    duration_ms: float | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionLog(BaseModel):
    """Collection of actions for a session."""

    session_id: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    actions: list[ActionEntry] = Field(default_factory=list)

    def add_action(self, action: ActionEntry) -> None:
        """Add an action to the log."""
        self.actions.append(action)


class SessionState(BaseModel):
    """Complete session state for persistence."""

    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    url: str | None = None
    storage_state_path: str | None = None
    trace_path: str | None = None
    har_path: str | None = None
    video_path: str | None = None
    action_log_path: str | None = None
    viewport: dict[str, int] = Field(default_factory=lambda: {"width": 1920, "height": 1080})
    user_agent: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()
