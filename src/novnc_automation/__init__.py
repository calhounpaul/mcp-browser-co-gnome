"""noVNC Automation - Browser automation with noVNC visualization."""

from novnc_automation.browser import AutomationBrowser
from novnc_automation.session import SessionManager
from novnc_automation.recording import RecordingManager, log_global_action, get_global_action_log_path
from novnc_automation.tunnel import TunnelManager
from novnc_automation.config import Config
from novnc_automation.docker import DockerOrchestrator, DockerStatus, PortConfig, quick_start, quick_stop

__version__ = "0.1.0"

__all__ = [
    "AutomationBrowser",
    "SessionManager",
    "RecordingManager",
    "TunnelManager",
    "Config",
    "DockerOrchestrator",
    "DockerStatus",
    "PortConfig",
    "quick_start",
    "quick_stop",
    "log_global_action",
    "get_global_action_log_path",
]
