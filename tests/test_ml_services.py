"""Tests for ML service manager."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from novnc_automation.ml_services import (
    SERVICE_CONFIG,
    MLServiceManager,
    ServiceName,
    get_ml_manager,
)


@pytest.fixture
def manager(tmp_path):
    return MLServiceManager(compose_dir=tmp_path, idle_timeout=60)


class TestServiceName:
    def test_enum_values(self):
        assert ServiceName.OMNIPARSER.value == "omniparser"
        assert ServiceName.GUI_ACTOR.value == "gui-actor"
        assert ServiceName.VLM.value == "vlm"


class TestServiceConfig:
    def test_all_services_configured(self):
        for service in ServiceName:
            assert service in SERVICE_CONFIG
            cfg = SERVICE_CONFIG[service]
            assert "port" in cfg
            assert "health_endpoint" in cfg
            assert "startup_timeout" in cfg
            assert "compose_service" in cfg
            assert "compose_profile" in cfg

    def test_omniparser_config(self):
        cfg = SERVICE_CONFIG[ServiceName.OMNIPARSER]
        assert cfg["port"] == 8010
        assert cfg["startup_timeout"] == 300
        assert cfg["compose_profile"] == "ml"

    def test_vlm_config(self):
        cfg = SERVICE_CONFIG[ServiceName.VLM]
        assert cfg["port"] == 8004
        assert cfg["startup_timeout"] == 180
        assert cfg["compose_profile"] == "vlm"

    def test_gui_actor_config(self):
        cfg = SERVICE_CONFIG[ServiceName.GUI_ACTOR]
        assert cfg["port"] == 8001
        assert cfg["startup_timeout"] == 360
        assert cfg["compose_profile"] == "ml"


class TestMLServiceManagerInit:
    def test_default_idle_timeout(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ML_IDLE_TIMEOUT", raising=False)
        monkeypatch.delenv("ML_ALWAYS_ON", raising=False)
        mgr = MLServiceManager(compose_dir=tmp_path)
        assert mgr.idle_timeout == 300

    def test_idle_timeout_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ML_IDLE_TIMEOUT", "120")
        mgr = MLServiceManager(compose_dir=tmp_path)
        assert mgr.idle_timeout == 120

    def test_always_on_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ML_ALWAYS_ON", "vlm,omniparser")
        monkeypatch.delenv("ML_IDLE_TIMEOUT", raising=False)
        mgr = MLServiceManager(compose_dir=tmp_path)
        assert "vlm" in mgr.always_on
        assert "omniparser" in mgr.always_on

    def test_always_on_from_init(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ML_ALWAYS_ON", raising=False)
        monkeypatch.delenv("ML_IDLE_TIMEOUT", raising=False)
        mgr = MLServiceManager(compose_dir=tmp_path, always_on=["vlm"])
        assert "vlm" in mgr.always_on

    def test_always_on_merges_env_and_init(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ML_ALWAYS_ON", "vlm")
        monkeypatch.delenv("ML_IDLE_TIMEOUT", raising=False)
        mgr = MLServiceManager(compose_dir=tmp_path, always_on=["omniparser"])
        assert "vlm" in mgr.always_on
        assert "omniparser" in mgr.always_on

    def test_status_callback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ML_IDLE_TIMEOUT", raising=False)
        monkeypatch.delenv("ML_ALWAYS_ON", raising=False)
        callback = MagicMock()
        mgr = MLServiceManager(compose_dir=tmp_path, on_status_change=callback)
        mgr._log_status(ServiceName.VLM, "starting")
        callback.assert_called_once_with(ServiceName.VLM, "starting")


class TestMLServiceManagerHealthCheck:
    async def test_check_health_success(self, manager):
        with patch("novnc_automation.ml_services.httpx.AsyncClient") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"status": "healthy"}

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager._check_health(ServiceName.OMNIPARSER)
            assert result is True

    async def test_check_health_failure_wrong_status(self, manager):
        with patch("novnc_automation.ml_services.httpx.AsyncClient") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"status": "loading"}

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager._check_health(ServiceName.OMNIPARSER)
            assert result is False

    async def test_check_health_connection_error(self, manager):
        with patch("novnc_automation.ml_services.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager._check_health(ServiceName.OMNIPARSER)
            assert result is False

    async def test_is_healthy_delegates(self, manager):
        with patch.object(manager, "_check_health", return_value=True) as mock:
            result = await manager.is_healthy(ServiceName.VLM)
            assert result is True
            mock.assert_called_once_with(ServiceName.VLM)


class TestMLServiceManagerEnsure:
    async def test_ensure_already_healthy(self, manager):
        """If service is already healthy, just update tracking."""
        with patch.object(manager, "_check_health", return_value=True), \
             patch.object(manager, "_stop_service") as mock_stop:
            result = await manager.ensure_service(ServiceName.VLM)
            assert result is True
            assert manager._active_service == ServiceName.VLM
            # Should not have tried to stop VLM itself
            for call in mock_stop.call_args_list:
                assert call.args[0] != ServiceName.VLM

    async def test_ensure_stops_other_services(self, manager):
        """Starting one service should stop other healthy services."""
        health_responses = {
            ServiceName.OMNIPARSER: True,  # Running, should be stopped
            ServiceName.GUI_ACTOR: False,
            ServiceName.VLM: False,  # Will be started
        }

        async def fake_check_health(service):
            return health_responses.get(service, False)

        call_count = 0

        async def fake_check_health_evolving(service):
            nonlocal call_count
            # After start, VLM becomes healthy
            if service == ServiceName.VLM:
                call_count += 1
                return call_count > 1  # Unhealthy first check, healthy after start
            if service == ServiceName.OMNIPARSER:
                return True
            return False

        with patch.object(manager, "_check_health", side_effect=fake_check_health_evolving), \
             patch.object(manager, "_stop_service", return_value=True) as mock_stop, \
             patch.object(manager, "_start_service", return_value=True), \
             patch.object(manager, "_wait_for_health", return_value=True):
            result = await manager.ensure_service(ServiceName.VLM)
            assert result is True
            # OmniParser should have been stopped
            mock_stop.assert_any_call(ServiceName.OMNIPARSER)

    async def test_ensure_start_failure(self, manager):
        with patch.object(manager, "_check_health", return_value=False), \
             patch.object(manager, "_start_service", return_value=False):
            result = await manager.ensure_service(ServiceName.VLM)
            assert result is False

    async def test_ensure_health_timeout(self, manager):
        with patch.object(manager, "_check_health", return_value=False), \
             patch.object(manager, "_start_service", return_value=True), \
             patch.object(manager, "_wait_for_health", return_value=False):
            result = await manager.ensure_service(ServiceName.VLM)
            assert result is False


class TestMLServiceManagerStatus:
    async def test_get_status_all_down(self, manager):
        with patch.object(manager, "_check_health", return_value=False):
            status = await manager.get_status()
            assert len(status) == 3
            for name in ["omniparser", "gui-actor", "vlm"]:
                assert name in status
                assert status[name]["healthy"] is False
                assert status[name]["active"] is False

    async def test_get_status_one_active(self, manager):
        manager._active_service = ServiceName.VLM
        manager._last_used[ServiceName.VLM] = datetime.now()

        async def fake_health(service):
            return service == ServiceName.VLM

        with patch.object(manager, "_check_health", side_effect=fake_health):
            status = await manager.get_status()
            assert status["vlm"]["healthy"] is True
            assert status["vlm"]["active"] is True
            assert status["vlm"]["last_used"] is not None
            assert status["omniparser"]["healthy"] is False


class TestMLServiceManagerMonitor:
    async def test_start_and_stop(self, manager):
        await manager.start()
        assert manager._monitor_task is not None
        assert not manager._monitor_task.done()

        await manager.stop()
        assert manager._shutting_down is True
