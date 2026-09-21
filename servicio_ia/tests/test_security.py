"""Unit tests for service-to-service auth and the /health liveness route.

``app.security`` reads ``AI_SERVICE_TOKEN`` from the environment at import
time, so the fixture below must set it before the module is first imported
in this test process (see conftest.py's sys.path setup for why ``app``
resolves to ``servicio_ia/app`` here).
"""

import os

import pytest
from fastapi import HTTPException

os.environ.setdefault("AI_SERVICE_TOKEN", "test-shared-secret")

from app.security import verify_service_token  # noqa: E402


def test_verify_service_token_accepts_the_configured_token():
    verify_service_token(x_service_token="test-shared-secret")


def test_verify_service_token_rejects_wrong_token():
    with pytest.raises(HTTPException) as exc_info:
        verify_service_token(x_service_token="wrong-token")

    assert exc_info.value.status_code == 401


def test_verify_service_token_rejects_empty_token():
    with pytest.raises(HTTPException) as exc_info:
        verify_service_token(x_service_token="")

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_health_check_reports_healthy_without_a_token():
    from app.main import health_check

    result = await health_check()

    assert result == {"status": "healthy", "service": "servicio_ia", "version": "0.1.0"}
