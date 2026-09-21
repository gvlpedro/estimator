"""Service-to-service auth for the AI service.

Being reachable only from the compose-internal network is not the same as
being safe to call from anything on that network -- any other container (or
anyone who gets a shell in one) could otherwise hit these endpoints too. The
business backend is the only intended caller, so it must present a shared
secret on every call. ``/health`` is exempt on purpose: the Docker
healthcheck runs unauthenticated from inside the container.

Reads ``AI_SERVICE_TOKEN`` at import time (not ``.get()``  with a default):
a missing token is a misconfiguration that should fail the service's
startup loudly, not silently accept unauthenticated requests.
"""

import hmac
import os

import structlog
from fastapi import Header, HTTPException

log = structlog.get_logger()

EXPECTED_TOKEN = os.environ["AI_SERVICE_TOKEN"]


def verify_service_token(x_service_token: str = Header(...)) -> None:
    # Constant-time compare: this is a shared secret, not a public ID --
    # a naive `==` leaks timing information about how many leading bytes
    # matched.
    if not hmac.compare_digest(x_service_token, EXPECTED_TOKEN):
        log.warning("ai_service_token_rejected")
        raise HTTPException(status_code=401, detail="Invalid service token")
