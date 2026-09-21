# =============================================================================
# ai-service/Dockerfile — servicio_ia (embedding pipeline + RAG estimation)
# =============================================================================
# Build context is the REPO ROOT (see docker-compose.yml: context: ., dockerfile:
# ai-service/Dockerfile), not this directory. servicio_ia/ shares a single
# pyproject.toml/uv.lock with the estimator backend (see
# business-backend/Dockerfile) — both services are built from the same
# dependency set and only diverge in which source tree gets copied in.
#
# This service has NO published port in docker-compose: it is reachable only
# from other containers on the compose network, authenticated with
# AI_SERVICE_TOKEN (see servicio_ia/app/security.py). /health is the one
# unauthenticated route, used by this image's own HEALTHCHECK and by
# docker-compose's `depends_on: condition: service_healthy`.

# =============================================================================
# Stage 1 — Builder
# =============================================================================
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ui/pyproject.toml is required for uv to resolve the workspace (ui is a
# workspace member) even though this image never runs the UI.
COPY pyproject.toml uv.lock* ./
COPY ui/pyproject.toml ui/

RUN uv sync --all-packages --no-install-workspace --no-dev


# =============================================================================
# Stage 2 — Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --create-home appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Only the AI service's own source — not app/ or ui/, which have their own
# dedicated Dockerfile (business-backend) or stay on the shared root one.
COPY servicio_ia/ /app/servicio_ia/

RUN chown -R appuser:appgroup /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

USER appuser

# servicio_ia/app/main.py imports as `from app.xxx import ...`, so the
# working directory must be servicio_ia/ for `app` to resolve to
# servicio_ia/app (baked in here instead of relying on a compose override).
WORKDIR /app/servicio_ia

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
