FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY ui/pyproject.toml ui/

RUN uv sync --all-packages --no-install-workspace --no-dev
FROM python:3.11-slim AS runtime

# Create a dedicated user and group
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --create-home appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY app/ /app/app/
RUN chown -R appuser:appgroup /app
ENV PATH="/app/.venv/bin:$PATH"
# Print logs immediately instead of buffering — critical for container logging.
ENV PYTHONUNBUFFERED=1
# Don't generate .pyc files at runtime; they add no value inside a container.
ENV PYTHONDONTWRITEBYTECODE=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
