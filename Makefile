# Top-level Makefile.
#
# setup: installs uv (if missing) and syncs the workspace (root + ui).
# run:   stops anything on the target ports, then launches FastAPI and
#        Streamlit in parallel. Re-running is safe — previous instances are
#        killed first. Ctrl-C tears both down together.
# stop:  kills whatever currently holds the configured ports.

SHELL        := /bin/bash
BACKEND_PORT ?= 8000
UI_PORT      ?= 8501
PORTS        := $(BACKEND_PORT) $(UI_PORT)

.PHONY: setup run stop

setup:
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@PATH="$$HOME/.local/bin:$$PATH" uv sync --all-packages

stop:
	@for p in $(PORTS); do \
		pids=$$(lsof -ti tcp:$$p 2>/dev/null); \
		if [ -n "$$pids" ]; then \
			echo "Stopping PIDs on :$$p — $$pids"; \
			kill -9 $$pids 2>/dev/null || true; \
		fi; \
	done

run: stop
	@echo "Backend → http://localhost:$(BACKEND_PORT)    UI → http://localhost:$(UI_PORT)"
	@export PATH="$$HOME/.local/bin:$$PATH"; \
	uv run uvicorn app.main:app --host 0.0.0.0 --port $(BACKEND_PORT) --reload & backend_pid=$$!; \
	uv run streamlit run ui/streamlit_app.py --server.port $(UI_PORT) --server.headless true & ui_pid=$$!; \
	trap "kill $$backend_pid $$ui_pid 2>/dev/null" INT TERM EXIT; \
	while kill -0 $$backend_pid 2>/dev/null && kill -0 $$ui_pid 2>/dev/null; do sleep 1; done
