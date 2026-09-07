# Top-level Makefile.
#
# setup:  installs uv (if missing) and syncs the workspace (root + ui).
# run:    stops anything on the target ports, then launches FastAPI and
#         Streamlit in parallel. Re-running is safe — previous instances are
#         killed first. Ctrl-C tears both down together.
# stop:   kills whatever currently holds the configured ports.
# stress: runs the stress harness against the in-process app and writes
#         evals/stress/results.csv + evals/stress/REPORT.md.
# docker_run: idempotent Docker bring-up of backend + servicio_ia + ui
#         (stops previous stack and stray port holders first), then prints
#         how to reach each service.

SHELL            := /bin/bash
BACKEND_PORT     ?= 8000
UI_PORT          ?= 8501
SERVICIO_IA_PORT ?= 8001
PORTS            := $(BACKEND_PORT) $(UI_PORT) $(SERVICIO_IA_PORT)

STRESS_SCENARIOS        ?= growing,pivot,contradiction
STRESS_ATTACHMENT_SIZES ?= 5KB,20KB,50KB,100KB
STRESS_TURNS            ?= 6
STRESS_REPEATS          ?= 3
STRESS_LATENCY_BUDGET   ?= 8000
STRESS_COST_BUDGET      ?= 0.05
STRESS_OUTPUT           ?= evals/stress/results.csv
STRESS_REPORT           ?= evals/stress/REPORT.md

.PHONY: setup run stop stress docker_run db_upgrade seed_ia

setup:
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@PATH="$$HOME/.local/bin:$$PATH" uv sync --all-packages

# Never kill Docker Desktop's own processes: on macOS the published container
# ports are held by com.docker.backend, and kill -9 on it takes down the whole
# Docker daemon. Container ports are released via `docker compose down`.
stop:
	@for p in $(PORTS); do \
		for pid in $$(lsof -ti tcp:$$p 2>/dev/null); do \
			name=$$(ps -p $$pid -o comm= 2>/dev/null); \
			case "$$name" in \
				*[Dd]ocker*) echo "Skipping Docker process $$pid ($$name) on :$$p";; \
				*) echo "Stopping PID $$pid ($$name) on :$$p"; kill -9 $$pid 2>/dev/null || true;; \
			esac; \
		done; \
	done

run: stop
	@echo "Backend → http://localhost:$(BACKEND_PORT)    UI → http://localhost:$(UI_PORT)"
	@export PATH="$$HOME/.local/bin:$$PATH"; \
	uv run uvicorn app.main:app --host 0.0.0.0 --port $(BACKEND_PORT) --reload & backend_pid=$$!; \
	uv run streamlit run ui/streamlit_app.py --server.port $(UI_PORT) --server.headless true & ui_pid=$$!; \
	trap "kill $$backend_pid $$ui_pid 2>/dev/null" INT TERM EXIT; \
	while kill -0 $$backend_pid 2>/dev/null && kill -0 $$ui_pid 2>/dev/null; do sleep 1; done

start_alembic:
	alembic init -t async alembic
	echo "Delete folder alembic/ and alembic.ini to reinitialize the migration environment."
	echo "Configuration is included in env.py and /versions, then you can use 'alembic upgrade head' to start"
	echo "Execute 'alembic history' to check migrations"

# Idempotent Docker bring-up: frees the published ports from stray local
# processes, tears down any previous compose stack, rebuilds and starts the
# backend (app), the AI service (servicio_ia) and the UI, then waits for
# their health endpoints before printing how to reach each service.
# Order matters: compose down FIRST (it releases the ports Docker holds),
# THEN stop (it clears stray local processes only — Docker's are skipped).
docker_run:
	@docker compose down --remove-orphans
	@$(MAKE) stop
	@docker compose up --build --detach
	@echo "Waiting for services to become healthy..."
	@for i in $$(seq 1 60); do \
		backend=$$(curl -sf -o /dev/null http://localhost:$(BACKEND_PORT)/health && echo ok || echo ko); \
		ia=$$(curl -sf -o /dev/null http://localhost:$(SERVICIO_IA_PORT)/health && echo ok || echo ko); \
		ui=$$(curl -sf -o /dev/null http://localhost:$(UI_PORT)/_stcore/health && echo ok || echo ko); \
		if [ "$$backend$$ia$$ui" = "okokok" ]; then break; fi; \
		sleep 2; \
	done; \
	echo ""; \
	echo "=========================================================="; \
	echo "  Servicios levantados:"; \
	echo "  Backend (app)  [$$backend]  http://localhost:$(BACKEND_PORT)  —  Swagger: http://localhost:$(BACKEND_PORT)/docs"; \
	echo "  Servicio IA    [$$ia]  http://localhost:$(SERVICIO_IA_PORT)  —  Swagger: http://localhost:$(SERVICIO_IA_PORT)/docs"; \
	echo "  UI (Streamlit) [$$ui]  http://localhost:$(UI_PORT)"; \
	echo "  Redis                localhost:6379 (interno)"; \
	echo "=========================================================="; \
	if [ "$$backend$$ia$$ui" != "okokok" ]; then \
		echo "  AVISO: algun servicio no responde aun (ko). Revisa: docker compose logs -f"; \
	fi

docker_check:
	docker compose exec postgres psql -U estimator -d estimator -c "SELECT version();"

# Apply the AI service schema (documents + chunks + pgvector extension) to
# the postgres of docker-compose. Requires the postgres service up.
db_upgrade:
	@export PATH="$$HOME/.local/bin:$$PATH"; uv run alembic upgrade head

# Seed the vector store with the 15 historical budgets of the sample corpus.
# Idempotent (409 = already ingested). Requires the AI service up (:8001)
# and the schema applied (make db_upgrade).
seed_ia:
	@export PATH="$$HOME/.local/bin:$$PATH"; uv run python servicio_ia/scripts/seed_budgets.py

stress:
	@export PATH="$$HOME/.local/bin:$$PATH"; \
	uv run python -m evals.stress.run \
		--scenarios $(STRESS_SCENARIOS) \
		--attachment-sizes $(STRESS_ATTACHMENT_SIZES) \
		--turns $(STRESS_TURNS) \
		--repeats $(STRESS_REPEATS) \
		--latency-budget-ms $(STRESS_LATENCY_BUDGET) \
		--cost-budget-usd $(STRESS_COST_BUDGET) \
		--output $(STRESS_OUTPUT) \
		--report $(STRESS_REPORT)
