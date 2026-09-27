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
UI_PORT          ?= 3000
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

.PHONY: setup run stop stress docker_run db_upgrade seed_ia dashboard

DASHBOARD_LOG  ?= /tmp/estimator-ai-service.log
DASHBOARD_HTML ?= servicio_ia/eval/reports/dashboard.html
DASHBOARD_JSON ?= servicio_ia/eval/reports/dashboard.json

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


docker_run:
	@docker compose down --remove-orphans
	@$(MAKE) stop
	@docker compose up --build --detach
	@echo "Waiting for services to become healthy..."
	@for i in $$(seq 1 60); do \
		backend=$$(curl -sf -o /dev/null http://localhost:$(BACKEND_PORT)/health && echo ok || echo ko); \
		ia=$$(docker inspect --format '{{.State.Health.Status}}' servicio-ia 2>/dev/null | grep -q '^healthy$$' && echo ok || echo ko); \
		ui=$$(curl -sf -o /dev/null http://localhost:$(UI_PORT)/_stcore/health && echo ok || echo ko); \
		if [ "$$backend$$ia$$ui" = "okokok" ]; then break; fi; \
		sleep 2; \
	done; \
	echo ""; \
	echo "=========================================================="; \
	echo "  Servicios levantados:"; \
	echo "  Backend (app)  [$$backend]  http://localhost:$(BACKEND_PORT)  —  Swagger: http://localhost:$(BACKEND_PORT)/docs"; \
	echo "  Servicio IA    [$$ia]  interno only, sin puerto publicado  —  docker compose exec ai-service curl localhost:8001/docs"; \
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


seed_ia:
	@export PATH="$$HOME/.local/bin:$$PATH"; \
	if [ "$$(docker compose ps -q ai-service 2>/dev/null)" != "" ]; then \
		docker compose run --rm ai-service python scripts/seed_budgets.py; \
	else \
		uv run python servicio_ia/scripts/seed_budgets.py; \
	fi


# Dumps BOTH ai-service's and estimator's container logs and renders the
# self-contained metrics dashboard (servicio_ia/eval/dashboard.py) from them
# as HTML + JSON -- covers direct servicio_ia /estimate calls AND UI-driven
# estimations via the business backend (two different code paths; see
# doc/evaluation.md). Requires the stack running (`make docker_run`); both
# services already log JSON by default (APP_ENV=production in
# docker-compose.yml), so this only warns instead of silently showing an
# empty dashboard if that's somehow not the case.
dashboard:
	@docker compose logs ai-service estimator --no-color > $(DASHBOARD_LOG) 2>/dev/null; \
	if [ ! -s $(DASHBOARD_LOG) ]; then \
		echo "No hay logs. ¿Está el stack arriba? Prueba 'make docker_run'."; \
		exit 1; \
	fi; \
	export PATH="$$HOME/.local/bin:$$PATH"; \
	uv run python servicio_ia/eval/dashboard.py $(DASHBOARD_LOG) \
		--html $(DASHBOARD_HTML) --json $(DASHBOARD_JSON); \
	if ! grep -q request_completed $(DASHBOARD_LOG); then \
		echo ""; \
		echo "AVISO: 0 eventos request_completed en $(DASHBOARD_LOG)."; \
		echo "Algún servicio no está logueando en JSON (APP_ENV != production)."; \
		echo "Para forzarlo puntualmente en ai-service:"; \
		echo "  docker compose run --rm -e APP_ENV=production ai-service uvicorn app.main:app --host 0.0.0.0 --port 8001"; \
		echo "  (lanza alguna estimación contra ese proceso y repite 'make dashboard')"; \
	fi

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
