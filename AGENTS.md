## Sesión 13 — Grafo de estimación (LangGraph)

El agente S12 (un único loop manual sobre la Responses API) se reestructura como un grafo LangGraph de cinco nodos secuenciales, persistencia en Postgres y una traza con Logfire:

```
START → extract_requirements → classify_components → search_budgets
      → generate_estimate → validate_and_consolidate → END
```

- **Estado** (`app/graph/state.py`): `EstimationState` (TypedDict), con `budget_matches` y `errors` como campos acumuladores (`Annotated[list, operator.add]`) — crecen conforme el grafo avanza en vez de ser reemplazados por cada nodo.
- **Nodos** (`app/graph/nodes.py`): funciones puras que reciben el estado y devuelven una actualización parcial. `search_budgets` y `generate_estimate` reutilizan literalmente `app/agents/tools.py` (retrieval S9-S10 + `calculate_estimate` de S12); `extract_requirements`/`classify_components` son nuevos, vía Responses API con salida estructurada (`text.format=json_schema`).
- **Persistencia**: `AsyncPostgresSaver` (`app/graph/checkpointer.py`) apunta al mismo Postgres que las tablas de embeddings — sin infraestructura nueva. Crea sus propias tablas (`checkpoints`, `checkpoint_writes`, `checkpoint_blobs`, `checkpoint_migrations`) que conviven con `documents`/`chunks`. Cada ejecución usa el `estimation_id` como `thread_id`.
- **Arista condicional**: `validate_and_consolidate` fija `status` ("validated" / "needs_review") según si algún componente quedó sin presupuestar o hubo errores; una `add_conditional_edges` enruta según ese campo (ambas ramas terminan en `END` por ahora — el reintento/fallback/HITL se monta en el directo).
- **Observabilidad**: Logfire instrumenta FastAPI, asyncpg y httpx (esto último captura las llamadas a la API de OpenAI, que usa el SDK sobre httpx), más un `logfire.span(...)` por nodo. `logfire.configure(send_to_logfire="if-token-present")`: sin `LOGFIRE_TOKEN` el servicio arranca igual, solo que sin mandar nada a la nube. Para trazas reales, pon un write token de <https://logfire.pydantic.dev> en `LOGFIRE_TOKEN` (`.env`).

**Endpoint** `POST /estimate` (`app/graph/router.py`): contrato estable hacia el backend de negocio — `{transcript, estimation_id}` → `{estimate, status}`.

```bash
curl -X POST http://localhost:8001/estimate \
  -H "Content-Type: application/json" \
  -d '{"transcript": "…", "estimation_id": "est-123"}'
```

**Ejecutar migración** :

```bash
docker compose up --build ai-service 

make db_upgrade && make seed_ia
```

**Ejecutar grafo agéntico**

```bash
uv run --env-file .env python servicio_ia/scripts/run_graph_s13.py \
    servicio_ia/s12_agent_resources/sample_transcript_complex.txt \
    --estimation-id demo-complex-001
```