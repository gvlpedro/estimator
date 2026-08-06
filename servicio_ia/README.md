# Servicio IA — Embedding pipeline

**Solo este servicio con Docker:**

```bash
docker compose up --build ai-service
```

**En local sin Docker (desde la raíz del repo):**

```bash
uv sync
uv run --env-file .env uvicorn app.main:app --port 8001 --app-dir servicio_ia
```

## Módulos

```
servicio_ia/app/
├── ingest/              # documentos dentro: contratos Budget, chunking estructural
│   ├── schemas.py       #   y POST /embeddings/ingest (chunk → embed → persist)
│   ├── chunker.py
│   └── router.py
├── embedding_pipeline/  # texto → vectores: embedder OpenAI, contratos de chunks,
│   ├── schemas.py       #   POST /embeddings/encode y POST /search
│   ├── embedder.py
│   └── router.py
├── storage/             # persistencia: engine/session async, modelos ORM y
│   ├── db.py            #   repositorio (todo el SQL vive aquí, incluido el
│   ├── models.py        #   ranking vectorial por distancia coseno)
│   └── repository.py
├── generation/rag/retrieval/  # reranking de un shortlist: wrapper de cross-encoder
│   ├── cross_encoder.py       #   (sentence-transformers, wired into POST /search via `rerank`)
│   ├── rrf.py                 #   Reciprocal Rank Fusion — pure function, no I/O (English docstrings)
│   ├── hybrid_search.py       #   orchestrates vector + lexical retrieval and fuses via RRF (English docstrings)
│   ├── retrieve.py            #   shared per-mode dispatch (vector/lexical/hybrid), reused at any k (English docstrings)
│   └── reranked_search.py     #   recall-then-rerank: wide retrieve() pass + cross-encoder re-score (English docstrings)
├── dependencies.py      # providers compartidos (embedder, chunker)
└── main.py
```

Los routers no construyen SQL: la detección de duplicados, la persistencia transaccional y el ranking vectorial están en `app/storage/repository.py`.

## Base de datos: migración y seed

Con el postgres del compose levantado (`docker compose up -d postgres`):

```bash
make db_upgrade   # alembic upgrade head — crea extensión vector + tablas documents/chunks
make seed_ia      # ingesta los 15 presupuestos históricos de data/budgets_sample.json
```

El seed es **idempotente**: usa `source_path` como clave (`data/budgets_sample.json::<budget_id>`), así que un 409 cuenta como "ya presente" y re-ejecutarlo tras un fallo parcial solo ingesta lo que falte. El embedding ocurre en el servidor, de modo que el script (`scripts/seed_budgets.py`) solo necesita stdlib — pero sí el servicio levantado en `:8001`.

## Endpoints

Swagger UI interactivo: <http://localhost:8001/docs> — desde ahí se puede invocar todo con "Try it out".

### GET /health

```bash
curl http://localhost:8001/health
# {"status":"healthy","service":"servicio_ia","version":"0.1.0"}
```

### POST /embeddings/ingest

Recibe un documento (presupuesto normalizado), lo trocea con chunking estructural, embebe los chunks en un batch y **persiste documento + chunks en Postgres dentro de una única transacción**. Devuelve identificadores y métricas — nunca los vectores.

```bash
# Construir el payload con el primer presupuesto del dataset de ejemplo
python3 -c "
import json
budgets = json.load(open('servicio_ia/data/budgets_sample.json'))
json.dump({
    'source_path': 'data/budgets_sample.json::' + budgets[0]['budget_id'],
    'document_type': 'historical_budget',
    'content': budgets[0],
}, open('/tmp/ingest.json', 'w'))
"

curl -X POST http://localhost:8001/embeddings/ingest \
  -H "Content-Type: application/json" \
  -d @/tmp/ingest.json
```

Respuesta `200 OK`:

```json
{
  "document_id": 1,
  "chunks_created": 4,
  "embedding_dimension": 1536,
  "ingestion_time_ms": 4398
}
```

Respuesta `409 Conflict` si ya existe un documento con ese `source_path` (la ingesta es idempotente por ruta de origen):

```json
{
  "detail": "Document already ingested",
  "document_id": 1
}
```

La transacción única garantiza que un fallo del embedder no deja filas huérfanas en `documents`: el documento se inserta con `flush` (visible solo dentro de la transacción) y solo el `commit` final, con todos los chunks ya añadidos, lo hace permanente.

### POST /embeddings/encode

Embebe textos crudos con `text-embedding-3-small` y devuelve los vectores, uno por texto y en el mismo orden. A diferencia de `/embeddings/ingest`, **no persiste nada**: expone el encoder en sí, para que otros servicios (o los ejercicios del directo) obtengan vectores sin pasar por el pipeline documental. Máximo 100 textos por petición (una petición = una llamada a la API de embeddings).

```bash
curl -X POST http://localhost:8001/embeddings/encode \
  -H "Content-Type: application/json" \
  -d '{"texts": ["REST API with JWT authentication", "payment gateway integration"]}'
```

Respuesta `200 OK` (vectores truncados):

```json
{
  "model": "text-embedding-3-small",
  "embedding_dimension": 1536,
  "embeddings": [[-0.0444, -0.0262, "…"], [0.0113, -0.0186, "…"]],
  "encode_time_ms": 512
}
```

### POST /search

Embebe la query con el mismo modelo usado en ingesta (`text-embedding-3-small`) y devuelve los `k` chunks más cercanos por distancia coseno (`<=>` de pgvector, menor = más similar).

```bash
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "REST API with OAuth authentication for fintech sector", "k": 5}'
```

Respuesta `200 OK` (resumida; `k` es opcional, por defecto 5):

```json
{
  "query": "REST API with OAuth authentication for fintech sector",
  "k": 5,
  "search_time_ms": 291,
  "results": [
    {
      "chunk_id": 17,
      "document_id": 5,
      "chunk_type": "budget_component",
      "content": "[Project: ...]\n\nComponent: User authentication...",
      "distance": 0.4561,
      "score": null,
      "metadata": {"budget_id": "BUD-2023-004", "client_sector": "finance", "...": "..."}
    }
  ]
}
```

### POST /search — `mode` field

`SearchRequest` accepts an optional `mode`, defaulting to `"vector"` so every existing caller (the business backend's search proxy never sets it) keeps getting today's response shape unchanged:

- **`"vector"`** (default): embeds the query and ranks by cosine distance, as above. `distance` is populated, `score` is `null`.
- **`"lexical"`**: ranks by Postgres full-text search (`ts_rank` over the generated `content_tsv` column, matched with `websearch_to_tsquery` so free-text queries work without a special syntax). No embedding call is made. `distance` is `null`, `score` holds the raw `ts_rank` weight (higher = more relevant).
- **`"hybrid"`**: runs the vector and lexical branches independently (each gets its own top-k) and fuses them with Reciprocal Rank Fusion (`app/generation/rag/retrieval/rrf.py`) — rank position is fused, not raw scores, since cosine distance and `ts_rank` live on incomparable scales. `distance` is `null`, `score` holds the fused RRF score (higher = more relevant).

```bash
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "MQTT sensor telemetry ingestion", "k": 5, "mode": "hybrid"}'
```

Note the inverted directionality between the two score fields: for `distance`, lower is better (it's a distance); for `score`, higher is better (it's a relevance weight). Only one of the two is ever non-null, depending on `mode`.

### POST /search — `rerank` field (recall-then-rerank)

`SearchRequest` also accepts an optional `rerank`, defaulting to `false` so every existing caller keeps getting today's response shape unchanged. When `true`, the endpoint layers a second stage on top of whichever `mode` is selected:

1. **Recall**: run the exact same per-`mode` ranking (`app/generation/rag/retrieval/retrieve.py`) but widened to `RECALL_K = 50` candidates instead of the caller's `k` — this stage optimizes for coverage, not precision, so a relevant chunk ranked outside the plain top-k still gets a chance.
2. **Rerank**: score every candidate's raw text against the query with a cross-encoder (`app/generation/rag/retrieval/cross_encoder.py`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) — a model that reads the (query, document) pair jointly, more accurate than comparing independent embeddings but too expensive to run over the whole corpus, which is why it only ever sees the 50-candidate shortlist. The top `k` by cross-encoder score is returned.

When `rerank` is `true`, `distance` is always `null` and `score` holds the cross-encoder's relevance score (higher = more relevant) — it replaces the branch-native score (cosine distance / `ts_rank` / RRF) shown when `rerank` is `false`, since that native score no longer determined the final order.

```bash
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "REST API with OAuth authentication for fintech sector", "k": 5, "mode": "vector", "rerank": true}'
```

**Por qué `cosine_distance` (`<=>`).** Los embeddings de OpenAI están normalizados, así que coseno e inner product rankean igual. Se usa coseno por convención RAG y porque el futuro índice HNSW usará `vector_cosine_ops`: si el operador de la query y la operator class del índice no coinciden, Postgres ignora el índice y cae a sequential scan **sin avisar**.

**Rendimiento.** Sin índice todavía: Postgres hace sequential scan completo. Para el volumen del corpus de ejemplo (decenas de documentos, cientos de chunks) es perfectamente aceptable — la latencia del endpoint la domina el embedding de la query, no la búsqueda vectorial. Observar esta latencia sin índice es uno de los puntos de partida del directo.

## Vector schema

El schema vive en `alembic/versions/0001_initial_schema.py` (tablas `documents` y `chunks`), con modelos ORM espejo en `app/storage/models.py`. Las decisiones de diseño (dos tablas, metadata JSONB, distancia coseno, ausencia deliberada de índice vectorial) están justificadas en la sección **"Vector schema decisions"** del [README raíz](../README.md).

Dos decisiones de implementación adicionales:

**`vector(1536)`.** Dimensionalidad de `text-embedding-3-small`. Está hardcodeada porque cambiarla implica re-embedear todo el corpus, así que no es una decisión que vaya a cambiar dinámicamente.

**`embedding` nullable.** Permite insertar un chunk en una transacción y rellenar el embedding después si el cálculo fallase. En este ejercicio no lo usamos así (ingestamos chunk+embedding atómicamente), pero deja la puerta abierta a la ingesta asíncrona que veremos en sesiones posteriores.

## Queries de ejemplo contra /search (query_examples.py)

Invoca `POST /search` con cinco queries que ejercitan el corpus desde ángulos distintos: componente directo conocido (sanity check), reformulación semántica (significado vs palabras), dominio ausente del corpus (las distancias suben), consulta ambigua (ranking sin match dominante) y consulta muy específica (discriminación entre tecnologías). Imprime el top-5 de cada una con distancia, chunk y preview del contenido.

El embedding de la query lo hace el servidor, así que el script no necesita API key ni dependencias (solo stdlib) — pero sí el servicio levantado y el corpus ingestado.

**Fuera del contenedor** (desde la raíz del repo):

```bash
uv run python servicio_ia/query_examples.py
```

**Con Docker** (con el servicio levantado — `docker compose up -d ai-service`):

```bash
docker compose run --rm ai-service python query_examples.py
```

La URL base se puede cambiar con `SERVICIO_IA_BASE_URL`: por defecto `http://localhost:8001` en el host; el compose la fija a `http://ai-service:8001` para que los contenedores one-off de `docker compose run` lleguen a la API. El output de un run contra el corpus de ejemplo está en [`output_examples.txt`](output_examples.txt).

## Golden-set evaluation (golden_eval.py)

Lee las 7 consultas de `evals/golden_retrieval.json` (5 adoptadas del golden set oficial del repositorio de referencia del curso + 2 propias dirigidas al punto débil de cada método — ver "Golden set y comparativa de configuraciones" en el [README raíz](../README.md#retrieval)) y las ejecuta contra las cuatro configuraciones (`vector`/`hybrid` x `rerank` on/off), **dos veces** — con `dedupe: false` y `dedupe: true` — para aislar el efecto de deduplicar por presupuesto antes de contar aciertos. Imprime precisión@5 (por presupuesto distinto, no por chunk) y latencia por configuración y pasada, más el desglose por consulta. Mismos requisitos que `query_examples.py` (servicio levantado, corpus ingestado, sin dependencias):

```bash
uv run python servicio_ia/golden_eval.py                # fuera del contenedor
docker compose run --rm ai-service python golden_eval.py  # con Docker
```

## Logs

Eventos estructurados (structlog, convención del proyecto): `embedding_ingest_received`, `embedding_ingest_duplicate` (409), `embedding_batch_processed` (chunks, tokens reales del `usage`, latencia), `embedding_rate_limited`, `embedding_ingest_completed`, `embedding_ingest_failed`, `encode_received`, `encode_completed`, `encode_failed`, `search_received`, `search_completed`, `search_failed`.

```bash
docker compose logs -f ai-service
```
