# Servicio IA — Embedding pipeline

**Solo este servicio con Docker:**

```bash
docker compose up --build ai_service
```

**En local sin Docker (desde la raíz del repo):**

```bash
uv sync
uv run --env-file .env uvicorn app.main:app --port 8001 --app-dir servicio_ia
```

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
      "metadata": {"budget_id": "BUD-2023-004", "client_sector": "finance", "...": "..."}
    }
  ]
}
```

**Por qué `cosine_distance` (`<=>`).** Los embeddings de OpenAI están normalizados, así que coseno e inner product rankean igual. Se usa coseno por convención RAG y porque el futuro índice HNSW usará `vector_cosine_ops`: si el operador de la query y la operator class del índice no coinciden, Postgres ignora el índice y cae a sequential scan **sin avisar**.

**Rendimiento.** Sin índice todavía: Postgres hace sequential scan completo. Para el volumen del corpus de ejemplo (decenas de documentos, cientos de chunks) es perfectamente aceptable — la latencia del endpoint la domina el embedding de la query, no la búsqueda vectorial. Observar esta latencia sin índice es uno de los puntos de partida del directo.

## Vector schema

El schema vive en `alembic/versions/0001_initial_schema.py` (tablas `documents` y `chunks`), con modelos ORM espejo en `app/models.py`. Las decisiones de diseño (dos tablas, metadata JSONB, distancia coseno, ausencia deliberada de índice vectorial) están justificadas en la sección **"Vector schema decisions"** del [README raíz](../README.md).

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

**Con Docker** (con el servicio levantado — `docker compose up -d ai_service`):

```bash
docker compose run --rm ai_service python query_examples.py
```

La URL base se puede cambiar con `SERVICIO_IA_BASE_URL`: por defecto `http://localhost:8001` en el host; el compose la fija a `http://ai_service:8001` para que los contenedores one-off de `docker compose run` lleguen a la API. El output de un run contra el corpus de ejemplo está en [`output_examples.txt`](output_examples.txt).

## Logs

Eventos estructurados (structlog, convención del proyecto): `embedding_ingest_received`, `embedding_ingest_duplicate` (409), `embedding_batch_processed` (chunks, tokens reales del `usage`, latencia), `embedding_rate_limited`, `embedding_ingest_completed`, `embedding_ingest_failed`, `search_received`, `search_completed`, `search_failed`.

```bash
docker compose logs -f ai_service
```
