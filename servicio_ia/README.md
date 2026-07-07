# Servicio IA — Embedding pipeline

Servicio FastAPI independiente que trocea presupuestos históricos normalizados y los vectoriza con `text-embedding-3-small` (OpenAI, dimensión por defecto 1536). Es el primer paso hacia la arquitectura RAG del estimador.

```
servicio_ia/
├── app/
│   ├── main.py                  # entrypoint FastAPI (puerto 8001)
│   └── embedding_pipeline/
│       ├── schemas.py           # contratos Pydantic (Budget, Chunk, Ingest*)
│       ├── chunker.py           # JSONStructuralChunker: 1 componente = 1 chunk
│       ├── embedder.py          # OpenAIEmbedder: batches, retries, coste
│       ├── router.py            # POST /ingest
│       └── SANITY_CHECK.md      # resultados del sanity check de embeddings
├── scripts/
│   └── compare.py               # similitud coseno a mano (stdlib, sin numpy)
└── data/
    └── budgets_sample.json      # 15 presupuestos / 63 componentes
```

Requisito: `OPENAI_API_KEY` en el `.env` de la raíz del repo.

## Arrancar el servicio

**Stack completa con Docker (backend + servicio_ia + ui + redis):**

```bash
make docker_run        # idempotente: para lo anterior y rearranca
```

**Solo este servicio con Docker:**

```bash
docker compose up --build servicio_ia
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

Recibe `{"budgets": [...]}` (lista de presupuestos normalizados), los trocea con chunking estructural, embebe cada chunk en batches y devuelve los vectores más las estadísticas del run.

```bash
# Construir el payload con el dataset de ejemplo
python3 -c "import json; json.dump({'budgets': json.load(open('servicio_ia/data/budgets_sample.json'))}, open('/tmp/ingest.json','w'))"

curl -X POST http://localhost:8001/embeddings/ingest \
  -H "Content-Type: application/json" \
  -d @/tmp/ingest.json
```

Respuesta (resumida):

```json
{
  "chunks": [
    {
      "chunk_id": "BUD-2022-001::EHR-001",
      "text": "[Project: ...]\n[Client sector: healthcare | ...]\n\nComponent: ...",
      "metadata": {"budget_id": "...", "client_sector": "...", "estimated_hours": 180},
      "token_count": 105,
      "embedding": [0.0123, ...]
    }
  ],
  "stats": {
    "total_budgets": 15,
    "total_chunks": 63,
    "total_tokens": 6016,
    "estimated_cost_usd": 0.00012032
  }
}
```

Códigos de estado:

- **200** — ingesta completada.
- **422** — el payload viola los invariantes del dataset (totales que no cuadran con la suma de componentes, dependencias a componentes inexistentes, sector fuera del vocabulario cerrado). Se rechaza ANTES de tocar la API de embeddings.
- **500** — error no controlado del proveedor de embeddings. El cliente recibe un mensaje genérico; el detalle queda en los logs (`embedding_ingest_failed`).

## Sanity check de embeddings (compare.py)

Embebe tres parejas de chunks y verifica que el orden de similitud coseno respete la semántica esperada (near-duplicates > relacionados > sin relación). Sale con código 1 si el orden no se cumple.

**Fuera del contenedor** (desde la raíz del repo):

```bash
uv run --env-file .env python servicio_ia/scripts/compare.py
```

**Dentro del contenedor** (con el servicio levantado):

```bash
docker compose exec servicio_ia python scripts/compare.py
```

Resultados del último run y comentario: [`app/embedding_pipeline/SANITY_CHECK.md`](app/embedding_pipeline/SANITY_CHECK.md).

## Logs

Eventos estructurados (structlog, convención del proyecto): `embedding_ingest_received`, `embedding_batch_processed` (chunks, tokens reales del `usage`, latencia), `embedding_rate_limited`, `embedding_ingest_completed`, `embedding_ingest_failed`.

```bash
docker compose logs -f servicio_ia
```
