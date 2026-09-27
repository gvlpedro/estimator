# Despliegue local

Guía mínima para levantar el stack completo (backend de negocio, servicio IA,
UI, Postgres y Redis) con Docker Compose, y verificar que los entregables de
despliegue funcionan.

## Arranque

```bash
cp .env.example .env
# completar OPENAI_API_KEY y/o ANTHROPIC_API_KEY, y AI_SERVICE_TOKEN
make docker_run
```

`make docker_run` es idempotente: libera los puertos publicados, tira abajo
cualquier stack previo, reconstruye las imágenes (`docker compose up --build
--detach`) y espera a que los health checks se pongan en verde antes de
imprimir cómo llegar a cada servicio.

Alternativa sin Makefile:

```bash
cp .env.example .env
docker compose up --build
```

## Servicios

| Servicio         | Contenedor         | Puerto publicado | Dockerfile                  |
| ---------------- | ------------------- | ----------------- | ---------------------------- |
| Backend (app)     | `estimator`         | `8000`             | `business-backend/Dockerfile` |
| Servicio IA       | `servicio-ia`        | ninguno (interno)  | `servicio_ia/Dockerfile`      |
| UI (Streamlit)    | `estimator-ui`      | `3000`             | `Dockerfile` (raíz, legacy)   |
| Postgres+pgvector | `estimator-postgres` | `5432`             | imagen `pgvector/pgvector:pg16` |
| Redis             | `estimator-redis`   | `6379`             | imagen `redis:7-alpine`       |

El servicio IA no publica puerto a propósito (Session 15): solo es alcanzable
desde otros contenedores en la red interna de compose, autenticado con el
header `X-Service-Token`. Para hablarle desde el host hay que entrar por un
contenedor, p.ej. `docker compose exec ai-service curl localhost:8001/docs`.

## Las 5 comprobaciones

1. **Ambos contenedores construyen y arrancan.**

   ```bash
   docker compose ps
   ```

   `estimator` y `servicio-ia` deben aparecer con estado `running` (o
   `healthy` una vez pasado el `start_period` del healthcheck).

2. **Los health checks de Docker están en verde.**

   ```bash
   docker inspect --format '{{.State.Health.Status}}' estimator
   docker inspect --format '{{.State.Health.Status}}' servicio-ia
   ```

   Ambos deben devolver `healthy`. `make docker_run` ya espera esto
   automáticamente antes de imprimir el resumen final.

3. **`.env.example` cubre todas las variables necesarias.**

   ```bash
   diff <(grep -oE '^[A-Z_]+=' .env.example | sort) \
        <(grep -oE '^[A-Z_]+=' .env | sort)
   ```

   Sin salida = las mismas claves están presentes en ambos archivos (sin
   valores reales en `.env.example`, que sí está versionado; `.env` está en
   `.gitignore`).

4. **`/health` del servicio IA responde sin autenticación.**

   ```bash
   docker compose exec ai-service \
     python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8001/health').read())"
   ```

   Debe devolver `{"status": "healthy", "service": "servicio_ia", "version": "0.1.0"}`.
   Para el backend de negocio, que sí publica puerto:

   ```bash
   curl -s http://localhost:8000/health
   ```

5. **La verificación de token de servicio protege los endpoints reales.**

   Sin token (o con uno incorrecto) → `401`:

   ```bash
   docker compose exec ai-service \
     python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/search')"
   # -> HTTPError: 422 (falta el body) o, con header incorrecto, 401 Invalid service token
   ```

   Con el token correcto (el mismo valor de `AI_SERVICE_TOKEN` en `.env`) →
   pasa la autenticación y llega a la lógica de negocio del endpoint:

   ```bash
   docker compose exec ai-service sh -c \
     'python -c "
   import json, urllib.request
   req = urllib.request.Request(
       \"http://localhost:8001/search\",
       data=json.dumps({\"query\": \"test\", \"k\": 1}).encode(),
       headers={\"Content-Type\": \"application/json\", \"X-Service-Token\": \"$AI_SERVICE_TOKEN\"},
   )
   print(urllib.request.urlopen(req).status)
   "'
   ```

   `/health` es la única ruta exenta de este chequeo a propósito (así el
   healthcheck de Docker no necesita conocer el secreto). Ver
   `servicio_ia/app/security.py`.
