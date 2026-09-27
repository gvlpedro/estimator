# Evaluación del servicio IA

Golden set + harness (`servicio_ia/eval/`) y dashboard de producción para las
estimaciones. Asume el stack ya levantado (`make docker_run`).

## ⚠️ Esto gasta tokens reales

`run_eval.py` llama al modelo real (`GRAPH_MODEL`) una vez por caso del
golden set. Gasto pequeño y deliberado — no lo lancéis sin querer.

## Cómo lanzar la evaluación

```bash
cp .env.example .env   # OPENAI_API_KEY/ANTHROPIC_API_KEY reales
make docker_run
docker compose exec ai-service python eval/run_eval.py
```

Sin Docker: `uv run python servicio_ia/eval/run_eval.py`.

Lee `servicio_ia/eval/golden_set.json` (5 casos: fácil, medio-alto, borde,
abstención y un segundo caso fácil de otro sector), llama a `POST /estimate`
una vez por caso, y escribe el informe en
`servicio_ia/eval/reports/run_<timestamp>.json`. `example_dry_run.json` es
un ejemplo **sintético** (sin llamar al modelo) que enseña la forma del
informe sin gastar nada.

## Cómo leer el dashboard

El ciclo es dos pasos: **lanza una estimación** (UI en `:3000`, o
directamente contra `servicio_ia /estimate` con el token de servicio) y
luego:

```bash
make dashboard
```

Esto vuelca los logs de `ai-service` y `estimator` (ambos ya loguean JSON,
`APP_ENV=production` en `docker-compose.yml`) y genera
`servicio_ia/eval/reports/dashboard.html` + `.json`.

```
== servicio_ia dashboard ==
requests observed      : 11
latency mean / p95 (ms): 4630.4 / 11400.0
cost mean / total (usd): 0.000822 / 0.009043
error rate             : 18.2%

path                            reqs    err   mean ms    p95 ms     $/req
/estimate                          6    17%    6990.0   11400.0 $  0.0013
/api/v1/sessions/sess-demo-1/e     1     0%    5200.0    5200.0 $  0.0007
```

- **requests / error rate**: todas las peticiones reales, salvo `/health`
  (se excluye a propósito: un healthcheck cada 30s dominaría cualquier
  media).
- **latencia media/p95**: la p95 es la que sufre un usuario real.
- **coste medio/total**: derivado de tokens; cada servicio usa su propio
  cálculo (`servicio_ia/app/observability.py` con tarifas por variable de
  entorno; `app/services/observability.py` reutiliza el coste que
  `llm_wrapper.py` ya calculaba).
- **por *path***: filas de `servicio_ia` (`/estimate`, `/search`, ...) y del
  backend de negocio (`/api/v1/estimate`, `/api/v1/sessions/{id}/estimate`
  — la que usa la UI) conviven en la misma tabla.

Si ves "0 eventos": el servicio no está logueando JSON (fuera de Docker hay
que fijar `APP_ENV=production` a mano).

Ejemplo ya generado con datos sintéticos, sin gastar nada:
`servicio_ia/eval/reports/dashboard_example.html`.
