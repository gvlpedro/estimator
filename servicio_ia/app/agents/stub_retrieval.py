"""Offline safety net for ``search_budgets``: a tiny canned corpus, no database.

Only used when the CLI script (``scripts/run_agent_s12.py``) is run with
``--stub`` -- for debugging the AGENT LOOP mechanics cheaply, without a
Postgres+pgvector stack or an ingested corpus. Returns the SAME shape
``app.agents.tools.search_budgets`` does, so ``calculate_estimate`` and the
trace formatting don't need to know which one ran. Keyword-matched, not
embedding-ranked -- good enough to exercise the loop, not a retrieval
quality benchmark.
"""

from __future__ import annotations

from typing import Any

_CORPUS: list[dict[str, Any]] = [
    {
        "keywords": ("oauth", "auth", "autenticación", "jwt", "login", "token", "sso"),
        "budget_id": "BUD-AUTH-2023-07",
        "component_id": "AUTH-01",
        "content_preview": "Backend de autenticación OAuth2 con JWT, refresh tokens y multi-tenant.",
        "sector": "finance",
        "technology": "Python",
        "complexity": "high",
        "year": 2023,
        "estimated_hours": 420.0,
    },
    {
        "keywords": ("oauth", "auth", "rate limiting", "seguridad", "psd2"),
        "budget_id": "BUD-AUTH-2022-11",
        "component_id": "AUTH-02",
        "content_preview": "Módulo de seguridad: rate limiting por cliente, SCA y gestión de consentimientos.",
        "sector": "finance",
        "technology": "Python",
        "complexity": "medium",
        "year": 2022,
        "estimated_hours": 380.0,
    },
    {
        "keywords": ("backend", "api", "pedidos", "orders", "rutas", "tarifas", "núcleo", "core"),
        "budget_id": "BUD-CORE-2023-02",
        "component_id": "CORE-01",
        "content_preview": (
            "Backend de negocio: gestión de pedidos, asignación de rutas y cálculo "
            "de tarifas con API REST."
        ),
        "sector": "logistics",
        "technology": "Python",
        "complexity": "high",
        "year": 2023,
        "estimated_hours": 1150.0,
    },
    {
        "keywords": ("backend", "api", "seguimiento", "tracking", "envíos"),
        "budget_id": "BUD-CORE-2022-09",
        "component_id": "CORE-02",
        "content_preview": "Servicio de seguimiento de envíos en tiempo real con API para terceros.",
        "sector": "logistics",
        "technology": "Python",
        "complexity": "medium",
        "year": 2022,
        "estimated_hours": 940.0,
    },
    {
        "keywords": ("erp", "sap", "integración", "integration", "idoc", "middleware", "facturación"),
        "budget_id": "BUD-SAP-2023-05",
        "component_id": "ERP-01",
        "content_preview": (
            "Integración con SAP: sincronización de facturación y maestro de "
            "artículos vía IDocs con reintentos."
        ),
        "sector": "industrial",
        "technology": "Java",
        "complexity": "high",
        "year": 2023,
        "estimated_hours": 860.0,
    },
    {
        "keywords": ("erp", "sap", "integración", "middleware", "clientes"),
        "budget_id": "BUD-SAP-2021-12",
        "component_id": "ERP-02",
        "content_preview": "Middleware de integración ERP con capa de reintentos y monitorización de colas.",
        "sector": "industrial",
        "technology": "Java",
        "complexity": "medium",
        "year": 2021,
        "estimated_hours": 720.0,
    },
    {
        "keywords": ("móvil", "mobile", "app", "android", "ios", "offline", "repartidor", "firma"),
        "budget_id": "BUD-APP-2023-03",
        "component_id": "APP-01",
        "content_preview": (
            "App móvil de reparto (Android/iOS) con hoja de ruta, firma, foto y "
            "modo offline con sincronización."
        ),
        "sector": "logistics",
        "technology": "React Native",
        "complexity": "high",
        "year": 2023,
        "estimated_hours": 780.0,
    },
    {
        "keywords": ("móvil", "mobile", "app", "híbrida", "react native"),
        "budget_id": "BUD-APP-2022-06",
        "component_id": "APP-02",
        "content_preview": "App híbrida React Native con sincronización offline y notificaciones push.",
        "sector": "ecommerce",
        "technology": "React Native",
        "complexity": "medium",
        "year": 2022,
        "estimated_hours": 640.0,
    },
    {
        "keywords": ("analítica", "analytics", "panel", "dashboard", "kpi", "cuadros", "alertas", "informes"),
        "budget_id": "BUD-BI-2023-01",
        "component_id": "BI-01",
        "content_preview": (
            "Panel de analítica con cuadros de mando de KPIs, filtros por "
            "zona/cliente y alertas por umbral."
        ),
        "sector": "logistics",
        "technology": "React",
        "complexity": "medium",
        "year": 2023,
        "estimated_hours": 560.0,
    },
    {
        "keywords": ("analítica", "analytics", "bi", "tendencias", "reporting"),
        "budget_id": "BUD-BI-2022-04",
        "component_id": "BI-02",
        "content_preview": "Reporting BI con tendencias históricas y export programado.",
        "sector": "media",
        "technology": "React",
        "complexity": "low",
        "year": 2022,
        "estimated_hours": 430.0,
    },
]


def stub_search_budgets(args: dict[str, Any]) -> dict[str, Any]:
    """Return canned historical items whose keywords appear in the query.

    A crude relevance score is derived from how many keywords matched (more
    matches -> higher score), so the shape mirrors the real reranked search.
    """
    query = args["query"]
    filters = args.get("filters") or {}
    sector = (filters.get("sector") or "").strip().lower()
    technology = (filters.get("technology") or "").strip().lower()

    q = query.lower()
    hits: list[dict[str, Any]] = []
    for entry in _CORPUS:
        matches = sum(1 for kw in entry["keywords"] if kw in q)
        if matches == 0:
            continue
        if sector and entry["sector"].lower() != sector:
            continue
        if technology and technology not in entry["technology"].lower():
            continue
        hits.append({**entry, "relevance_score": round(0.3 + 0.1 * matches, 4)})

    hits.sort(key=lambda h: h["relevance_score"], reverse=True)

    items = [
        {
            "budget_id": h["budget_id"],
            "component_id": h["component_id"],
            "estimated_hours": h["estimated_hours"],
            "sector": h["sector"],
            "technology": h["technology"],
            "complexity": h["complexity"],
            "year": h["year"],
            "relevance_score": h["relevance_score"],
            "preview": h["content_preview"],
        }
        for h in hits[:5]
    ]

    budgets = ", ".join(str(item["budget_id"]) for item in items)
    summary = f"{len(items)} historical item(s) found" + (f" (budgets: {budgets})" if items else "")
    return {"query": query, "filters": filters, "results": items, "summary": summary}
