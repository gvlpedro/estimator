# Sanity check de embeddings — Sesión 07

Resultados de `scripts/compare.py` sobre `data/budgets_sample.json`, con
`text-embedding-3-small` (dimensión por defecto, 1536) y similitud coseno
calculada a mano con la biblioteca estándar (sin numpy).

Fecha del run: 2026-07-07.

## Las tres parejas

| # | Chunk A | Chunk B | Relación esperada | Similitud coseno |
|---|---------|---------|-------------------|------------------|
| 1 | `BUD-2024-014::AUTH-001` (OAuth 2.0 backend, fintech, Rails) | `BUD-2023-004::AUTH-001` (OAuth 2.0 backend, fintech, Rails) | Mismo concepto, mismo sector y stack | **0.8795** |
| 2 | `BUD-2022-003::PAY-001` (pagos Stripe, grocery, Node) | `BUD-2024-009::CHK-001` (checkout Redsys/Bizum, moda, Laravel) | Concepto relacionado, distinto vertical y stack | **0.7019** |
| 3 | `BUD-2024-014::AUTH-001` (OAuth backend, fintech) | `BUD-2025-013::PDM-001` (mantenimiento predictivo ML, parque eólico) | Sin relación | **0.4454** |

Ordenamiento esperado (pareja 1 > pareja 2 > pareja 3): **PASS**.

## Comentario

- **El espacio de embeddings preserva la semántica que nos importa.** Dos
  backends OAuth de clientes fintech distintos quedan a 0.88 — el modelo los
  reconoce como "el mismo tipo de trabajo", que es exactamente la señal que
  un estimador necesita para recuperar componentes históricos comparables.
- **La pareja intermedia (0.70) es la más informativa.** Ambos chunks hablan
  de cobrar al cliente, pero con pasarelas, sectores y stacks distintos. Que
  caiga claramente por debajo de los near-duplicates y claramente por encima
  del par no relacionado indica que el embedding pondera el QUÉ (dominio
  funcional) por encima del CON QUÉ (tecnología concreta).
- **El suelo no es 0.** El par "sin relación" da 0.45, no 0.0: ambos chunks
  comparten plantilla (contextual chunk header), registro técnico y formato.
  Con `text-embedding-3-small` las similitudes "no relacionadas" suelen
  flotar en 0.2–0.5; los umbrales absolutos tipo "0.5 = relacionado" son una
  trampa — lo que discrimina es el ORDEN relativo y la separación entre
  bandas (aquí ~0.18 y ~0.26), no el valor absoluto.
- **Implicación para el retrieval futuro:** un top-k por similitud funcionará,
  pero un umbral de corte absoluto habrá que calibrarlo empíricamente sobre
  este corpus, no copiarlo de un tutorial.
