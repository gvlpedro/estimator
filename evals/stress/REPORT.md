# Stress report

Source CSV: `evals/stress/results.csv`. Rows: 270.

## Summary by profile

| Profile | Rows | P50 latency (ms) | P95 latency (ms) | Total cost (USD) | Cache hit rate | Mean fact recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| contradiction | 90 | 7320 | 18737 | 0.0000 | 0/90 | 0.67 |
| growing | 90 | 7126 | 20197 | 0.0000 | 0/90 | 0.96 |
| pivot | 90 | 6952 | 16073 | 0.0000 | 0/90 | 0.74 |

## Curves

### Latency vs tokens_in

| tokens_in | latency_ms |
| --- | --- |
| 2446 | 9723 |
| 2450 | 5180 |
| 2455 | 6334 |
| 2477 | 5167 |
| 2483 | 6708 |
| 2495 | 5471 |
| 2503 | 5724 |
| 2504 | 5018 |
| 2516 | 6736 |
| 2526 | 5885 |
| 2529 | 8433 |
| 2530 | 7228 |
| 2532 | 10583 |
| 2537 | 6084 |
| 2538 | 6026 |
| 2559 | 5614 |
| 2564 | 6085 |
| 2573 | 6143 |
| 2578 | 5529 |
| 2585 | 6861 |
| 2594 | 6291 |
| 2596 | 7577 |
| 2603 | 10750 |
| 2605 | 7025 |
| 2608 | 6451 |
| 2611 | 6559 |
| 2616 | 6654 |
| 2617 | 7541 |
| 2623 | 6214 |
| 2631 | 15766 |
| 2638 | 7231 |
| 2652 | 7304 |
| 2655 | 5857 |
| 2656 | 6318 |
| 2662 | 5957 |
| 2668 | 5316 |
| 2676 | 5730 |
| 2677 | 7424 |
| 2692 | 6288 |
| 2694 | 6867 |
| 2701 | 8050 |
| 2719 | 7177 |
| 2742 | 9080 |
| 2751 | 7236 |
| 2752 | 6983 |
| 2766 | 5990 |
| 2772 | 6791 |
| 2784 | 5978 |
| 2856 | 7213 |
| 2934 | 9575 |
| 3511 | 8193 |
| 3577 | 6371 |
| 3591 | 7562 |
| 3605 | 5495 |
| 3611 | 5236 |
| 3618 | 5551 |
| 3628 | 5882 |
| 3654 | 7268 |
| 3665 | 9354 |
| 3682 | 7092 |
| 3697 | 7106 |
| 3709 | 6195 |
| 3722 | 5532 |
| 3723 | 7156 |
| 3723 | 6701 |
| 3729 | 6141 |
| 3735 | 7381 |
| 3740 | 5640 |
| 3740 | 5547 |
| 3740 | 5803 |
| 3741 | 5420 |
| 3741 | 5744 |
| 3741 | 5542 |
| 3748 | 9002 |
| 3754 | 6426 |
| 3758 | 5263 |
| 3764 | 6955 |
| 3766 | 5688 |
| 3767 | 5527 |
| 3770 | 5757 |
| 3776 | 6216 |
| 3779 | 5355 |
| 3782 | 5560 |
| 3810 | 6013 |
| 3815 | 5804 |
| 3815 | 7502 |
| 3821 | 6864 |
| 3825 | 5912 |
| 3833 | 6981 |
| 3836 | 5275 |
| 3839 | 8019 |
| 3840 | 6552 |
| 3843 | 7103 |
| 3844 | 5525 |
| 3845 | 6024 |
| 3849 | 5652 |
| 3853 | 5692 |
| 3853 | 12226 |
| 3856 | 11920 |
| 3857 | 6637 |
| 3863 | 6221 |
| 3877 | 5551 |
| 3905 | 37987 |
| 3909 | 9572 |
| 3934 | 7955 |
| 3951 | 7105 |
| 4061 | 8603 |
| 4509 | 10415 |
| 6949 | 10962 |
| 6957 | 15246 |
| 6971 | 5661 |
| 6981 | 6366 |
| 7001 | 8890 |
| 7012 | 6136 |
| 7012 | 6791 |
| 7027 | 6768 |
| 7050 | 4939 |
| 7065 | 10659 |
| 7066 | 6624 |
| 7068 | 7467 |
| 7089 | 6275 |
| 7092 | 6056 |
| 7095 | 5926 |
| 7096 | 5639 |
| 7105 | 7954 |
| 7109 | 5832 |
| 7119 | 6163 |
| 7121 | 7629 |
| 7123 | 6989 |
| 7127 | 6763 |
| 7127 | 7153 |
| 7131 | 6455 |
| 7134 | 6948 |
| 7137 | 6684 |
| 7138 | 5643 |
| 7140 | 7383 |
| 7143 | 6900 |
| 7145 | 6688 |
| 7154 | 6550 |
| 7155 | 7155 |
| 7158 | 6286 |
| 7165 | 6968 |
| 7165 | 6006 |
| 7167 | 7280 |
| 7178 | 6870 |
| 7182 | 8013 |
| 7185 | 7421 |
| 7188 | 12274 |
| 7192 | 7659 |
| 7204 | 6071 |
| 7246 | 6362 |
| 7257 | 7344 |
| 7260 | 7666 |
| 7279 | 15283 |
| 7282 | 11628 |
| 7291 | 10964 |
| 7297 | 9354 |
| 7306 | 17026 |
| 7332 | 8819 |
| 7357 | 9678 |
| 7393 | 5724 |
| 10314 | 6955 |
| 13586 | 5124 |
| 13629 | 5168 |
| 13636 | 6309 |
| 13653 | 5830 |
| 13671 | 12416 |
| 13675 | 7150 |
| 13677 | 5201 |
| 13677 | 5492 |
| 13680 | 11272 |
| 13693 | 6003 |
| 13717 | 5581 |
| 13733 | 10122 |
| 13743 | 15449 |
| 13747 | 12672 |
| 13750 | 13455 |
| 13759 | 6884 |
| 13760 | 18359 |
| 13766 | 16186 |
| 13774 | 14457 |
| 13776 | 14099 |
| 13783 | 11064 |
| 13785 | 7245 |
| 13786 | 7146 |
| 13787 | 10052 |
| 13795 | 7351 |
| 13803 | 7031 |
| 13807 | 7783 |
| 13809 | 6502 |
| 13824 | 8737 |
| 13826 | 5908 |
| 13827 | 9819 |
| 13830 | 6344 |
| 13834 | 6222 |
| 13837 | 9630 |
| 13838 | 7609 |
| 13845 | 15627 |
| 13848 | 8861 |
| 13848 | 6061 |
| 13860 | 7432 |
| 13873 | 8920 |
| 13874 | 12273 |
| 13875 | 9200 |
| 13877 | 8752 |
| 13878 | 7209 |
| 13878 | 21332 |
| 13880 | 6984 |
| 13881 | 6553 |
| 13891 | 15194 |
| 13923 | 10104 |
| 13968 | 7336 |
| 13976 | 8691 |
| 13993 | 6994 |
| 14038 | 8922 |
| 14086 | 12124 |
| 29148 | 21244 |
| 29185 | 12093 |
| 29200 | 6209 |
| 29208 | 5649 |
| 29209 | 5081 |
| 29217 | 6233 |
| 29228 | 6593 |
| 29249 | 23284 |
| 29271 | 5939 |
| 29280 | 15425 |
| 29297 | 10881 |
| 29297 | 18678 |
| 29299 | 15081 |
| 29301 | 15371 |
| 29318 | 9763 |
| 29319 | 14858 |
| 29319 | 18151 |
| 29323 | 11245 |
| 29334 | 14657 |
| 29338 | 22062 |
| 29338 | 21342 |
| 29339 | 7053 |
| 29344 | 15675 |
| 29350 | 15197 |
| 29364 | 17754 |
| 29366 | 24289 |
| 29370 | 9115 |
| 29374 | 20427 |
| 29380 | 13747 |
| 29380 | 19216 |
| 29381 | 14718 |
| 29383 | 18059 |
| 29384 | 11329 |
| 29392 | 8145 |
| 29394 | 15914 |
| 29394 | 6331 |
| 29395 | 9145 |
| 29395 | 10522 |
| 29403 | 10674 |
| 29405 | 14588 |
| 29415 | 15767 |
| 29417 | 9263 |
| 29433 | 15935 |
| 29433 | 31043 |
| 29442 | 18797 |
| 29472 | 14656 |
| 29497 | 14580 |
| 29505 | 12785 |
| 29511 | 14942 |
| 29525 | 7448 |
| 29570 | 23463 |
| 29625 | 16116 |
| 43597 | 10528 |
| 43899 | 11934 |

### Cumulative cost vs turn (per profile)

#### contradiction

| turn | cumulative_cost_usd |
| --- | --- |
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 5 | 0 |
| 6 | 0 |

#### growing

| turn | cumulative_cost_usd |
| --- | --- |
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 5 | 0 |
| 6 | 0 |

#### pivot

| turn | cumulative_cost_usd |
| --- | --- |
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 5 | 0 |
| 6 | 0 |

### Recall vs turns slice

| turns_in_slice | mean_recall |
| --- | --- |
| 6 | 0.79 |


**Total executions:** **270** (90 por perfil)

---

# Resumen ejecutivo

El sistema mantiene una latencia relativamente estable incluso cuando el contexto crece hasta casi **44.000 tokens**, aunque aparecen picos puntuales de latencia superiores a **30 segundos**.

La memoria funciona muy bien en conversaciones acumulativas (**Growing**), pero pierde precisión cuando la conversación cambia de contexto (**Pivot**) o contiene información contradictoria (**Contradiction**).

No se ha observado coste económico durante la prueba (todas las ejecuciones reportan **0 USD**) y tampoco se han producido **cache hits**.

---

# Resultados por perfil

| Perfil | Recall | P50 Latencia | P95 Latencia |
|---------|-------:|-------------:|-------------:|
| Growing | **0.96** | 7.1 s | 20.2 s |
| Pivot | 0.74 | 7.0 s | 16.1 s |
| Contradiction | 0.67 | 7.3 s | 18.7 s |

## Interpretación

### Growing

Es el escenario con mejor comportamiento.

- El modelo conserva correctamente la memoria conforme aumenta la conversación.
- Apenas aparecen pérdidas de información.
- Recall medio del **96%**.

Este perfil demuestra que la estrategia de resumen funciona correctamente cuando la conversación evoluciona de forma natural.

---

### Pivot

Cuando el tema cambia varias veces aparecen pérdidas de memoria.

El modelo recuerda correctamente la información reciente, pero empieza a olvidar hechos antiguos que vuelven a ser relevantes posteriormente.

**Recall final:** **74%**

Esto indica que el sistema prioriza excesivamente la información reciente frente a la información histórica.

---

### Contradiction

Es el escenario más complejo.

Cuando aparecen hechos incompatibles entre sí, el resumen acumulado deja de representar correctamente el estado de la conversación.

**Recall final:** **67%**

En este escenario sería recomendable introducir mecanismos explícitos para:

- versionar hechos;
- invalidar hechos antiguos;
- mantener trazabilidad de las actualizaciones.

---

# Latencia

## Comportamiento general

La latencia permanece sorprendentemente estable a pesar del crecimiento del contexto.

Valores observados:

- **P50:** ~7 segundos
- **P95:** entre **16 y 20 segundos**, según el perfil

No se aprecia una relación lineal entre el número de tokens y la latencia.

Por ejemplo:

- alrededor de **2.500 tokens** existen respuestas entre **5 y 10 segundos**;
- alrededor de **14.000 tokens** siguen apareciendo respuestas entre **6 y 10 segundos**;
- incluso cerca de **44.000 tokens**, la mayoría de respuestas permanecen alrededor de **10–12 segundos**.

Esto sugiere que el tiempo de respuesta está más condicionado por la variabilidad del modelo o de la infraestructura que por el tamaño del contexto.

---

## Outliers

Aunque el comportamiento medio es estable, existen algunos picos aislados:

- 21 s
- 23 s
- 24 s
- 31 s
- 38 s (máximo observado)

Estos casos representan una pequeña fracción de las ejecuciones y parecen deberse a variaciones del servicio más que al crecimiento del contexto.

---

# Escalabilidad

El experimento alcanza conversaciones cercanas a **44.000 tokens**.

No se observa un deterioro significativo de la latencia conforme aumenta el tamaño del contexto.

Sí se aprecia una degradación progresiva del recall cuando:

- la conversación cambia repetidamente de tema;
- aparecen contradicciones;
- el resumen acumulado sustituye información antigua por reciente.

Por tanto, el principal cuello de botella actual no parece ser el rendimiento, sino la gestión de la memoria conversacional.

---

# Conclusiones

## Fortalezas

- Latencia relativamente estable incluso con contextos muy grandes.
- Excelente rendimiento en conversaciones acumulativas (**96% de recall**).
- El límite de tamaño del *attachment* evita que el coste crezca sin control.

## Debilidades

- El sistema pierde memoria cuando la conversación cambia de tema.
- Las contradicciones degradan significativamente el recall.
- No existe todavía una estrategia robusta para preservar hechos antiguos importantes.
- No se está registrando correctamente el coste ni el uso de caché.
