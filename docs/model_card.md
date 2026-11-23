# Model card: recomendador de platillos de Puzzle

> Resultados finales del notebook 15 (prueba: meses 11 y 12 de la simulación).

## Uso previsto

Ordenar los platillos del menú de un restaurante para un comensal registrado de Puzzle, en dos listas: "para ti" y "para la mesa".

## Usos no previstos

- Decidir sobre la seguridad alimentaria: las alergias son un **filtro duro** previo, no una salida del modelo.
- Recomendar restaurantes (fase 2).

## Datos de entrenamiento

Sintéticos (ver [`supuestos.md`](supuestos.md)), anclados en fuentes públicas del INEGI y de SEPOMEX.

## Métricas

**Modelo:** LightGBM LambdaRank (`models/lgbm_para_ti.txt`, `models/lgbm_para_la_mesa.txt`), con 33 features calculadas a la fecha (`ranking/feature_sets.py`).

Prueba (81,225 consultas, 5,919 comensales); intervalos bootstrap del 95 % por comensal:

| "Para ti" | NDCG@3 | HitRate@3 |
|---|---|---|
| Popularidad (línea base) | 0.655 (0.652 a 0.658) | 0.792 |
| Regresión logística | 0.703 (0.700 a 0.706) | 0.839 |
| **LightGBM** | **0.735 (0.733 a 0.738)** | **0.864** |

- Mejora sobre la popularidad: **+0.080** de NDCG@3 (IC 0.078 a 0.083). Sobre la logística: **+0.032** (IC 0.031 a 0.034). El criterio de éxito era de +0.01.
- **Primera vez en el restaurante:** 0.727. **Persona nueva** (contrafactual): 0.692. **Restaurante nuevo** (contrafactual): 0.710.
- **"Para la mesa":** Hit@1 = 0.831 (IC 0.821 a 0.841), contra 0.742 de la popularidad de compartir.
- **Cobertura del catálogo en el top 3:** 79 %, contra 61 % de la popularidad.

**Explicabilidad** (SHAP relativo a la consulta, notebook 14): 12 de 12 signos coinciden con la verdad simulada. El picante actúa como bisagra. La cocina está subestimada respecto a la verdad.

## Limitaciones y sesgos conocidos

- Rendimiento validado sobre datos simulados, no reales.
- Sesgo de popularidad y de posición (se mide con `RecommendationSlate` y con la prueba A/B).
- Los comensales sin historial dependen del onboarding y de la popularidad.
- **Señales débiles en la simulación** (el modelo no podrá aprenderlas bien porque los datos casi no las contienen):
  - **Ocasiones especiales:** van a lugares solo ~5 % más caros que una visita normal (nb 03).
  - **Distancia:** quien más evita trasladarse va a restaurantes a 1.46 km contra 1.8 km de quien menos (nb 03), porque el conjunto de consideración ya son los 30 más cercanos.
  - **Precio dentro del menú:** el precio relativo casi no explica la popularidad de un platillo (ρ = 0.05, nb 07); pesan más los gustos.
  - **Sensibilidad al precio y gusto por la variedad:** casi no se ven en el comportamiento (ρ = 0.10 con el ticket y ρ = 0.04 con los restaurantes distintos, nb 08); el ticket lo fija el presupuesto y la elección de lugar, la lealtad.
- **El efecto del contexto es un supuesto, no una estimación** (S16): los multiplicadores de cocina, tipo de plato, platillo y tamaño de mesa por franja siguen hábitos conocidos de la CDMX. Son los **primeros parámetros a recalibrar** con datos reales de Puzzle; lo que el modelo aprenda de la hora refleja estos supuestos.
- **Compartir según quién se sienta es un supuesto** (S17): la forma (proporcional al gusto medio de la mesa) y el peso se recalibran con datos reales, junto con el contexto.
- **Sin segmentos de comensal:** los gustos simulados son continuos e independientes, así que el clustering no encuentra grupos (silueta ≤ 0.10, nb 08). El modelo usa el perfil continuo; con datos reales se vuelve a probar.
- **Cocina e ingredientes subestimados** (nb 14): la afinidad por cocina se calcula con los restaurantes visitados, no con los platillos pedidos, y no hay feature de ingredientes. Son mejoras de fase 2.
- **Muestra:** entrenado con el 30 % de las cuentas registradas (6,061) por la memoria disponible (nb 10).
- **Parámetros no identificados:** la calibración (nb 04) muestra que los hechos estilizados se cumplen en todo el rango explorado; los valores exactos de los parámetros requieren datos reales.

## Privacidad

Solo comensales registrados y con consentimiento de personalización. Las alergias son datos sensibles (LFPDPPP).
