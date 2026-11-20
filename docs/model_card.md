# Model card: recomendador de platillos de Puzzle

> Se completa en el notebook 15, con los resultados finales.

## Uso previsto

Ordenar los platillos del menú de un restaurante para un comensal registrado de Puzzle, en dos listas: "para ti" y "para la mesa".

## Usos no previstos

- Decidir sobre la seguridad alimentaria: las alergias son un **filtro duro** previo, no una salida del modelo.
- Recomendar restaurantes (fase 2).

## Datos de entrenamiento

Sintéticos (ver [`supuestos.md`](supuestos.md)), anclados en fuentes públicas del INEGI y de SEPOMEX.

## Métricas

NDCG@5, HitRate@3, Recall@5 y MAP@5 por tipo de plato y por segmento, con intervalos de confianza bootstrap. _Pendiente._

## Limitaciones y sesgos conocidos

- Rendimiento validado sobre datos simulados, no reales.
- Sesgo de popularidad y de posición (se mide con `RecommendationSlate` y con la prueba A/B).
- Los comensales sin historial dependen del onboarding y de la popularidad.
- **Señales débiles en la simulación** (el modelo no podrá aprenderlas bien porque los datos casi no las contienen):
  - **Ocasiones especiales:** van a lugares solo ~5 % más caros que una visita normal (nb 03).
  - **Distancia:** quien más evita trasladarse va a restaurantes a 1.46 km contra 1.8 km de quien menos (nb 03), porque el conjunto de consideración ya son los 30 más cercanos.
- **Parámetros no identificados:** la calibración (nb 04) muestra que los hechos estilizados se cumplen en todo el rango explorado; los valores exactos de los parámetros requieren datos reales.

## Privacidad

Solo comensales registrados y con consentimiento de personalización. Las alergias son datos sensibles (LFPDPPP).
