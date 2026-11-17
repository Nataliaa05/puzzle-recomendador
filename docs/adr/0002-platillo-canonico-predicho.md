# 0002 · Transferencia por platillo canónico predicho

**Estado:** Aceptada (2026-11-16)

## Contexto

En un restaurante nuevo, el comensal no tiene interacciones con sus platillos: los IDs son distintos en cada restaurante, así que el filtrado colaborativo por ID no transfiere. Sí transfieren los atributos y el "qué platillo es" ("Spaghetti alla carbonara" = `carbonara`).

## Decisión

- Existe un catálogo `CanonicalDish` con sus alias.
- Una homologación (normalización + n-gramas de caracteres + KNN) sugiere el canónico; el restaurante lo confirma o lo corrige.
- **Las variables usan `canonical_pred`**, el resultado de ese proceso, **nunca `canonical_true`**, que vive solo en `ground_truth/`.
- La simulación imita la confirmación humana con una tasa configurable.

## Consecuencias

- La evaluación refleja los errores reales de la homologación, en lugar de una homologación perfecta que no existirá en producción.
- La exactitud de la homologación se mide por separado (notebook 09).
- Hay un test de propiedades que verifica que ninguna variable lea `canonical_true`.
