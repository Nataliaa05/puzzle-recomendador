# 0003 · Todas las variables se calculan "a la fecha"

**Estado:** Aceptada (2026-11-16)

## Contexto

Si un perfil, una popularidad o una estadística se calcula con todo el dataset, incluye información del periodo de prueba, y la evaluación sale inflada.

## Decisión

- Perfiles, popularidad (del restaurante y global), estadísticas y codificaciones usan **solo** datos anteriores a cada visita.
- La partición es temporal: train en los meses 1–9 (con validación cruzada por origen móvil), valid en el mes 10 y test en los meses 11–12, que se usa una sola vez.
- Los negativos son los platillos del menú **disponible a la fecha** que no se pidieron. Para eso se usan el historial de precio y el de disponibilidad.
- Imputadores y codificadores se ajustan dentro de un `Pipeline`, solo con train.

## Consecuencias

- Puzzle necesita `MenuItemPriceHistory` y `MenuItemAvailabilityEvent`.
- Hay tests de propiedades que verifican que ninguna variable use datos posteriores a su visita.
