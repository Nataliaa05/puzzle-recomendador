# 0005 · Puertos y adaptadores: el mismo código para datos sintéticos y para Puzzle

**Estado:** Aceptada (2026-11-16)

## Contexto

El proyecto sirve para el curso (datos sintéticos) y para Puzzle (datos reales en Postgres). Si el modelo lee archivos de forma directa, habría que reescribirlo para llevarlo a producción.

## Decisión

- Un puerto `Repository` (`src/puzzle_recsys/io/base.py`) con dos adaptadores: `ParquetRepository` (curso) y `PostgresRepository` (Puzzle).
- Las tablas sintéticas siguen el contrato `docs/schema-recomendador.prisma` (en snake_case).
- La simulación escribe la verdad oculta en `ground_truth/`, fuera del contrato: solo la lee la evaluación.

## Consecuencias

- Para pasar a Puzzle, solo cambia el adaptador. Features, modelos y evaluación no se tocan.
- Cualquier cambio al esquema de Puzzle se refleja en el contrato y en la simulación.
