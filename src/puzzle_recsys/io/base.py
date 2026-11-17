"""Puerto de acceso a datos (arquitectura de puertos y adaptadores).

El resto del paquete (features, modelos, evaluación) solo conoce este
contrato. En el curso se usa ``ParquetRepository`` (datos sintéticos);
en Puzzle, ``PostgresRepository`` (datos reales). Cambiar de uno a otro
no toca el modelo.

Las tablas siguen el contrato ``docs/schema-recomendador.prisma`` en
snake_case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pandas as pd

#: Tablas del contrato de datos que el modelo puede leer.
CONTRACT_TABLES: tuple[str, ...] = (
    "restaurants",
    "canonical_dishes",
    "canonical_dish_aliases",
    "menu_items",
    "menu_item_attributes",
    "menu_item_price_history",
    "menu_item_availability_events",
    "diners",
    "diner_profiles",
    "dining_sessions",
    "guests",
    "order_items",
    "dish_ratings",
    "visit_ratings",
    "recommendation_slates",
)


class Repository(Protocol):
    """Contrato mínimo de lectura y escritura de tablas."""

    def read_table(self, name: str) -> pd.DataFrame:
        """Lee una tabla del contrato."""
        ...

    def write_table(self, name: str, df: pd.DataFrame) -> None:
        """Escribe una tabla del contrato."""
        ...
