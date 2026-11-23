"""Segmentos de evaluación y filtro de grupos sin positivos.

- **Tramos de historia:** el desempeño se reporta por cuántas visitas previas
  tenía la persona, porque al lanzar la app casi nadie tendrá historia.
- **Grupos sin positivos:** una consulta (persona, visita y tipo de plato) en la
  que no se pidió nada de ese tipo no enseña qué se prefiere. Se excluye y las
  métricas son **condicionales**: "dado que pidió postre, ¿acertamos cuál?".

Usado en: notebooks 10, 13, 15.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_BINS = (0, 1, 6, 21, np.inf)
HISTORY_LABELS = ("0", "1-5", "6-20", "21+")


def history_band(previous_visits: pd.Series) -> pd.Series:
    """Tramo de historia: ``0``, ``1-5``, ``6-20`` o ``21+`` visitas previas."""
    return pd.cut(previous_visits, HISTORY_BINS, right=False, labels=HISTORY_LABELS)


def drop_groups_without_positive(df: pd.DataFrame, *, group: str, label: str) -> pd.DataFrame:
    """Quita los grupos donde ninguna fila tiene etiqueta positiva."""
    keep = df.groupby(group, sort=False)[label].transform("max") > 0
    return df[keep]
