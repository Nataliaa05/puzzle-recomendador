"""Partición temporal train/valid/test y validación cruzada por origen móvil.

En un recomendador el futuro no puede entrenar al pasado: si se mezclaran
visitas al azar, el modelo aprendería de pedidos posteriores a la visita que
predice. Por eso la partición es **por fecha** (``config/models.yaml``):

- Meses 1 a 9: entrenamiento (con validación cruzada por origen móvil).
- Mes 10: validación (selección de modelo, early stopping).
- Meses 11 y 12: prueba, que se usa **una sola vez**.

Usado en: notebooks 10, 12.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def month_index(times: pd.Series, *, start: str, tz: str) -> pd.Series:
    """Mes relativo al inicio (1, 2, ...) en hora **local**, contando desde ``start``."""
    local = pd.to_datetime(times).dt.tz_convert(tz)
    origin = pd.Timestamp(start, tz=tz)
    months = (local.dt.year - origin.year) * 12 + (local.dt.month - origin.month)
    before_day = local.dt.day < origin.day
    return (months - before_day.astype(int) + 1).astype(int)


def assign_split(months: pd.Series, split: Mapping[str, Sequence[int]]) -> pd.Series:
    """Etiqueta ``train`` / ``valid`` / ``test`` según los rangos de meses (inclusivos)."""
    out = pd.Series(np.full(len(months), None, dtype=object), index=months.index)
    for name in ("train", "valid", "test"):
        lo, hi = split[name]
        out[months.between(lo, hi)] = name
    return out


def rolling_origin_folds(
    months: pd.Series, *, train_end: int, n_folds: int
) -> list[tuple[pd.Series, pd.Series]]:
    """Pliegues por origen móvil dentro del entrenamiento.

    El pliegue ``k`` valida en el mes ``train_end - n_folds + k`` y entrena con
    **todos los meses anteriores**: imita cómo se reentrenaría el modelo cada mes.
    """
    folds = []
    for m in range(train_end - n_folds + 1, train_end + 1):
        folds.append((months < m, months == m))
    return folds
