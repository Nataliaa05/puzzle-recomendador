"""Features de calendario: franja de comida y fin de semana.

La hora exacta tiene 24 valores con poca señal cada uno; la **franja** agrupa
las horas según el horario de comidas en México (comida fuerte de 12 a 17 h).
Es la única definición de franja del proyecto: la usan el simulador y el EDA.

Usado en: notebooks 03, 07, 10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MEAL_PERIODS = ("desayuno", "comida", "cena")
WEEKEND_DAYS = (4, 5, 6)  # viernes a domingo (0 = lunes)


def period_index(hours: np.ndarray) -> np.ndarray:
    """Índice de franja por hora local: 0 desayuno (5 a 11), 1 comida (12 a 17), 2 cena."""
    h = np.asarray(hours)
    return np.select([(h >= 5) & (h < 12), (h >= 12) & (h < 18)], [0, 1], 2)


def meal_period(hours: pd.Series) -> pd.Series:
    """Franja de comida por hora local, como categórica ordenada (``MEAL_PERIODS``)."""
    labels = np.asarray(MEAL_PERIODS)[period_index(hours.to_numpy())]
    return pd.Series(
        pd.Categorical(labels, categories=MEAL_PERIODS, ordered=True), index=hours.index
    )
