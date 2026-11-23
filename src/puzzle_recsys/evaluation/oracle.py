"""Ranking oráculo: ordenar por la utilidad **verdadera** (lee ``ground_truth/``).

Ningún modelo puede saber más que la verdad. Ordenar cada consulta por la
utilidad determinística verdadera da un **techo** para las métricas: cuánto se
podría acertar si se conocieran los gustos latentes exactos. Ni siquiera el
oráculo llega a 1, porque la elección simulada tiene una parte aleatoria
(Gumbel): a veces la persona pide algo que no era su favorito.

**Techo aproximado:** se usan las piezas reconstruibles de la utilidad (cocina,
proteína, ingredientes, picante, atractivo y calidad; ver
``explainability/ground_truth_check.py``) más el bono por franja del platillo
(desayuno, alcohol). El término de precio no se incluye porque depende de un
precio ideal por persona que no se guarda; por eso el techo real es algo **mayor**.

Usado en: notebooks 15.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from puzzle_recsys.explainability.ground_truth_check import true_components

_PIECES = ("cocina", "proteina", "ingredientes", "picante", "atractivo_calidad")


def oracle_scores(
    cands: pd.DataFrame,
    tastes: pd.DataFrame,
    items: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    period_bonus: np.ndarray | None = None,
) -> np.ndarray:
    """Utilidad verdadera aproximada de cada candidato (mismo orden que ``cands``).

    ``period_bonus`` es el bono por franja de cada fila (opcional).
    """
    parts = true_components(cands, tastes, items, weights)
    score = parts[list(_PIECES)].sum(axis=1).to_numpy()
    if period_bonus is not None:
        score = score + np.asarray(period_bonus, dtype=float)
    return score
