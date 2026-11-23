"""Análisis de potencia Monte Carlo: probabilidad de detectar una mejora real.

Si una prueba no encuentra diferencia, puede ser que no la haya **o** que haya
pocos datos para verla. La potencia responde: "si la mejora real fuera de
``effect``, ¿en qué fracción de las veces la prueba la detectaría?".

Procedimiento (sobre las diferencias pareadas observadas):

1. Se centran las diferencias en 0 (mundo sin mejora) y se les suma ``effect``
   (mundo con una mejora conocida). Así se conserva la variabilidad real entre
   personas y consultas.
2. Se remuestrean ``size`` personas, se corre la prueba de permutación y se anota
   si ``p < alpha``.
3. La potencia es la fracción de simulaciones con detección. Lo convencional es
   pedir al menos 0.8.

Usado en: notebooks 13.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from puzzle_recsys.evaluation.resampling import cluster_sums


def power_curve(
    diff: pd.Series,
    clusters: pd.Series,
    *,
    effect: float,
    sizes: Sequence[int],
    n_sims: int = 200,
    n_iter: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Potencia de la permutación pareada por persona para cada número de personas."""
    s, n = cluster_sums(diff - diff.mean(), clusters)
    s = s + effect * n
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        detected = 0
        for _ in range(n_sims):
            pick = rng.integers(0, len(s), size)
            ss, nn = s[pick], n[pick]
            observed = ss.sum() / nn.sum()
            signs = rng.choice([-1.0, 1.0], size=(n_iter, size))
            perm = (signs @ ss) / nn.sum()
            p = (1 + np.sum(np.abs(perm) >= abs(observed))) / (n_iter + 1)
            detected += p < alpha
        rows.append({"comensales": size, "potencia": detected / n_sims})
    return pd.DataFrame(rows)
