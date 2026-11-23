"""Pruebas pareadas por comensal: permutación y bootstrap pareado.

**Pareadas:** los dos modelos se evalúan en **las mismas consultas**, así que se
compara la diferencia consulta por consulta (``diff = métrica_B - métrica_A``).
Eso quita la variación entre consultas fáciles y difíciles y da mucha más
potencia que comparar dos promedios sueltos.

**Por comensal:** las consultas de una persona no son independientes, así que la
aleatorización se hace por persona (ver ``resampling.py``).

- **Permutación (cambio de signo):** si los modelos fueran iguales, la diferencia
  de cada persona podría tener cualquier signo. Se invierten signos al azar por
  persona miles de veces; el p-valor es la proporción de veces que la diferencia
  así obtenida es al menos tan grande como la observada.
- **Bootstrap pareado:** remuestrea personas y da el intervalo de confianza de la
  diferencia media. Si el intervalo no incluye 0, la diferencia es real.

Usado en: notebooks 13, 15.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from puzzle_recsys.evaluation.resampling import bootstrap_means, cluster_sums


def paired_permutation(
    diff: pd.Series, clusters: pd.Series, *, n_iter: int = 5000, seed: int = 0
) -> dict[str, float]:
    """Prueba de permutación por cambio de signo, a nivel persona (dos colas)."""
    s, n = cluster_sums(diff, clusters)
    total = n.sum()
    observed = s.sum() / total
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_iter, len(s)))
    perm = (signs @ s) / total
    p = (1 + np.sum(np.abs(perm) >= abs(observed))) / (n_iter + 1)
    return {"diferencia": float(observed), "p": float(p)}


def paired_bootstrap(
    diff: pd.Series,
    clusters: pd.Series,
    *,
    n_iter: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Diferencia media, su intervalo bootstrap por persona y un p-valor de dos colas."""
    means = bootstrap_means(diff, clusters, n_boot=n_iter, seed=seed)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    tail = min(np.mean(means <= 0), np.mean(means >= 0))
    p = max(min(1.0, 2 * tail), 1 / (n_iter + 1))
    return {
        "diferencia": float(diff.mean()),
        "inferior": float(lo),
        "superior": float(hi),
        "p": float(p),
    }
