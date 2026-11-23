"""Intervalos de confianza bootstrap de métricas, **por comensal**.

Las consultas de una misma persona no son independientes: si a alguien el modelo
le atina, le atina en muchas de sus consultas. Remuestrear consultas sueltas
haría el intervalo demasiado angosto (falsa precisión). Por eso se remuestrea
**personas** (bootstrap por conglomerado) y se toman todas sus consultas.

Usado en: notebooks 13, 15.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_sums(values: pd.Series, clusters: pd.Series | None) -> tuple[np.ndarray, np.ndarray]:
    """Suma y número de valores por conglomerado (cada valor es su propio grupo si no hay)."""
    if clusters is None:
        v = values.to_numpy(dtype=float)
        return v, np.ones_like(v)
    g = pd.Series(values.to_numpy(dtype=float)).groupby(clusters.to_numpy())
    return g.sum().to_numpy(), g.size().to_numpy(dtype=float)


def bootstrap_means(
    values: pd.Series, clusters: pd.Series | None, *, n_boot: int, seed: int
) -> np.ndarray:
    """Medias de ``n_boot`` remuestreos (por conglomerado, con reemplazo)."""
    s, n = cluster_sums(values, clusters)
    k = len(s)
    w = np.random.default_rng(seed).multinomial(k, np.full(k, 1 / k), size=n_boot)
    return (w @ s) / (w @ n)


def bootstrap_ci(
    values: pd.Series,
    *,
    clusters: pd.Series | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Media e intervalo de confianza percentil ``1 - alpha``."""
    means = bootstrap_means(values, clusters, n_boot=n_boot, seed=seed)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"media": float(values.mean()), "inferior": float(lo), "superior": float(hi)}
