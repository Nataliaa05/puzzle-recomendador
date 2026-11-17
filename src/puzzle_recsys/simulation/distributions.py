"""Distribuciones y muestreadores de la simulación.

Todas las funciones aleatorias reciben un ``numpy.random.Generator`` como
argumento (nunca usan estado global), así cada corrida es reproducible a
partir de la semilla de ``config/simulation.yaml``.

Usado en: notebooks 01, 02, 03.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def allocate_proportional(sizes: np.ndarray, n: int) -> np.ndarray:
    """Reparte ``n`` unidades entre estratos, proporcional a su tamaño.

    Usa el método del mayor resto (Hamilton) y respeta la capacidad de cada
    estrato: si a uno le tocaría más de lo que tiene, el excedente se
    redistribuye entre los demás.

    Parameters
    ----------
    sizes : np.ndarray
        Tamaño de cada estrato (enteros positivos).
    n : int
        Total a asignar.

    Returns
    -------
    np.ndarray
        Enteros con suma ``n`` y ``0 <= alloc[i] <= sizes[i]``.

    Raises
    ------
    ValueError
        Si ``n`` es negativo o mayor que la población.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    total = int(sizes.sum())
    if n < 0 or n > total:
        raise ValueError(f"n={n} es negativo o mayor que la población ({total})")

    alloc = np.zeros_like(sizes)
    remaining = n
    open_ = sizes > 0
    while remaining > 0:
        capacity = sizes - alloc
        weights = np.where(open_, sizes, 0).astype(float)
        quota = remaining * weights / weights.sum()
        base = np.minimum(np.floor(quota).astype(np.int64), capacity)
        leftover = remaining - int(base.sum())
        if leftover > 0:
            remainders = np.where(open_ & (capacity - base > 0), quota - np.floor(quota), -1.0)
            for idx in np.argsort(-remainders, kind="stable")[:leftover]:
                if remainders[idx] >= 0:
                    base[idx] += 1
        alloc += base
        remaining = n - int(alloc.sum())
        open_ = alloc < sizes
    return alloc


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    strata: Sequence[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Muestra ``n`` filas sin reemplazo con asignación proporcional por estrato.

    Parameters
    ----------
    df : pd.DataFrame
        Población.
    n : int
        Tamaño de la muestra.
    strata : Sequence[str]
        Columnas que definen los estratos.
    rng : np.random.Generator
        Generador aleatorio.

    Returns
    -------
    pd.DataFrame
        Muestra con el índice original, ordenada por estrato.
    """
    # Se itera sobre los grupos (no se buscan por clave): en pandas 3 la clave
    # de un groupby con una sola columna cambia de escalar a tupla.
    groups = [group for _, group in df.groupby(list(strata), sort=True, observed=True)]
    alloc = allocate_proportional(np.array([len(g) for g in groups]), n)

    parts = []
    for group, k in zip(groups, alloc, strict=True):
        if k == 0:
            continue
        chosen = rng.choice(len(group), size=int(k), replace=False)
        parts.append(group.iloc[np.sort(chosen)])
    return pd.concat(parts) if parts else df.iloc[0:0]
