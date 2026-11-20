"""Análisis univariado: distribuciones, asimetría, colas.

Cada variable se describe en su **unidad natural** (precio por platillo, ticket
por mesa, visitas por comensal): mezclar unidades es el error clásico del EDA.

Usado en: notebooks 06.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

_NORMALITY_SAMPLE = 5000


def describe_numeric(series: pd.Series, *, seed: int = 0) -> dict[str, Any]:
    """Resumen de una variable numérica pensado para decidir transformaciones.

    Además de los momentos, reporta la asimetría **después de log** (si todos
    los valores son positivos) y el p-valor de D'Agostino-Pearson sobre una
    muestra de 5,000. Con millones de filas cualquier prueba rechaza la
    normalidad por diferencias irrelevantes; muestrear mantiene la prueba
    interpretable.

    Returns
    -------
    dict
        ``n``, ``media``, ``mediana``, ``desv_est``, ``p01``, ``p99``,
        ``asimetria``, ``curtosis`` (exceso), ``asimetria_log``, ``p_normalidad``.
    """
    values = series.dropna().astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    sample = values if len(values) <= _NORMALITY_SAMPLE else rng.choice(values, _NORMALITY_SAMPLE)
    positive = bool((values > 0).all())
    return {
        "n": len(values),
        "media": float(values.mean()),
        "mediana": float(np.median(values)),
        "desv_est": float(values.std(ddof=1)),
        "p01": float(np.percentile(values, 1)),
        "p99": float(np.percentile(values, 99)),
        "asimetria": float(stats.skew(values)),
        "curtosis": float(stats.kurtosis(values)),
        "asimetria_log": float(stats.skew(np.log(values))) if positive else float("nan"),
        "p_normalidad": (
            float(stats.normaltest(sample).pvalue) if len(sample) >= 8 else float("nan")
        ),
    }


def frequency_table(
    series: pd.Series,
    *,
    top: int | None = None,
    other_label: str = "otras",
    order: list[Any] | None = None,
) -> pd.DataFrame:
    """Frecuencias de una categórica con faltantes explícitos y cola agrupada.

    Parameters
    ----------
    top : int, optional
        Categorías que se muestran; el resto se suma en ``other_label``.
    order : list, optional
        Orden natural de las categorías (días, estrellas, horas). El acumulado
        se calcula **en ese orden**; si no se da, en orden de frecuencia.

    Returns
    -------
    pd.DataFrame
        Índice = categoría; columnas ``n``, ``%`` y ``% acumulado``.
    """
    counts = series.astype(object).fillna("(faltante)").value_counts()
    if top is not None and len(counts) > top:
        counts = pd.concat([counts.iloc[:top], pd.Series({other_label: counts.iloc[top:].sum()})])
    if order is not None:
        counts = counts.reindex(order, fill_value=0)
    share = counts / counts.sum() * 100
    return pd.DataFrame({"n": counts, "%": share, "% acumulado": share.cumsum()})


def ticket_per_session(lines: pd.DataFrame) -> pd.Series:
    """Ticket de cada mesa en **pesos**: suma de cantidad por precio unitario."""
    amount = lines["quantity"] * lines["unit_price_cents"] / 100
    return amount.groupby(lines["session_id"]).sum()
