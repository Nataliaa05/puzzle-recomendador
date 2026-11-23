"""SHAP para LightGBM: valores por fila, importancia por grupo y gráficas de dependencia.

**SHAP** reparte el puntaje de cada candidato entre sus features: el puntaje es
la suma de un valor base más una contribución por feature (valores de Shapley,
de teoría de juegos). Una contribución positiva empuja el platillo **hacia arriba**
en el ranking.

Se usa el TreeSHAP **nativo** de LightGBM (``pred_contrib=True``): es exacto para
árboles, rápido y entiende las categóricas nativas (canónico, tipo de plato) sin
codificarlas.

**Variables correlacionadas:** si dos features dicen casi lo mismo (p. ej.
popularidad en el restaurante y pedidos previos del platillo), SHAP reparte el
crédito entre las dos de forma algo arbitraria. Por eso además se reporta la
importancia **por grupo** de features con el mismo significado.

Usado en: notebooks 14.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def tree_shap(booster: lgb.Booster, x: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Valores SHAP por fila y feature, y el valor base (en la escala del puntaje)."""
    contrib = booster.predict(x, pred_contrib=True)
    values = pd.DataFrame(contrib[:, :-1], columns=x.columns, index=x.index)
    return values, float(contrib[0, -1])


def group_importance(shap_values: pd.DataFrame, groups: Mapping[str, Sequence[str]]) -> pd.Series:
    """Importancia de cada grupo: media de |suma de los SHAP del grupo| por fila."""
    out = {name: shap_values[list(cols)].sum(axis=1).abs().mean() for name, cols in groups.items()}
    return pd.Series(out).sort_values(ascending=False)


def plot_dependence(
    x: pd.Series, shap: pd.Series, *, title: str, ax: Axes | None = None, bins: int = 20
) -> Axes:
    """Nube del valor SHAP contra el valor de la feature, con la media por tramo."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ok = x.notna()
    xv, sv = x[ok].astype(float), shap[ok]
    ax.scatter(xv, sv, s=3, alpha=0.15, color="#4C72B0")
    edges = np.unique(np.quantile(xv, np.linspace(0, 1, bins + 1)))
    if len(edges) > 2:
        mids = pd.cut(xv, edges, include_lowest=True)
        mean = sv.groupby(mids, observed=True).mean()
        centers = xv.groupby(mids, observed=True).median()
        ax.plot(
            centers.to_numpy(),
            mean.to_numpy(),
            color="#C44E52",
            linewidth=2,
            label="media por tramo",
        )
        ax.legend(fontsize=7)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(x.name)
    ax.set_ylabel("SHAP")
    return ax


def relative_to_query(values: pd.DataFrame, query_ids: np.ndarray | pd.Series) -> pd.DataFrame:
    """Cada valor menos el promedio de su consulta.

    En un ranking solo importa lo que **distingue** a un candidato de los demás de
    su consulta. Una feature igual para todos los candidatos (p. ej. el ticket
    habitual de la persona) mueve el puntaje de todos por igual y no cambia el
    orden: su valor relativo es 0.
    """
    q = np.asarray(query_ids)
    return values - values.groupby(q).transform("mean")


def top_reasons(row: pd.Series, n: int = 3) -> str:
    """Las ``n`` contribuciones de mayor magnitud, como texto: ``feature (+0.12)``."""
    top = row.reindex(row.abs().sort_values(ascending=False).index[:n])
    return ", ".join(f"{k} ({v:+.2f})" for k, v in top.items())
