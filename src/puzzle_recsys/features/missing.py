"""Tratamiento de faltantes: mediana, KNN e indicadores de faltante.

Reglas:

1. **Se aprende solo del entrenamiento.** La mediana (o los vecinos) se calculan
   con el train y se aplican a valid y test; si no, el test "contamina" al modelo.
2. **Indicador de faltante:** que falte un dato a veces es información (p. ej.
   "primera visita: no hay hábito previo"). La columna ``<var>__faltante`` la conserva.
3. **LightGBM no necesita imputar** (maneja NaN de forma nativa); la imputación es
   para la regresión logística de referencia.

Usado en: notebooks 10, 11.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Porcentaje de faltantes por columna (solo las que tienen alguno), de mayor a menor."""
    pct = df.isna().mean().mul(100)
    pct = pct[pct > 0].sort_values(ascending=False)
    return pct.rename("pct_faltante").rename_axis("variable").reset_index()


class MedianImputer:
    """Imputa con la mediana del entrenamiento y agrega indicadores de faltante."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = list(columns)

    def fit(self, train: pd.DataFrame) -> MedianImputer:
        """Aprende la mediana de cada columna en el entrenamiento."""
        self.medians_ = train[self.columns].median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Devuelve ``df`` con las columnas imputadas y sus indicadores ``__faltante``."""
        out = df.copy()
        for col in self.columns:
            out[f"{col}__faltante"] = out[col].isna().astype(np.int8)
            out[col] = out[col].fillna(self.medians_[col])
        return out


def imputation_error(
    df: pd.DataFrame,
    *,
    target: str,
    predictors: Sequence[str],
    k: int,
    mask_share: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compara mediana contra KNN escondiendo a propósito valores **conocidos**.

    Se toma una fracción ``mask_share`` de las filas con ``target`` observado, se
    borra su valor, se imputa con cada método y se mide el error absoluto medio
    (MAE) contra el valor real. Así se elige el método con evidencia.
    """
    known = df.dropna(subset=[target, *predictors]).reset_index(drop=True)
    hide = rng.random(len(known)) < mask_share
    truth = known.loc[hide, target].to_numpy()
    masked = known[[target, *predictors]].copy()
    masked.loc[hide, target] = np.nan

    median_pred = np.full(hide.sum(), masked[target].median())
    scaled = masked.copy()
    scaled[list(predictors)] = StandardScaler().fit_transform(masked[list(predictors)])
    knn = KNNImputer(n_neighbors=k).fit_transform(scaled)[:, 0]
    rows = [
        {"metodo": "mediana", "MAE": float(np.abs(median_pred - truth).mean())},
        {"metodo": "KNN", "MAE": float(np.abs(knn[hide] - truth).mean())},
    ]
    return pd.DataFrame(rows).assign(n_ocultos=int(hide.sum()))
