"""One-hot (baja cardinalidad) y target encoding por pliegue (alta cardinalidad).

- **One-hot** para variables con pocas categorías (tipo de plato, proteína,
  franja, cocina): una columna 0/1 por categoría. Las categorías se fijan con el
  **entrenamiento**; lo raro o nuevo cae en "otras", así valid y test siempre
  tienen las mismas columnas.
- **Target encoding** para variables con miles de categorías (restaurante,
  canónico): se reemplaza la categoría por la tasa media de la etiqueta en esa
  categoría, **suavizada** hacia la media global. En el entrenamiento se calcula
  **fuera de pliegue** (con las otras partes del train), porque si una fila
  usara su propia etiqueta el modelo "vería la respuesta".

Usado en: notebooks 10, 11.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


class OneHotTop:
    """One-hot con las ``top`` categorías más frecuentes del entrenamiento (el resto: "otras")."""

    def __init__(self, columns: Sequence[str], *, top: int | None = None) -> None:
        self.columns = list(columns)
        self.top = top

    def fit(self, train: pd.DataFrame) -> OneHotTop:
        """Fija las categorías de cada columna con el entrenamiento."""
        self.categories_ = {}
        for col in self.columns:
            counts = train[col].astype(str).value_counts()
            keep = counts.index[: self.top] if self.top else counts.index
            self.categories_[col] = sorted(keep)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Columnas ``<var>=<categoría>`` (0/1), siempre las mismas."""
        parts = []
        for col in self.columns:
            cats = self.categories_[col]
            values = df[col].astype(str)
            values = values.where(values.isin(cats), "otras")
            dummies = pd.get_dummies(values, prefix=col, prefix_sep="=", dtype=np.int8)
            names = [f"{col}={c}" for c in cats] + [f"{col}=otras"]
            parts.append(dummies.reindex(columns=names, fill_value=0))
        return pd.concat(parts, axis=1)


def _smoothed(sums: pd.Series, counts: pd.Series, prior: float, m: float) -> pd.Series:
    return (sums + m * prior) / (counts + m)


def target_encode_oof(
    df: pd.DataFrame, *, col: str, target: str, folds: pd.Series, smoothing: float
) -> pd.Series:
    """Target encoding **fuera de pliegue**: cada fila se codifica con los otros pliegues."""
    out = pd.Series(np.nan, index=df.index)
    for f in pd.unique(folds):
        fit = df[folds != f]
        stats = fit.groupby(col)[target].agg(["sum", "count"])
        prior = float(fit[target].mean())
        enc = _smoothed(stats["sum"], stats["count"], prior, smoothing)
        rows = folds == f
        out[rows] = df.loc[rows, col].map(enc).fillna(prior).to_numpy()
    return out


class TargetEncoder:
    """Target encoding suavizado aprendido con **todo** el entrenamiento (para valid y test)."""

    def __init__(self, col: str, *, smoothing: float) -> None:
        self.col = col
        self.smoothing = smoothing

    def fit(self, train: pd.DataFrame, *, target: str) -> TargetEncoder:
        """Aprende la tasa suavizada por categoría."""
        stats = train.groupby(self.col)[target].agg(["sum", "count"])
        self.prior_ = float(train[target].mean())
        self.mapping_ = _smoothed(stats["sum"], stats["count"], self.prior_, self.smoothing)
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Codifica; las categorías nunca vistas reciben la media global."""
        return df[self.col].map(self.mapping_).fillna(self.prior_)
