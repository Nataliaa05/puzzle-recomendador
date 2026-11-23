"""LightGBM LambdaRank para "para ti" y "para la mesa".

**Por qué LambdaRank y no un clasificador:** la regresión logística (notebook 11)
aprende "¿lo va a pedir?" para cada platillo por separado. LambdaRank aprende
directamente a **ordenar** dentro de cada consulta: su función de pérdida está
construida para mejorar NDCG, así que solo le importa que los positivos queden
**arriba de los negativos de la misma consulta**.

Detalles que importan:

- Las filas de una consulta deben estar **contiguas**; ``group`` dice cuántas
  filas tiene cada consulta.
- Las categóricas (incluido el canónico) entran como categoría **nativa** de
  LightGBM, con las categorías fijadas en el entrenamiento: una categoría nueva
  queda como faltante en vez de recibir un código inventado.
- Los NaN se dejan: LightGBM aprende hacia qué lado mandarlos (notebook 10).

Usado en: notebooks 12 a 15.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from puzzle_recsys.ranking.feature_sets import FeatureSet, model_frame


@dataclass(frozen=True)
class LgbmData:
    """Matriz lista para LightGBM, ordenada por consulta."""

    X: pd.DataFrame
    y: np.ndarray
    group: np.ndarray
    query_id: np.ndarray
    categories: dict[str, list[str]]


def to_lgbm(
    df: pd.DataFrame, features: FeatureSet, categories: Mapping[str, list[str]] | None = None
) -> LgbmData:
    """Prepara ``df`` para LightGBM; ``categories`` se aprende si no se da (solo en train)."""
    frame = model_frame(df, features).sort_values("query_id", kind="stable")
    cat_cols = [*features.categorical, *features.target_encoded]
    if categories is None:
        categories = {c: sorted(frame[c].unique()) for c in cat_cols}
    x = frame[list(features.dense)].copy()
    for c in cat_cols:
        known = frame[c].where(frame[c].isin(categories[c]))  # nueva -> faltante
        x[c] = pd.Categorical(known, categories=categories[c])
    qid = frame["query_id"].to_numpy()
    _, sizes = np.unique(qid, return_counts=True)
    y = frame["y"].to_numpy() if "y" in frame else np.zeros(len(frame))
    return LgbmData(X=x, y=y, group=sizes, query_id=qid, categories=dict(categories))


def fit_ranker(
    train: LgbmData,
    valid: LgbmData,
    params: Mapping[str, Any],
    *,
    num_boost_round: int,
    early_stopping_rounds: int,
    seed: int,
    eval_at: int = 3,
) -> lgb.Booster:
    """Entrena LambdaRank con early stopping sobre NDCG@``eval_at`` en ``valid``."""
    base = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [eval_at],
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(train.X, train.y, group=train.group, free_raw_data=False)
    dvalid = lgb.Dataset(valid.X, valid.y, group=valid.group, reference=dtrain)
    return lgb.train(
        base | dict(params),
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )


def predict(
    booster: lgb.Booster,
    df: pd.DataFrame,
    features: FeatureSet,
    categories: Mapping[str, list[str]],
) -> np.ndarray:
    """Puntaje de cada fila de ``df``, en el **mismo orden** que ``df``."""
    data = to_lgbm(df, features, categories)
    scores = booster.predict(data.X, num_iteration=booster.best_iteration)
    return pd.Series(scores, index=data.X.index).reindex(df.index).to_numpy()
