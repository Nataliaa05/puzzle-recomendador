"""Modelo base: regresión logística dentro de un ``Pipeline`` de scikit-learn.

El ``Pipeline`` garantiza que **todo lo que se aprende de los datos** (medianas,
categorías, codificaciones, escalas) se aprenda solo con los datos de ``fit``.
Así se puede usar sin fuga dentro de cualquier validación cruzada.

- Numéricas: mediana + indicador de faltante, y estandarización.
- Categóricas: one-hot con las categorías frecuentes (lo raro o nuevo, "otras").
- Alta cardinalidad (canónico): ``TargetEncoder`` de scikit-learn, que en
  ``fit_transform`` codifica **fuera de pliegue** (validación cruzada interna).

Es un modelo **puntual** (cada candidato por separado); para ordenar se usa su
probabilidad.

Usado en: notebooks 11.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from puzzle_recsys.ranking.feature_sets import FeatureSet


def make_logistic(features: FeatureSet, *, c: float = 1.0, seed: int = 0) -> Pipeline:
    """Pipeline de preprocesamiento + regresión logística (``c``: inverso de la regularización)."""
    pre = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                ),
                list(features.dense),
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=13),
                list(features.categorical),
            ),
            (
                "te",
                make_pipeline(
                    TargetEncoder(
                        target_type="binary", cv=KFold(5, shuffle=True, random_state=seed)
                    ),
                    StandardScaler(),
                ),
                list(features.target_encoded),
            ),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=c, max_iter=500))])
