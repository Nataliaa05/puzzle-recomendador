"""Homologación de la cocina de un restaurante: reglas sobre el nombre + KNN.

El SCIAN del DENUE solo distingue algunos tipos (mariscos, tacos y tortas,
antojitos, cafeterías, pizzas). El resto se resuelve así:

1. **Reglas** (``data/external/cuisine_rules.yaml``): expresiones regulares
   sobre el nombre normalizado ("trattoria" → italiana). Si el nombre no
   dice nada, se usa el valor por defecto del código SCIAN.
2. **KNN** para lo que queda sin etiqueta: vecinos por n-gramas de caracteres
   del nombre, código SCIAN y ubicación.

Para que el KNN no aprenda a "leer la palabra clave", las palabras que
dispararon una regla se **quitan** del nombre antes de construir las
variables (``masked_name``). Así, los restaurantes etiquetados se parecen a
los no etiquetados, que por definición no tienen palabra clave.

Usado en: notebooks 01.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str | None) -> str:
    """Minúsculas, sin acentos y sin signos: ``"Café  Niño, S.A."`` → ``"cafe nino s a"``."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return _NON_ALNUM.sub(" ", text).strip()


@dataclass(frozen=True)
class CuisineRules:
    """Reglas de homologación (ordenadas) y valores por defecto por SCIAN."""

    rules: tuple[tuple[str, re.Pattern[str]], ...]
    scian_defaults: dict[str, str]
    #: Etiquetas que dependen de una MARCA (p. ej. cadenas): el KNN no debe
    #: aprenderlas, porque un restaurante sin marca no "se parece" a una cadena.
    non_transferable: frozenset[str] = frozenset()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CuisineRules:
        """Construye las reglas a partir del contenido del YAML."""
        rules = tuple((r["cuisine"], re.compile(r["pattern"])) for r in data["name_rules"])
        defaults = {str(k): v for k, v in (data.get("scian_defaults") or {}).items()}
        non_transferable = frozenset(
            r["cuisine"] for r in data["name_rules"] if r.get("transferable", True) is False
        )
        return cls(rules=rules, scian_defaults=defaults, non_transferable=non_transferable)

    @classmethod
    def from_yaml(cls, path: Path) -> CuisineRules:
        """Lee ``data/external/cuisine_rules.yaml``."""
        with path.open(encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))


def classify_by_rules(
    names: pd.Series, scian_codes: pd.Series, rules: CuisineRules
) -> pd.DataFrame:
    """Etiqueta la cocina con reglas sobre el nombre y, si no, con el SCIAN.

    Parameters
    ----------
    names : pd.Series
        Nombres de los establecimientos.
    scian_codes : pd.Series
        Código SCIAN de 6 dígitos (texto), alineado con ``names``.
    rules : CuisineRules
        Reglas cargadas.

    Returns
    -------
    pd.DataFrame
        ``normalized_name``, ``cuisine_rule`` (``NaN`` si nada coincide),
        ``rule_source`` (``"NAME"``, ``"SCIAN"`` o ``NaN``) y ``masked_name``.
    """
    normalized = names.map(normalize_name)
    labels, sources, masked = [], [], []
    for text, code in zip(normalized, scian_codes.astype(str), strict=True):
        label = next((c for c, pat in rules.rules if pat.search(text)), None)
        if label is not None:
            source = "NAME"
        elif code in rules.scian_defaults:
            label, source = rules.scian_defaults[code], "SCIAN"
        else:
            source = None
        stripped = text
        for _, pat in rules.rules:
            stripped = pat.sub(" ", stripped)
        labels.append(label)
        sources.append(source)
        masked.append(" ".join(stripped.split()))
    return pd.DataFrame(
        {
            "normalized_name": normalized.to_numpy(),
            "cuisine_rule": labels,
            "rule_source": sources,
            "masked_name": masked,
        },
        index=names.index,
    )


def _features(
    df: pd.DataFrame, *, geo_weight: float, scian_weight: float, min_df: int
) -> sparse.csr_matrix:
    """Nombre (n-gramas de caracteres, L2) + SCIAN (one-hot) + ubicación (estandarizada)."""
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df)
    try:
        text = tfidf.fit_transform(df["masked_name"].fillna(""))
    except ValueError:  # vocabulario vacío (todos los nombres quedaron vacíos)
        text = sparse.csr_matrix((len(df), 0))
    scian = OneHotEncoder(handle_unknown="ignore").fit_transform(df[["scian_code"]])
    geo = StandardScaler().fit_transform(df[["lat", "lng"]].to_numpy(dtype=float))
    return sparse.hstack([text, scian * scian_weight, sparse.csr_matrix(geo * geo_weight)]).tocsr()


def impute_cuisine_knn(
    df: pd.DataFrame,
    *,
    n_neighbors: int,
    min_confidence: float,
    default: str,
    geo_weight: float = 1.0,
    scian_weight: float = 1.0,
    min_df: int = 1,
    exclude_from_training: Collection[str] = (),
) -> pd.DataFrame:
    """Completa la cocina de los restaurantes sin etiqueta con KNN.

    Parameters
    ----------
    df : pd.DataFrame
        Debe tener ``masked_name``, ``scian_code``, ``lat``, ``lng`` y
        ``cuisine_rule`` (``NaN`` = sin etiqueta).
    n_neighbors : int
        Vecinos a consultar (ponderados por la inversa de la distancia).
    min_confidence : float
        Proporción mínima de votos de la clase ganadora; debajo se usa ``default``.
    default : str
        Cocina para los casos de baja confianza (supuesto declarado).
    geo_weight, scian_weight : float, optional
        Peso relativo de la ubicación y del SCIAN frente al nombre.
    min_df : int, optional
        Frecuencia mínima de un n-grama para entrar al vocabulario.
    exclude_from_training : Collection[str], optional
        Etiquetas que conservan su valor pero NO se usan para entrenar el KNN
        (ver :attr:`CuisineRules.non_transferable`).

    Returns
    -------
    pd.DataFrame
        Copia de ``df`` con ``cuisine_pred``, ``cuisine_confidence`` y
        ``cuisine_source`` (``"RULE"``, ``"KNN"`` o ``"DEFAULT"``).
    """
    out = df.copy()
    labeled = out["cuisine_rule"].notna().to_numpy()
    out["cuisine_pred"] = out["cuisine_rule"]
    out["cuisine_confidence"] = np.where(labeled, 1.0, np.nan)
    out["cuisine_source"] = np.where(labeled, "RULE", None)
    trainable = labeled & ~out["cuisine_rule"].isin(list(exclude_from_training)).to_numpy()
    if labeled.all() or not trainable.any():
        return out

    x = _features(out, geo_weight=geo_weight, scian_weight=scian_weight, min_df=min_df)
    knn = KNeighborsClassifier(
        n_neighbors=min(n_neighbors, int(trainable.sum())), weights="distance", algorithm="brute"
    )
    knn.fit(x[trainable], out.loc[trainable, "cuisine_rule"].to_numpy())
    proba = knn.predict_proba(x[~labeled])
    best = proba.argmax(axis=1)
    confidence = proba[np.arange(len(best)), best]
    predicted = knn.classes_[best]
    confident = confidence >= min_confidence

    idx = out.index[~labeled]
    out.loc[idx, "cuisine_pred"] = np.where(confident, predicted, default)
    out.loc[idx, "cuisine_confidence"] = confidence
    out.loc[idx, "cuisine_source"] = np.where(confident, "KNN", "DEFAULT")
    return out


def evaluate_knn_holdout(
    df: pd.DataFrame,
    *,
    holdout_fraction: float,
    rng: np.random.Generator,
    **knn_kwargs: Any,
) -> dict[str, float]:
    """Mide la exactitud de la imputación ocultando etiquetas conocidas.

    Se ocultan etiquetas que vinieron del **nombre** (``rule_source == "NAME"``):
    son las más parecidas a los casos reales sin etiqueta. Las de origen SCIAN
    se excluyen porque el KNN las recuperaría trivialmente por el código, y
    las no transferibles (``exclude_from_training``) porque el KNN, por diseño,
    no debe predecirlas.

    Returns
    -------
    dict[str, float]
        ``accuracy`` y ``macro_f1`` sobre los ocultos; ``baseline_accuracy``
        (predecir siempre la clase más frecuente), que es la vara contra la que
        se juzga al KNN; y ``coverage`` (proporción que el KNN resolvió con
        confianza suficiente, sin caer en el default).
    """
    excluded = list(knn_kwargs.get("exclude_from_training", ()))
    candidates = df.index[df["rule_source"].eq("NAME") & ~df["cuisine_rule"].isin(excluded)]
    n_hold = max(1, round(len(candidates) * holdout_fraction))
    hidden = rng.choice(candidates, size=n_hold, replace=False)

    masked = df.copy()
    truth = masked.loc[hidden, "cuisine_rule"].copy()
    masked.loc[hidden, "cuisine_rule"] = None
    pred = impute_cuisine_knn(masked, **knn_kwargs).loc[hidden]

    majority = masked["cuisine_rule"].dropna().mode().iloc[0]
    return {
        "n_holdout": float(n_hold),
        "baseline_accuracy": float((truth == majority).mean()),
        "accuracy": float(accuracy_score(truth, pred["cuisine_pred"])),
        "macro_f1": float(f1_score(truth, pred["cuisine_pred"], average="macro")),
        "coverage": float(pred["cuisine_source"].eq("KNN").mean()),
    }
