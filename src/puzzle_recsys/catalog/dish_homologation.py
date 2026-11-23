"""Homologación de platillos: nombre del menú → platillo canónico.

Cada restaurante escribe sus platillos como quiere ("TACOS DE PASTOR especial",
"Al pastor (orden)"). El modelo necesita el **canónico** (``tacos_pastor``),
porque es lo que se transfiere entre restaurantes.

Algoritmo (vecino más cercano, k = 1):

1. **Limpiar el nombre:** minúsculas, sin acentos ni signos, y sin palabras de
   relleno de menú ("especial", "de la casa").
2. **n-gramas de caracteres** (TF-IDF) contra la referencia: nombres canónicos
   y alias **conocidos**. Los n-gramas toleran errores de dedo ("Limonda").
3. Solo se comparan platillos del **mismo tipo** (el tipo de plato lo da la carta).
4. **Pista de la cocina:** se suma ``cuisine_weight`` a los canónicos de la misma
   cocina que el restaurante ("Ranchera" en una pizzería es una pizza).

La **confianza** es la similitud coseno del nombre con el vecino elegido.

En producción el canónico se **sugiere** y el restaurante lo confirma o corrige
(``canonical_source``); :func:`simulate_confirmation` reproduce ese paso.

Usado en: notebooks 09.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from puzzle_recsys.catalog.cuisine_imputation import normalize_name


def clean_dish_name(name: str, stop_words: Sequence[str]) -> str:
    """Normaliza el nombre y quita el relleno de menú; si no queda nada, conserva el original."""
    base = normalize_name(name)
    if not stop_words:
        return base
    pattern = r"\b(" + "|".join(re.escape(normalize_name(w)) for w in stop_words) + r")\b"
    cleaned = re.sub(r"\s+", " ", re.sub(pattern, " ", base)).strip()
    return cleaned or base


def split_aliases(
    aliases: pd.DataFrame, *, known_share: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parte los alias en **conocidos** (referencia del KNN) y **no vistos**.

    En la simulación los nombres de los menús salieron de la misma lista de
    alias; si el KNN la conociera completa, la exactitud sería casi perfecta y
    engañosa. Los alias no vistos imitan los nombres nuevos que escribe un
    restaurante real. Los nombres canónicos siempre son conocidos (no van aquí).
    """
    known = rng.random(len(aliases)) < known_share
    return aliases[known].reset_index(drop=True), aliases[~known].reset_index(drop=True)


class DishHomologator:
    """KNN (k = 1) sobre n-gramas de caracteres del nombre, por tipo de plato."""

    def __init__(
        self,
        *,
        stop_words: Sequence[str] = (),
        cuisine_weight: float = 0.0,
        ngram_range: tuple[int, int] = (2, 4),
    ) -> None:
        self.stop_words = tuple(stop_words)
        self.cuisine_weight = cuisine_weight
        self.ngram_range = ngram_range

    def fit(self, reference: pd.DataFrame) -> DishHomologator:
        """Indexa la referencia: ``canonical_id``, ``name``, ``course``, ``cuisine``."""
        ref = reference.assign(
            clean=reference["name"].map(lambda n: clean_dish_name(n, self.stop_words))
        ).drop_duplicates(["canonical_id", "clean"])
        self.vectorizer_ = TfidfVectorizer(
            analyzer="char_wb", ngram_range=self.ngram_range, sublinear_tf=True
        ).fit(ref["clean"])
        self.ref_matrix_ = self.vectorizer_.transform(ref["clean"])
        self.ref_ids_ = ref["canonical_id"].to_numpy()
        self.ref_course_ = ref["course"].to_numpy()
        self.ref_cuisine_ = ref["cuisine"].to_numpy()
        return self

    def predict(self, items: pd.DataFrame, *, batch: int = 4_000) -> pd.DataFrame:
        """``items``: ``name``, ``course``, ``cuisine`` (la del restaurante).

        Returns
        -------
        pd.DataFrame
            ``canonical_pred`` y ``confianza`` (similitud del nombre, 0 a 1),
            con el mismo índice que ``items``.
        """
        clean = items["name"].map(lambda n: clean_dish_name(n, self.stop_words))
        courses = items["course"].to_numpy()
        cuisines = items["cuisine"].to_numpy()
        preds, confs = [], []
        for start in range(0, len(items), batch):
            sl = slice(start, start + batch)
            sim = (self.vectorizer_.transform(clean.iloc[sl]) @ self.ref_matrix_.T).toarray()
            same_course = courses[sl, None] == self.ref_course_[None, :]
            bonus = self.cuisine_weight * (cuisines[sl, None] == self.ref_cuisine_[None, :])
            score = np.where(same_course, sim + bonus, -np.inf)
            best = score.argmax(axis=1)
            preds.append(self.ref_ids_[best])
            confs.append(sim[np.arange(len(best)), best])
        return pd.DataFrame(
            {"canonical_pred": np.concatenate(preds), "confianza": np.concatenate(confs)},
            index=items.index,
        )


def simulate_confirmation(
    pred: pd.Series,
    truth: pd.Series,
    restaurant_ids: pd.Series,
    *,
    rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simula la revisión del restaurante: una fracción ``rate`` revisa **todo** su menú.

    Si revisa, el canónico queda correcto: ``CONFIRMED`` si el KNN acertó y
    ``MANUAL`` si lo corrigió. Si no revisa, queda la sugerencia (``SUGGESTED``).
    Es el único paso que usa la verdad, porque simula una acción humana.
    """
    restaurants = pd.unique(restaurant_ids)
    reviewed = dict(zip(restaurants, rng.random(len(restaurants)) < rate, strict=True))
    did = restaurant_ids.map(reviewed).to_numpy(dtype=bool)
    hit = pred.to_numpy() == truth.to_numpy()
    source = np.where(~did, "SUGGESTED", np.where(hit, "CONFIRMED", "MANUAL"))
    final = np.where(did, truth.to_numpy(), pred.to_numpy())
    return pd.DataFrame({"canonical_pred": final, "canonical_source": source}, index=pred.index)


def accuracy_by_group(pred: pd.Series, truth: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Exactitud por grupo (p. ej. tipo de nombre) y total."""
    hit = pd.Series(pred.to_numpy() == truth.to_numpy(), index=groups.index)
    by = hit.groupby(groups.to_numpy())
    out = pd.DataFrame({"n": by.size(), "exactitud": by.mean()})
    out.loc["total"] = [len(hit), hit.mean()]
    out["n"] = out["n"].astype(int)
    return out.rename_axis("grupo").reset_index()


def uses_unseen_alias(names: pd.Series, unseen: Sequence[str]) -> pd.Series:
    """Marca los nombres que contienen (como palabras completas) algún alias no visto."""
    words = sorted({normalize_name(a) for a in unseen if normalize_name(a)}, key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b")
    return names.map(lambda n: bool(pattern.search(normalize_name(n))))


def threshold_table(
    confidence: pd.Series, hit: pd.Series, thresholds: Sequence[float]
) -> pd.DataFrame:
    """Cobertura y exactitud si solo se aceptan sugerencias con confianza ≥ umbral.

    Sirve para decidir cuándo **pedirle al restaurante** que confirme: debajo
    del umbral la sugerencia es poco fiable.
    """
    conf = confidence.to_numpy()
    ok = hit.to_numpy()
    rows = []
    for t in thresholds:
        keep = conf >= t
        rows.append(
            {
                "umbral": t,
                "cobertura": float(keep.mean()),
                "exactitud": float(ok[keep].mean()) if keep.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)
