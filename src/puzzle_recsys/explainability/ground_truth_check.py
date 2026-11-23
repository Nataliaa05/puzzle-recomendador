"""Contraste SHAP contra la verdad oculta: signo, orden de importancia y forma.

Como los datos son simulados, se sabe **exactamente** cómo eligió cada persona
(``simulation/choice.py``)::

    u = w_c·log(C·a_cocina) + w_p·log(P·a_proteína) + w_i·(gustos·ingredientes)
        - penalización·max(0, picante - tolerancia) + atractivo + calidad
        - λ·|log(precio / precio_ideal)|

Eso permite revisar si el modelo aprendió **por las razones correctas**. No se
comparan magnitudes (SHAP mide el modelo, no la utilidad, y las features son
aproximaciones observables de los gustos latentes). Se compara:

- **Signo:** ¿la feature empuja hacia donde debería?
- **Orden:** ¿las piezas que más varían entre candidatos en la verdad son las que
  más pesan en el modelo?
- **Forma:** p. ej. el picante solo debería castigar **por encima** de lo tolerado.

Usado en: notebooks 14.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def true_components(
    cands: pd.DataFrame,
    tastes: pd.DataFrame,
    items: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Piezas de la utilidad **verdadera** de cada candidato (sin el precio).

    ``tastes`` (por ``diner_id``) trae los vectores latentes; ``items`` (por
    ``menu_item_id``) trae índices de cocina y proteína (-1 = neutral), lista de
    índices de ingredientes, picante, atractivo y calidad.
    """
    t = tastes.reindex(cands["diner_id"])
    it = items.reindex(cands["menu_item_id"])
    cuisine_idx = it["cuisine_idx"].to_numpy(dtype=int)
    protein_idx = it["protein_idx"].to_numpy(dtype=int)

    def log_rel(vectors: pd.Series, idx: np.ndarray) -> np.ndarray:
        out = np.zeros(len(idx))
        for r, (vec, k) in enumerate(zip(vectors, idx, strict=True)):
            if k >= 0:
                out[r] = np.log(len(vec) * vec[k] + 1e-9)
        return out

    ingredients = np.array(
        [
            float(np.sum(np.asarray(pref)[list(ing)])) if len(ing) else 0.0
            for pref, ing in zip(t["ingredient_pref"], it["ingredients"], strict=True)
        ]
    )
    excess = np.maximum(0, it["spice"].to_numpy() - t["spice_tolerance"].to_numpy())
    out = cands[["query_id", "diner_id", "menu_item_id"]].copy()
    out["cocina"] = weights["affinity_weight"] * log_rel(t["cuisine_affinity"], cuisine_idx)
    out["proteina"] = weights["protein_weight"] * log_rel(t["protein_affinity"], protein_idx)
    out["ingredientes"] = weights["ingredient_weight"] * ingredients
    out["picante"] = -weights["spice_penalty"] * excess
    out["atractivo_calidad"] = (it["appeal"] + it["quality"]).to_numpy()
    return out


def within_query_spread(
    df: pd.DataFrame, columns: Sequence[str], *, group: str = "query_id"
) -> pd.Series:
    """Desviación estándar **dentro de cada consulta**, promediada entre consultas.

    En un ranking solo importa lo que distingue a los candidatos de una misma
    consulta: algo constante dentro de la consulta no puede cambiar el orden.
    """
    return df.groupby(group)[list(columns)].std(ddof=0).mean()


def observed_signs(
    shap_values: pd.DataFrame, features: pd.DataFrame, *, min_abs_rho: float = 0.1
) -> pd.DataFrame:
    """Dirección de cada feature: Spearman entre su valor y su valor SHAP.

    ``+`` si al subir la feature sube el puntaje, ``-`` si baja, ``0`` si la
    relación es débil (|rho| < ``min_abs_rho``) o la feature casi no se usa.
    """
    rows = []
    for col in shap_values.columns:
        x = pd.to_numeric(features[col], errors="coerce")
        s = shap_values[col]
        ok = x.notna() & s.notna()
        rho = float(stats.spearmanr(x[ok], s[ok])[0]) if ok.sum() > 2 and s[ok].std() > 0 else 0.0
        rho = 0.0 if np.isnan(rho) else rho
        sign = "+" if rho >= min_abs_rho else "-" if rho <= -min_abs_rho else "0"
        rows.append({"feature": col, "rho": rho, "signo": sign})
    return pd.DataFrame(rows)


PROTEINS = ("BEEF", "PORK", "CHICKEN", "FISH", "SEAFOOD", "PLANT", "EGG", "CHEESE", "MIXED")


def truth_items(
    menu: pd.DataFrame,
    canonical_cuisine: pd.Series,
    ingredients: pd.DataFrame,
    appeal: pd.Series,
    quality: pd.Series,
    *,
    cuisines: Sequence[str],
    ingredient_names: Sequence[str],
) -> pd.DataFrame:
    """Atributos **verdaderos** de cada platillo, con los mismos índices que el simulador.

    ``menu`` trae ``menu_item_id``, ``canonical_true``, ``protein`` y ``spice_level``;
    ``ingredients`` es ``menu_item_attributes`` filtrado a ``kind == 'ingredient'``.
    """
    cuisine_pos = {c: k for k, c in enumerate(cuisines)}
    protein_pos = {p: k for k, p in enumerate(PROTEINS)}
    ing_pos = {n: k for k, n in enumerate(ingredient_names)}
    ing = ingredients.groupby("menu_item_id")["value"].agg(lambda v: [ing_pos[x] for x in v])
    return pd.DataFrame(
        {
            "cuisine_idx": menu["canonical_true"]
            .map(canonical_cuisine)
            .map(cuisine_pos)
            .fillna(-1)
            .astype(int)
            .to_numpy(),
            "protein_idx": menu["protein"].map(protein_pos).fillna(-1).astype(int).to_numpy(),
            "ingredients": menu["menu_item_id"]
            .map(ing)
            .map(lambda v: v if isinstance(v, list) else [])
            .to_numpy(),
            "spice": menu["spice_level"].to_numpy(),
            "appeal": menu["canonical_true"].map(appeal).to_numpy(),
            "quality": menu["menu_item_id"].map(quality).to_numpy(),
        },
        index=menu["menu_item_id"].to_numpy(),
    )
