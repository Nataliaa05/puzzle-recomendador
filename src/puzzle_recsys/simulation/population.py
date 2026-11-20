"""Población de comensales registrados: dónde viven, cuánto gastan y qué les gusta.

- **Ubicación:** muestreo con probabilidad proporcional a la población de cada
  AGEB (Censo 2020) y un punto uniforme dentro del polígono.
- **Presupuesto por visita:** generación **condicional**. El percentil
  socioeconómico del comensal (escolaridad de su AGEB) se empata con el decil
  de ingreso de la ENIGH; dentro de esa ventana se hace un **bootstrap
  suavizado** ponderado por el factor de expansión (kernel lognormal), y el
  gasto trimestral por persona se divide entre las salidas por trimestre
  (supuesto S2).
- **Gustos latentes (la verdad oculta):** afinidad por cocina y proteína
  (Dirichlet), ingredientes que le gustan o evita, tolerancia al picante,
  sensibilidades a precio y distancia (lognormales), hábitos por tipo de plato
  y rasgo "compartidor" (Beta), dieta y alergias.
- **Onboarding:** versión ruidosa e incompleta de esa verdad. Nunca inventa
  alergias (reportar una alergia que no existe no es un error verosímil).

Usado en: notebooks 03.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import contains_xy

TASTE_COURSES: tuple[str, ...] = ("STARTER", "DRINK", "DESSERT", "SIDE")


# ---------------------------------------------------------------- ubicación


def sample_homes(ageb: gpd.GeoDataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Ubica ``n`` hogares: AGEB con probabilidad ∝ población, punto uniforme dentro.

    Parameters
    ----------
    ageb : gpd.GeoDataFrame
        AGEB con ``cvegeo`` y ``population`` en un CRS proyectado (metros).
    n : int
        Número de comensales.
    rng : np.random.Generator
        Generador aleatorio.

    Returns
    -------
    pd.DataFrame
        ``cvegeo``, ``lat`` y ``lng`` (WGS84).
    """
    weights = ageb["population"].to_numpy(dtype=float)
    chosen = rng.choice(len(ageb), size=n, p=weights / weights.sum())
    xs = np.empty(n)
    ys = np.empty(n)
    for idx in np.unique(chosen):
        slots = np.flatnonzero(chosen == idx)
        polygon = ageb.geometry.iloc[idx]
        minx, miny, maxx, maxy = polygon.bounds
        filled = 0
        while filled < len(slots):  # muestreo por rechazo dentro del polígono
            need = len(slots) - filled
            cand_x = rng.uniform(minx, maxx, need * 3)
            cand_y = rng.uniform(miny, maxy, need * 3)
            ok = contains_xy(polygon, cand_x, cand_y)
            take = min(int(ok.sum()), need)
            xs[slots[filled : filled + take]] = cand_x[ok][:take]
            ys[slots[filled : filled + take]] = cand_y[ok][:take]
            filled += take
    points = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=ageb.crs).to_crs("EPSG:4326")
    return pd.DataFrame(
        {
            "cvegeo": ageb["cvegeo"].to_numpy()[chosen],
            "lat": points.y.to_numpy(),
            "lng": points.x.to_numpy(),
        }
    )


# ---------------------------------------------------------------- presupuesto


def weighted_deciles(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Decil (1..10) de cada valor según la distribución **ponderada**."""
    order = np.argsort(values, kind="stable")
    cum = np.cumsum(np.asarray(weights, dtype=float)[order])
    cum /= cum[-1]
    deciles = np.empty(len(values), dtype=np.int64)
    deciles[order] = np.minimum(np.ceil(cum * 10 - 1e-12).astype(np.int64), 10)
    return np.maximum(deciles, 1)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


def sample_budgets(
    ses_percentile: np.ndarray,
    households: pd.DataFrame,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Presupuesto por visita y frecuencia relativa, condicionados al nivel socioeconómico.

    El gasto trimestral por persona ``s`` (ENIGH) se **descompone** en
    frecuencia x ticket con una elasticidad ``gamma``:

    - ``ticket = (s_med / salidas) · (s / s_med)^gamma``
    - ``frecuencia = (s / s_med)^(1-gamma)``

    de modo que ``ticket · salidas · frecuencia = s`` exactamente. Quien gasta
    más sale más seguido **y** gasta algo más por visita (con ``gamma`` < 1), en vez
    de gastar todo el excedente en una sola salida.

    Parameters
    ----------
    ses_percentile : np.ndarray
        Percentil (0, 1) del nivel socioeconómico de cada comensal.
    households : pd.DataFrame
        Hogares ENIGH con ``weight``, ``income_pc`` y ``food_away_pc``.
    cfg : Mapping
        Bloque ``population.budget`` (``outings_per_quarter``, ``smoothing_sigma``,
        ``decile_window``, ``spend_per_visit_elasticity``).
    rng : np.random.Generator
        Generador aleatorio.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Presupuesto por visita (pesos) y factor de frecuencia (1 = típico).
    """
    eats_out = households[households["food_away_pc"] > 0]  # supuesto S11: el usuario come fuera
    deciles = weighted_deciles(eats_out["income_pc"].to_numpy(), eats_out["weight"].to_numpy())
    spend = eats_out["food_away_pc"].to_numpy()
    weight = eats_out["weight"].to_numpy(dtype=float)
    median = _weighted_median(spend, weight)

    target = np.clip(np.ceil(np.asarray(ses_percentile) * 10).astype(int), 1, 10)
    sampled = np.empty(len(target))
    window = cfg["decile_window"]
    for d in np.unique(target):
        slots = np.flatnonzero(target == d)
        pool = np.flatnonzero(np.abs(deciles - d) <= window)
        p = weight[pool] / weight[pool].sum()
        picks = rng.choice(pool, size=len(slots), p=p)  # bootstrap ponderado por el factor
        sampled[slots] = spend[picks] * rng.lognormal(0.0, cfg["smoothing_sigma"], len(slots))

    gamma = cfg["spend_per_visit_elasticity"]
    relative = sampled / median
    ticket = median / cfg["outings_per_quarter"] * relative**gamma
    return ticket, relative ** (1 - gamma)


# ---------------------------------------------------------------- gustos


@dataclass(frozen=True)
class DinerTastes:
    """Gustos latentes de la población (fila i = comensal i). Es la verdad oculta."""

    cuisine_names: tuple[str, ...]
    protein_names: tuple[str, ...]
    ingredient_names: tuple[str, ...]
    allergen_names: tuple[str, ...]
    cuisine_affinity: np.ndarray  # (n, C), filas suman 1
    protein_affinity: np.ndarray  # (n, P), filas suman 1
    ingredient_pref: np.ndarray  # (n, I) en {-1, 0, +1}
    spice_tolerance: np.ndarray  # (n,) 0..3
    price_sensitivity: np.ndarray  # (n,) λ
    distance_sensitivity: np.ndarray  # (n,) β por km
    variety_seeking: np.ndarray  # (n,)
    sharer: np.ndarray  # (n,)
    course_habit: np.ndarray  # (n, len(TASTE_COURSES))
    diet: np.ndarray  # (n,) "ninguna" | "vegetariano" | "vegano"
    allergies: np.ndarray  # (n, A) bool

    def __len__(self) -> int:
        """Número de comensales."""
        return len(self.spice_tolerance)


def sample_tastes(
    n: int,
    *,
    cuisines: Sequence[str],
    cuisine_share: np.ndarray,
    proteins: Sequence[str],
    ingredients: Sequence[str],
    allergens: Sequence[str],
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
) -> DinerTastes:
    """Sortea los gustos latentes de ``n`` comensales.

    La afinidad por cocina es Dirichlet centrada en la **oferta real** de la
    ciudad (``cuisine_share``): en promedio, la gente come lo que hay.

    Parameters
    ----------
    n : int
        Número de comensales.
    cuisines, proteins, ingredients, allergens : Sequence[str]
        Catálogos (definen el orden de las columnas).
    cuisine_share : np.ndarray
        Participación de cada cocina en la oferta (suma 1).
    cfg : Mapping
        Bloque ``population`` de la configuración.
    rng : np.random.Generator
        Generador aleatorio.
    """
    taste = cfg["taste"]
    share = np.asarray(cuisine_share, dtype=float)
    alpha_c = taste["cuisine_concentration"] * (share / share.sum()) + 0.02
    alpha_p = np.full(len(proteins), taste["protein_concentration"] / len(proteins))
    u = rng.random((n, len(ingredients)))
    ingredient_pref = np.where(
        u < taste["ingredient_like_rate"],
        1,
        np.where(u > 1 - taste["ingredient_dislike_rate"], -1, 0),
    )
    lam = taste["price_sensitivity_lognormal"]
    beta = taste["distance_sensitivity_lognormal"]
    habits = np.column_stack(
        [
            rng.beta(taste["course_habit_beta"][c]["a"], taste["course_habit_beta"][c]["b"], n)
            for c in TASTE_COURSES
        ]
    )
    diets = cfg["diets"]
    d = rng.random(n)
    diet = np.where(
        d < diets["vegano"],
        "vegano",
        np.where(d < diets["vegano"] + diets["vegetariano"], "vegetariano", "ninguna"),
    )
    prevalence = np.array([cfg["allergy_prevalence"][a] for a in allergens])
    return DinerTastes(
        cuisine_names=tuple(cuisines),
        protein_names=tuple(proteins),
        ingredient_names=tuple(ingredients),
        allergen_names=tuple(allergens),
        cuisine_affinity=rng.dirichlet(alpha_c, n),
        protein_affinity=rng.dirichlet(alpha_p, n),
        ingredient_pref=ingredient_pref.astype(np.int8),
        spice_tolerance=rng.choice(4, size=n, p=taste["spice_tolerance_probs"]),
        price_sensitivity=rng.lognormal(lam["mu"], lam["sigma"], n),
        distance_sensitivity=rng.lognormal(beta["mu"], beta["sigma"], n),
        variety_seeking=rng.beta(
            taste["variety_seeking_beta"]["a"], taste["variety_seeking_beta"]["b"], n
        ),
        sharer=rng.beta(taste["sharer_trait_beta"]["a"], taste["sharer_trait_beta"]["b"], n),
        course_habit=habits,
        diet=diet,
        allergies=rng.random((n, len(allergens))) < prevalence,
    )


def drift_cuisine_affinity(
    affinity: np.ndarray, alpha: np.ndarray, blend: float, rng: np.random.Generator
) -> np.ndarray:
    """Deriva mensual del gusto: mezcla con un nuevo sorteo Dirichlet (sigue sumando 1)."""
    fresh = rng.dirichlet(alpha, affinity.shape[0])
    return (1 - blend) * affinity + blend * fresh


def onboarding_answers(
    tastes: DinerTastes, cfg: Mapping[str, Any], rng: np.random.Generator
) -> pd.DataFrame:
    """Respuestas del onboarding: ruidosas, incompletas y sin alergias inventadas.

    Returns
    -------
    pd.DataFrame
        ``completed``, ``favorite_cuisines`` (3, con ruido), ``declared_spice``
        y ``declared_allergies`` (subconjunto de las verdaderas).
    """
    n = len(tastes)
    completed = rng.random(n) < cfg["completion_rate"]
    flip = cfg["answer_flip_rate"]
    names = np.array(tastes.cuisine_names)
    top3 = np.argsort(-tastes.cuisine_affinity, axis=1)[:, :3]
    favorites, spices, allergies = [], [], []
    for i in range(n):
        if not completed[i]:
            favorites.append([])
            spices.append(np.nan)
            allergies.append([])
            continue
        picks = [
            names[c] if rng.random() >= flip else names[rng.integers(len(names))] for c in top3[i]
        ]
        favorites.append(list(dict.fromkeys(picks)))
        spice = tastes.spice_tolerance[i]
        if rng.random() < flip:
            spice = int(np.clip(spice + rng.choice([-1, 1]), 0, 3))
        spices.append(spice)
        true = [tastes.allergen_names[j] for j in np.flatnonzero(tastes.allergies[i])]
        allergies.append([a for a in true if rng.random() < cfg["allergies_reported_rate"]])
    return pd.DataFrame(
        {
            "completed": completed,
            "favorite_cuisines": favorites,
            "declared_spice": pd.array(spices, dtype="Float64"),
            "declared_allergies": allergies,
        }
    )
