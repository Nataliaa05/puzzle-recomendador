"""Modelos de elección discreta de la simulación (logit mixto con ruido Gumbel).

Cada comensal elige maximizando ``utilidad determinística + ε``, con
``ε ~ Gumbel(0, escala)``. Eso es exactamente un **logit**: la probabilidad de
elegir una opción es ``softmax(utilidad / escala)``. Como los coeficientes
(sensibilidad a precio y distancia, gustos) son **propios de cada comensal**,
el modelo es un **logit mixto**, que relaja la independencia de alternativas
irrelevantes del logit simple.

Reglas que NO son pesos sino **filtros duros** (ADR 0001):

- Un platillo con un alérgeno del comensal nunca es elegible.
- Un vegetariano nunca recibe carne; un vegano, nada de origen animal.
- Para compartir: ningún alérgeno de NADIE en la mesa.

Usado en: notebooks 03, 15 (oráculo).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ItemFeatures:
    """Atributos de los platillos servidos, en arreglos alineados por índice."""

    cuisine_idx: np.ndarray  # -1 = pool general (neutral)
    protein_idx: np.ndarray  # -1 = sin proteína (neutral)
    ingredients: np.ndarray  # (N, I) multi-hot int8 (densa: ~110 columnas)
    spice: np.ndarray
    course: np.ndarray  # str
    shareable: np.ndarray  # bool
    allergens: np.ndarray  # (N, A) bool
    vegetarian: np.ndarray  # bool
    vegan: np.ndarray  # bool
    appeal: np.ndarray  # atractivo latente del canónico (cola larga)
    quality: np.ndarray  # calidad de ejecución del restaurante (Laplace)


@dataclass(frozen=True)
class DinerVector:
    """Gustos de UN comensal (una fila de ``DinerTastes``) más su presupuesto."""

    cuisine_affinity: np.ndarray
    protein_affinity: np.ndarray
    ingredient_pref: np.ndarray
    spice_tolerance: int
    price_sensitivity: float
    budget: float
    diet: str
    allergies: np.ndarray  # bool (A,)


def _log_relative(affinity: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``log(K · a_k)``: 0 = gusto promedio; índices -1 son neutrales (0)."""
    k = len(affinity)
    out = np.zeros(len(idx))
    known = idx >= 0
    out[known] = np.log(k * affinity[idx[known]] + 1e-9)
    return out


def feasible(items: ItemFeatures, cand: np.ndarray, diner: DinerVector) -> np.ndarray:
    """Máscara de platillos elegibles: sin alérgenos del comensal y compatibles con su dieta."""
    ok = ~(items.allergens[cand] & diner.allergies).any(axis=1)
    if diner.diet == "vegano":
        ok &= items.vegan[cand]
    elif diner.diet == "vegetariano":
        ok &= items.vegetarian[cand]
    return ok


def dish_utility(
    items: ItemFeatures,
    cand: np.ndarray,
    prices: np.ndarray,
    ideal_prices: np.ndarray,
    diner: DinerVector,
    cfg: Mapping[str, Any],
) -> np.ndarray:
    """Utilidad determinística de cada platillo candidato para un comensal.

    ``u = w_c·log(C·a_cocina) + w_p·log(P·a_proteína) + w_i·(gustos·ingredientes)
    - penalización·max(0, picante - tolerancia) + atractivo + calidad
    - λ·|log(precio / precio_ideal)|``

    El precio entra como **distancia a un precio ideal** (no como "más barato
    es mejor"). Con ``-λ·log(precio/presupuesto)`` el presupuesto sería una
    constante igual para todas las opciones y **se cancelaría en el logit**: no
    influiría en qué se elige. El precio ideal lo calcula :func:`ideal_prices`.

    Parameters
    ----------
    items : ItemFeatures
        Atributos de todos los platillos.
    cand : np.ndarray
        Índices de los candidatos.
    prices : np.ndarray
        Precio vigente de cada candidato (pesos).
    ideal_prices : np.ndarray
        Precio ideal del comensal para cada candidato (pesos).
    diner : DinerVector
        Gustos del comensal.
    cfg : Mapping
        Bloque ``choice`` de la configuración.
    """
    u = cfg["affinity_weight"] * _log_relative(diner.cuisine_affinity, items.cuisine_idx[cand])
    u += cfg["protein_weight"] * _log_relative(diner.protein_affinity, items.protein_idx[cand])
    u += cfg["ingredient_weight"] * (items.ingredients[cand] @ diner.ingredient_pref)
    u -= cfg["spice_penalty"] * np.maximum(0, items.spice[cand] - diner.spice_tolerance)
    u += items.appeal[cand] + items.quality[cand]
    u -= diner.price_sensitivity * np.abs(np.log(prices / ideal_prices))
    return u


def ideal_prices(
    menu_typical: np.ndarray,
    budget: float,
    course_reference: np.ndarray,
    budget_weight: float,
) -> np.ndarray:
    """Precio ideal: mezcla geométrica entre lo típico de ESE menú y el presupuesto.

    ``ideal = típico_del_menú^(1 - w) · (presupuesto · referencia_del_tipo)^w``

    Es **anclaje de precios**: en un restaurante caro (p. ej., un aniversario)
    el comensal se ajusta a la carta y pide algo "normal para ese lugar"; en
    uno barato, lo mismo. Su presupuesto mueve el ideal hacia arriba o abajo
    con peso ``w``.

    Parameters
    ----------
    menu_typical : np.ndarray
        Precio mediano del tipo de plato de cada candidato en ese menú, a esa hora.
    budget : float
        Presupuesto de la visita (ya multiplicado si es una ocasión especial).
    course_reference : np.ndarray
        Fracción típica del presupuesto que se va en ese tipo de plato.
    budget_weight : float
        ``w`` en [0, 1]: 0 = solo la carta; 1 = solo el presupuesto.
    """
    return menu_typical ** (1 - budget_weight) * (budget * course_reference) ** budget_weight


def gumbel_choice(
    utility: np.ndarray,
    scale: float,
    rng: np.random.Generator,
    position_bonus: np.ndarray | None = None,
) -> int:
    """Índice elegido: ``argmax(u + bonus_posición + Gumbel(0, escala))``."""
    shock = rng.gumbel(0.0, scale, len(utility))
    total = utility + shock if position_bonus is None else utility + position_bonus + shock
    return int(np.argmax(total))


def group_utility(member_utilities: np.ndarray, least_misery_weight: float) -> np.ndarray:
    """Utilidad de la mesa: mezcla de promedio (gusto general) y mínimo (miseria mínima).

    Parameters
    ----------
    member_utilities : np.ndarray
        Matriz (miembros x platillos).
    least_misery_weight : float
        0 = solo promedio; 1 = solo el que menos lo disfruta.
    """
    mean = member_utilities.mean(axis=0)
    worst = member_utilities.min(axis=0)
    return (1 - least_misery_weight) * mean + least_misery_weight * worst


def restaurant_utility(
    cuisine_idx: np.ndarray,
    distance_km: np.ndarray,
    price_level: np.ndarray,
    previous_visits: np.ndarray,
    diner: DinerVector,
    distance_sensitivity: float,
    variety_seeking: float,
    cfg: Mapping[str, Any],
) -> np.ndarray:
    """Utilidad de ir a cada restaurante candidato (solo para simular visitas).

    ``u = w_c·log(C·a_cocina) - β·distancia - λ·|log(precio/presupuesto)|
    + lealtad·log(1 + visitas) + variedad·[nunca visitado]``
    """
    u = cfg["affinity_weight"] * _log_relative(diner.cuisine_affinity, cuisine_idx)
    u -= distance_sensitivity * distance_km
    u -= diner.price_sensitivity * np.abs(np.log(price_level / diner.budget))
    u += cfg["loyalty_weight"] * np.log1p(previous_visits)
    u += variety_seeking * (previous_visits == 0)
    return u


def shared_group_utility(
    member_utilities: np.ndarray,
    eligible: np.ndarray,
    *,
    least_misery_weight: float,
    exclusion_penalty: float,
) -> np.ndarray:
    """Utilidad de compartir cada platillo, contando SOLO a quienes pueden comerlo.

    Un platillo para compartir no se descarta para toda la mesa porque una
    persona sea alérgica: se reparte entre quienes sí pueden comerlo y esa
    persona no come de ese plato. La decisión del grupo se toma con esas
    personas (promedio + miseria mínima) y se resta un **costo social** por
    dejar fuera a alguien.

    Parameters
    ----------
    member_utilities : np.ndarray
        Matriz (miembros x platillos) de utilidades individuales.
    eligible : np.ndarray
        Matriz booleana (miembros x platillos): ``True`` si esa persona puede
        comer ese platillo (sin sus alérgenos y compatible con su dieta).
    least_misery_weight : float
        0 = solo promedio; 1 = solo quien menos lo disfruta (entre los elegibles).
    exclusion_penalty : float
        Costo por la fracción de la mesa que queda fuera.

    Returns
    -------
    np.ndarray
        Utilidad por platillo; ``-inf`` si menos de 2 personas pueden comerlo.
    """
    n_eligible = eligible.sum(axis=0)
    masked = np.where(eligible, member_utilities, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(masked, axis=0)
        worst = np.nanmin(masked, axis=0)
    excluded = 1.0 - n_eligible / member_utilities.shape[0]
    utility = (1 - least_misery_weight) * mean + least_misery_weight * worst
    utility -= exclusion_penalty * excluded
    return np.where(n_eligible >= 2, utility, -np.inf)
