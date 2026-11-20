"""Efecto del contexto (franja de comida y fin de semana) sobre las decisiones.

En la realidad no se come igual a las 9 que a las 21: en el desayuno se va a una
cafetería y se pide café y chilaquiles; en la cena gana la taquería y aparece el
alcohol; el fin de semana las mesas son más grandes. Sin esto, las features de
hora no tendrían ninguna señal y el modelo no aprendería a usar el contexto.

Cuatro mecanismos, todos con valores en ``config/simulation.yaml`` (supuesto S16):

1. **Cocina por franja:** multiplica el atractivo de cada cocina al elegir lugar.
2. **Tipo de plato por franja:** multiplica la probabilidad de pedir cada tipo.
3. **Platillo por franja:** bono a platillos de desayuno y a bebidas con alcohol.
4. **Tamaño de mesa:** más mesas grandes en fin de semana y menos en el desayuno.

Usado en: notebooks 03, 04.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from puzzle_recsys.features.calendar import MEAL_PERIODS, WEEKEND_DAYS, period_index

_MAX_RATE = 0.98


def cuisine_period_log(cuisines: Sequence[str], cfg: Mapping[str, Any]) -> np.ndarray:
    """Matriz (franja x cocina) de ``weight · log(multiplicador)``; 0 si no se declaró."""
    out = np.zeros((len(MEAL_PERIODS), len(cuisines)))
    for k, cuisine in enumerate(cuisines):
        mult = cfg["cuisine_period"].get(cuisine)
        if mult is not None:
            out[:, k] = cfg["weight"] * np.log(np.asarray(mult, dtype=float))
    return out


def course_rate(base: float, course: str, period: int, cfg: Mapping[str, Any]) -> float:
    """Probabilidad de pedir ``course`` en esa franja (base x multiplicador, tope 0.98)."""
    mult = cfg["course_rate_period"].get(MEAL_PERIODS[period], {}).get(course, 1.0)
    return min(_MAX_RATE, base * mult)


def dish_period_bonus(
    canonical_ids: Sequence[str], library: Any, cfg: Mapping[str, Any]
) -> np.ndarray:
    """Matriz (platillo x franja) con el bono de utilidad por franja.

    Platillos de desayuno: lista explícita en la configuración. Bebidas con
    alcohol: se detectan por **ingrediente**, así una bebida nueva con cerveza
    o mezcal queda cubierta sin tocar la lista.
    """
    breakfast = np.array([cfg["breakfast_bonus"][p] for p in MEAL_PERIODS], dtype=float)
    alcohol = np.array([cfg["alcohol_bonus"][p] for p in MEAL_PERIODS], dtype=float)
    alcohol_ing = set(cfg["alcohol_ingredients"])
    breakfast_set = set(cfg["breakfast_dishes"])
    per_dish: dict[str, np.ndarray] = {}
    for dish in library.dishes:
        bonus = np.zeros(len(MEAL_PERIODS))
        if dish.slug in breakfast_set:
            bonus += breakfast
        if alcohol_ing & set(dish.ingredients):
            bonus += alcohol
        per_dish[dish.slug] = bonus
    zero = np.zeros(len(MEAL_PERIODS))
    return np.vstack([per_dish.get(c, zero) for c in canonical_ids])


def pax_probabilities(
    values: np.ndarray,
    probs: np.ndarray,
    *,
    period: int,
    weekend: bool,
    cfg: Mapping[str, Any],
) -> np.ndarray:
    """Distribución del tamaño de mesa según contexto (renormalizada)."""
    w = np.asarray(probs, dtype=float).copy()
    large = np.asarray(values) >= cfg["large_from"]
    if weekend:
        w[large] *= cfg["weekend_factor"]
    if period == 0:
        w[large] *= cfg["breakfast_factor"]
    return w / w.sum()


def expected_pax_distribution(
    values: np.ndarray,
    probs: np.ndarray,
    weekday_weights: Sequence[float],
    hourly_profile: Mapping[str, float],
    cfg: Mapping[str, Any],
) -> np.ndarray:
    """Distribución marginal esperada del tamaño de mesa (mezcla sobre días y horas).

    Es contra lo que se valida la simulación: ya no es ``pax_probs`` tal cual,
    porque el contexto la deforma a propósito.
    """
    days = np.asarray(weekday_weights, dtype=float) / np.sum(weekday_weights)
    hours = np.array([int(h) for h in hourly_profile])
    hour_w = np.array(list(hourly_profile.values()), dtype=float)
    hour_w /= hour_w.sum()
    periods = period_index(hours)
    out = np.zeros(len(values))
    for day, p_day in enumerate(days):
        for period, p_hour in zip(periods, hour_w, strict=True):
            out += (
                p_day
                * p_hour
                * pax_probabilities(
                    values, probs, period=int(period), weekend=day in WEEKEND_DAYS, cfg=cfg
                )
            )
    return out
