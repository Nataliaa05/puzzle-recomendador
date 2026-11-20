"""Calibración de la simulación por hipercubo latino.

Problema: los parámetros de la simulación (escala del ruido Gumbel, peso de
la afinidad, sensibilidad al precio) no se pueden estimar con datos reales,
que no existen todavía. Si el ruido es muy grande, las elecciones son casi
aleatorias y ningún modelo aprende; si es muy chico, el problema es trivial.

Solución: explorar el espacio de parámetros con un **hipercubo latino** (LHS):
cada parámetro se divide en ``n`` franjas iguales y cada franja se usa
exactamente una vez. Con pocas corridas se cubre todo el rango (mejor que una
grilla, que crece exponencialmente, o que el azar puro, que deja huecos).

Usado en: notebooks 04.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc


def latin_hypercube(
    bounds: Mapping[str, tuple[float, float]], *, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Diseño LHS de ``n`` puntos dentro de ``bounds``.

    Parameters
    ----------
    bounds : Mapping[str, tuple[float, float]]
        ``{"choice.gumbel_scale": (0.5, 2.0), ...}``.
    n : int
        Número de corridas.
    rng : np.random.Generator
        Generador aleatorio (reproducibilidad).

    Returns
    -------
    pd.DataFrame
        Una fila por corrida y una columna por parámetro.
    """
    names = list(bounds)
    sampler = qmc.LatinHypercube(d=len(names), rng=rng)
    unit = sampler.random(n)
    lo = np.array([bounds[k][0] for k in names])
    hi = np.array([bounds[k][1] for k in names])
    return pd.DataFrame(qmc.scale(unit, lo, hi), columns=names)


def with_params(cfg: Mapping[str, Any], params: Mapping[str, float]) -> dict[str, Any]:
    """Copia de ``cfg`` con parámetros reemplazados por ruta (``"choice.gumbel_scale"``).

    Falla con ``KeyError`` si la ruta no existe: un error de dedo en el nombre de
    un parámetro no debe pasar en silencio (calibraría algo que no existe).
    """
    new = copy.deepcopy(dict(cfg))
    for path, value in params.items():
        *parents, leaf = path.split(".")
        node = new
        for key in parents:
            node = node[key]
        if leaf not in node:
            raise KeyError(f"Parámetro inexistente: {path}")
        node[leaf] = float(value)
    return new


def run_scenario(
    world: Mapping[str, Any],
    cfg: Mapping[str, Any],
    calendar: Mapping[str, Any],
    *,
    n_diners: int,
    months: int,
    seed: int,
    price_sensitivity_scale: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Corre una simulación reducida sobre la MISMA población del notebook 03.

    Parameters
    ----------
    world : Mapping
        ``index``, ``tastes``, ``homes``, ``budgets``, ``frequency`` y ``anonymous``
        (gustos y presupuestos de los acompañantes no registrados).
    cfg : Mapping
        Bloques de configuración que usa el motor (``visits``, ``choice``...).
    calendar : Mapping
        ``start_date`` y ``timezone``.
    n_diners, months : int
        Tamaño del escenario (los primeros ``n_diners`` comensales).
    seed : int
        Semilla de esta corrida (réplicas Monte Carlo = semillas distintas).
    price_sensitivity_scale : float, optional
        Multiplica la sensibilidad al precio (λ) de cada comensal.
    """
    from dataclasses import replace

    from puzzle_recsys.simulation.engine import simulate_visits

    tastes = world["tastes"]
    keep = np.arange(min(n_diners, len(tastes)))
    subset = replace(
        tastes,
        **{
            field: getattr(tastes, field)[keep]
            for field in (
                "cuisine_affinity",
                "protein_affinity",
                "ingredient_pref",
                "spice_tolerance",
                "price_sensitivity",
                "distance_sensitivity",
                "variety_seeking",
                "sharer",
                "course_habit",
                "diet",
                "allergies",
            )
        },
    )
    subset = replace(subset, price_sensitivity=subset.price_sensitivity * price_sensitivity_scale)
    return simulate_visits(
        world["index"],
        subset,
        world["homes"].iloc[keep].reset_index(drop=True),
        world["budgets"][keep],
        world["anonymous"],
        cfg,
        calendar,
        months=months,
        rng=np.random.default_rng(seed),
        frequency=world["frequency"][keep],
    )
