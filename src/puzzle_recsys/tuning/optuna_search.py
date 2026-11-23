"""Optimización bayesiana de hiperparámetros con Optuna (TPE).

**Por qué bayesiana y no búsqueda en malla:** cada entrenamiento cuesta minutos.
Una malla de 5 hiperparámetros con 5 valores cada uno serían 3,125 corridas. TPE
(Tree-structured Parzen Estimator) usa los resultados anteriores para proponer
la siguiente combinación: separa los intentos en "buenos" y "malos", modela la
densidad de cada grupo y prueba donde la razón buenos/malos es mayor. Con unas
decenas de intentos suele llegar cerca del óptimo.

El espacio de búsqueda vive en ``config/models.yaml``: ``low``, ``high`` y, si la
escala es multiplicativa (tasa de aprendizaje, número de hojas), ``log: true``.

Usado en: notebooks 12.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import optuna


def suggest(trial: optuna.Trial, space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Propone un valor por hiperparámetro; si los límites son enteros, el valor también."""
    params = {}
    for name, spec in space.items():
        log = bool(spec.get("log", False))
        if isinstance(spec["low"], int) and isinstance(spec["high"], int):
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=log)
        else:
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=log
            )
    return params


def run_study(
    objective: Callable[[dict[str, Any]], float],
    space: Mapping[str, Mapping[str, Any]],
    *,
    n_trials: int,
    seed: int,
) -> optuna.Study:
    """Maximiza ``objective(params)`` con TPE reproducible (misma semilla, mismos intentos)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda trial: objective(suggest(trial, space)), n_trials=n_trials)
    return study
