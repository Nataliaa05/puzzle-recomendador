"""Análisis bivariado no paramétrico: Spearman, Kruskal-Wallis, chi².

Con millones de filas **todo** sale significativo (p ≈ 0): el p-valor solo dice
que el efecto no es exactamente cero. Por eso cada prueba devuelve además un
**tamaño del efecto** (rho, ε², V de Cramér), que es lo que se interpreta.

Se usan pruebas no paramétricas porque ninguna variable es normal (notebook 06).

Usado en: notebooks 07.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Umbrales convencionales (Cohen 1988; Tomczak y Tomczak 2014 para ε²).
_THRESHOLDS = {
    "rho": (0.1, 0.3, 0.5),
    "V": (0.1, 0.3, 0.5),
    "epsilon2": (0.01, 0.06, 0.14),
}
_LABELS = ("despreciable", "pequeño", "moderado", "fuerte")


def effect_label(value: float, kind: str) -> str:
    """Traduce un tamaño de efecto a palabras (``kind``: ``rho``, ``V`` o ``epsilon2``)."""
    small, medium, large = _THRESHOLDS[kind]
    v = abs(value)
    if v < small:
        return _LABELS[0]
    if v < medium:
        return _LABELS[1]
    if v < large:
        return _LABELS[2]
    return _LABELS[3]


def spearman_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Correlación de Spearman para cada par de columnas (sin repetir).

    Spearman mide relaciones **monótonas** (no solo lineales) y es robusta a
    colas largas, porque trabaja con rangos.

    Returns
    -------
    pd.DataFrame
        ``a``, ``b``, ``rho``, ``p``, ``n`` y ``efecto``.
    """
    rows = []
    for a, b in combinations(df.columns, 2):
        pair = df[[a, b]].dropna()
        rho, p = stats.spearmanr(pair[a], pair[b])
        rows.append(
            {
                "a": a,
                "b": b,
                "rho": float(rho),
                "p": float(p),
                "n": len(pair),
                "efecto": effect_label(float(rho), "rho"),
            }
        )
    return pd.DataFrame(rows)


def kruskal_effect(df: pd.DataFrame, value: str, group: str) -> dict[str, Any]:
    """Kruskal-Wallis: si la distribución de ``value`` cambia entre grupos.

    Tamaño del efecto ``ε² = H / (n - 1)``: proporción de la variación en los
    rangos explicada por el grupo (0 a 1).
    """
    data = df[[value, group]].dropna()
    samples = [g[value].to_numpy() for _, g in data.groupby(group, observed=True)]
    h, p = stats.kruskal(*samples)
    n = len(data)
    eps2 = float(h / (n - 1))
    return {
        "H": float(h),
        "p": float(p),
        "n": n,
        "grupos": len(samples),
        "epsilon2": eps2,
        "efecto": effect_label(eps2, "epsilon2"),
    }


def cramers_v(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    """Chi² de independencia y V de Cramér con corrección de sesgo (Bergsma 2013).

    La corrección evita que tablas grandes con poca asociación real aparenten
    un V positivo solo por el número de categorías.
    """
    table = pd.crosstab(x, y)
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    v = float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0
    return {"chi2": float(chi2), "p": float(p), "n": int(n), "V": v, "efecto": effect_label(v, "V")}
