"""Reglas de calidad de datos: completitud, unicidad, integridad, validez, consistencia, outliers.

Cada función responde una pregunta concreta y devuelve un número, para poder
armar un reporte y para poder probarla de forma aislada.

Usado en: notebooks 05.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def completeness(df: pd.DataFrame) -> pd.Series:
    """Proporción de valores NO nulos por columna (1 = completa)."""
    return df.notna().mean()


def duplicate_keys(df: pd.DataFrame, keys: list[str]) -> int:
    """Filas sobrantes con la misma clave (0 = la clave es única)."""
    return int(df.duplicated(subset=keys).sum())


def orphan_keys(child: pd.DataFrame, child_key: str, parent: pd.DataFrame, parent_key: str) -> int:
    """Integridad referencial: filas cuya llave foránea NO existe en la tabla padre.

    Los nulos no cuentan como huérfanos: una llave opcional vacía es válida
    (p. ej. ``guest_id`` en un pedido que cargó el mesero).
    """
    values = child[child_key].dropna()
    return int((~values.isin(parent[parent_key])).sum())


def out_of_range(series: pd.Series, low: float, high: float) -> int:
    """Valores fuera de ``[low, high]`` (los nulos no cuentan)."""
    values = series.dropna()
    return int(((values < low) | (values > high)).sum())


def robust_outliers(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Outliers por **z-score modificado** (Iglewicz y Hoaglin).

    ``z = 0.6745 · (x - mediana) / MAD``. A diferencia del z-score clásico, usa
    mediana y desviación absoluta mediana, que no se dejan arrastrar por los
    propios outliers ni suponen normalidad (y estas variables no son normales).

    Returns
    -------
    pd.Series
        Booleana: ``True`` si ``|z| > threshold``.
    """
    values = series.astype(float)
    median = values.median()
    mad = (values - median).abs().median()
    if mad == 0:
        mad = (values - median).abs().mean() or 1.0
    z = 0.6745 * (values - median) / mad
    return z.abs() > threshold


def price_consistency(lines: pd.DataFrame, price_history: pd.DataFrame) -> float:
    """Proporción de líneas cobradas al precio vigente en el momento del pedido."""
    # Misma resolución temporal en ambos lados (Parquet puede devolver µs o ns).
    left = lines[["menu_item_id", "unit_price_cents", "created_at"]].assign(
        created_at=lambda d: pd.to_datetime(d["created_at"], utc=True).dt.as_unit("ns")
    )
    right = price_history[["menu_item_id", "price_cents", "valid_from"]].assign(
        valid_from=lambda d: pd.to_datetime(d["valid_from"], utc=True).dt.as_unit("ns")
    )
    left, right = left.sort_values("created_at"), right.sort_values("valid_from")
    merged = pd.merge_asof(
        left, right, left_on="created_at", right_on="valid_from", by="menu_item_id"
    )
    return float((merged["unit_price_cents"] == merged["price_cents"]).mean())


def missingness_test(df: pd.DataFrame, target: str, against: str) -> float:
    """¿Faltar ``target`` depende de ``against``? p-valor de la comparación.

    Compara la distribución de ``against`` entre las filas donde ``target``
    falta y donde está: Kolmogorov-Smirnov si es numérica, chi-cuadrada si es
    categórica. Un p-valor chico indica que el faltante **no** es completamente
    aleatorio (evidencia contra MCAR).

    Ojo: con datos observados solo se puede distinguir MCAR de MAR. Si el
    faltante depende del **propio valor que falta** (MNAR), los datos observados
    no lo revelan; aquí solo se puede verificar porque es una simulación.
    """
    missing = df[target].isna()
    values = df[against]
    if pd.api.types.is_numeric_dtype(values):
        return float(stats.ks_2samp(values[missing].dropna(), values[~missing].dropna()).pvalue)
    table = pd.crosstab(values, missing)
    if table.shape[1] < 2:
        return 1.0
    return float(stats.chi2_contingency(table)[1])


def summarize_numeric(series: pd.Series) -> dict[str, float]:
    """Resumen robusto: mediana, IQR, % de outliers (z modificado)."""
    q1, q3 = np.nanpercentile(series, [25, 75])
    return {
        "mediana": float(series.median()),
        "IQR": float(q3 - q1),
        "% outliers": float(robust_outliers(series.dropna()).mean() * 100),
    }
