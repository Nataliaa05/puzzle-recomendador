"""Funciones de graficado reutilizables (estilo único del proyecto).

Usado en: notebooks 01, 04-15.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


def plot_world(
    ageb: gpd.GeoDataFrame,
    restaurants: pd.DataFrame,
    *,
    top_cuisines: int = 8,
    save_to: Path | None = None,
) -> Axes:
    """Mapa del mundo simulado: densidad de población por AGEB + restaurantes por cocina.

    Parameters
    ----------
    ageb : gpd.GeoDataFrame
        AGEB con ``population`` (cualquier CRS proyectado en metros).
    restaurants : pd.DataFrame
        Tabla ``restaurants`` con ``lat``, ``lng`` y ``cuisine_pred``.
    top_cuisines : int, optional
        Cocinas con color propio; el resto se agrupa como "otras".
    save_to : Path, optional
        Si se indica, guarda la figura (PNG, 150 dpi).

    Returns
    -------
    Axes
        Eje de matplotlib con el mapa.
    """
    density = ageb.assign(density=ageb["population"] / (ageb.geometry.area / 1e6))
    points = gpd.GeoDataFrame(
        restaurants,
        geometry=gpd.points_from_xy(restaurants["lng"], restaurants["lat"]),
        crs="EPSG:4326",
    ).to_crs(ageb.crs)
    top = points["cuisine_pred"].value_counts().index[:top_cuisines]
    points["cocina"] = points["cuisine_pred"].where(points["cuisine_pred"].isin(top), "otras")

    _, ax = plt.subplots(figsize=(9, 11))
    density.plot(
        column="density",
        scheme="quantiles",
        k=5,
        cmap="Greys",
        linewidth=0,
        legend=True,
        legend_kwds={"title": "hab/km²", "loc": "lower left"},
        ax=ax,
    )
    points.plot(column="cocina", categorical=True, markersize=6, legend=True, ax=ax, cmap="tab10")
    ax.set_title("Mundo simulado: población por AGEB y restaurantes por cocina")
    ax.set_axis_off()
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_to, dpi=150, bbox_inches="tight")
    return ax


def plot_lorenz(counts: pd.Series, *, title: str, ax: Axes | None = None) -> Axes:
    """Curva de Lorenz: qué % de los pedidos concentra cada % de platillos (más pedidos primero)."""
    import numpy as np

    values = np.sort(np.asarray(counts, dtype=float))[::-1]
    x = np.arange(1, len(values) + 1) / len(values)
    y = np.cumsum(values) / values.sum()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, label="simulado")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="todos igual de populares")
    ax.axvline(0.2, color="grey", linewidth=0.8)
    ax.set_xlabel("fracción de platillos (más pedidos primero)")
    ax.set_ylabel("fracción acumulada de pedidos")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return ax


def plot_compare(
    expected: pd.Series, observed: pd.Series, *, title: str, ax: Axes | None = None
) -> Axes:
    """Barras lado a lado: distribución configurada contra la simulada."""
    frame = pd.DataFrame({"configurado": expected, "simulado": observed})
    ax = frame.plot.bar(ax=ax, figsize=(6, 3.5), rot=0)
    ax.set_title(title)
    ax.set_ylabel("proporción")
    return ax


def plot_distribution(
    series: pd.Series, *, title: str, log: bool = False, bins: int = 50, ax: Axes | None = None
) -> Axes:
    """Histograma con media y mediana marcadas; ``log=True`` usa bins logarítmicos.

    La separación entre la media y la mediana deja ver la asimetría a simple vista.
    """
    import numpy as np

    values = series.dropna().astype(float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    edges = np.geomspace(values.min(), values.max(), bins) if log else bins
    ax.hist(values, bins=edges, color="#4C72B0", alpha=0.85)
    ax.axvline(values.median(), color="black", linewidth=1.2, label="mediana")
    ax.axvline(values.mean(), color="#C44E52", linestyle="--", linewidth=1.2, label="media")
    if log:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_ylabel("frecuencia")
    ax.legend(fontsize=8)
    return ax


def plot_frequencies(table: pd.DataFrame, *, title: str, ax: Axes | None = None) -> Axes:
    """Barras horizontales de una tabla de ``frequency_table`` (columna ``%``)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    table["%"].iloc[::-1].plot.barh(ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("% del total")
    return ax
