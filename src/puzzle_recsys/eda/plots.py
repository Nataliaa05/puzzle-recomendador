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
