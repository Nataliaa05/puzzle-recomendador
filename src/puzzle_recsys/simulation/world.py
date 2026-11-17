"""Capa Mundo: restaurantes reales (DENUE) y AGEB con población.

- ``restaurants``: muestra estratificada del universo de clientes posibles,
  con su cocina homologada (reglas + KNN). Es la tabla del contrato.
- ``ageb``: polígonos de AGEB urbanas con su población; se usa en el
  notebook 03 para ubicar a los comensales con probabilidad proporcional
  a la población.

Usado en: notebooks 01, 03.
"""

from __future__ import annotations

from collections.abc import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

from puzzle_recsys.simulation.distributions import stratified_sample

#: Columnas de la tabla ``restaurants`` del contrato.
RESTAURANT_COLUMNS: tuple[str, ...] = (
    "restaurant_id",
    "name",
    "scian_code",
    "cuisine_pred",
    "cuisine_source",
    "cuisine_confidence",
    "staff_band",
    "cve_mun",
    "alcaldia",
    "ageb_cvegeo",
    "lat",
    "lng",
    "timezone",
)


def build_restaurants(
    classified: pd.DataFrame,
    *,
    n: int,
    strata: Sequence[str],
    rng: np.random.Generator,
    timezone: str,
) -> pd.DataFrame:
    """Muestra ``n`` restaurantes estratificados y los lleva al contrato.

    Parameters
    ----------
    classified : pd.DataFrame
        Universo con la cocina ya imputada (salida de
        :func:`~puzzle_recsys.catalog.cuisine_imputation.impute_cuisine_knn`).
    n : int
        Restaurantes de la simulación.
    strata : Sequence[str]
        Columnas de estratificación (p. ej. SCIAN y alcaldía), para conservar
        la mezcla real de giros y de geografía.
    rng : np.random.Generator
        Generador aleatorio.
    timezone : str
        Zona horaria IANA de los restaurantes.

    Returns
    -------
    pd.DataFrame
        Tabla ``restaurants`` con :data:`RESTAURANT_COLUMNS`.
    """
    sample = stratified_sample(classified, n, strata, rng).assign(timezone=timezone)
    return sample.loc[:, list(RESTAURANT_COLUMNS)].reset_index(drop=True)


def join_population(
    polygons: gpd.GeoDataFrame, population: pd.DataFrame
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Une los polígonos de AGEB con su población del censo.

    Solo quedan las AGEB con polígono **y** población conocida: son las únicas
    donde se puede ubicar a un comensal con probabilidad proporcional a la
    población. Lo que se pierde se reporta, no se esconde.

    Parameters
    ----------
    polygons : gpd.GeoDataFrame
        Capa ``<ent>a.shp`` del Marco Geoestadístico (columnas ``CVEGEO``...).
    population : pd.DataFrame
        Salida de :func:`~puzzle_recsys.sources.inegi.read_ageb_population`.

    Returns
    -------
    tuple[gpd.GeoDataFrame, dict[str, int]]
        La capa (``cvegeo``, ``cve_mun``, ``alcaldia``, ``population``,
        ``geometry``) en el CRS original, y el conteo de AGEB perdidas.
    """
    poly = polygons.rename(columns={"CVEGEO": "cvegeo"})[["cvegeo", "geometry"]]
    merged = poly.merge(population, on="cvegeo", how="inner")
    lost = {
        "census_without_polygon": int((~population["cvegeo"].isin(poly["cvegeo"])).sum()),
        "polygon_without_census": int((~poly["cvegeo"].isin(population["cvegeo"])).sum()),
        "missing_population": int(merged["population"].isna().sum()),
    }
    layer = merged.dropna(subset=["population"]).reset_index(drop=True)
    layer = gpd.GeoDataFrame(
        layer[["cvegeo", "cve_mun", "alcaldia", "population", "geometry"]],
        geometry="geometry",
        crs=polygons.crs,
    )
    return layer, lost
