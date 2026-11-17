"""Lectura de productos del INEGI: Censo 2020 por AGEB y Marco Geoestadístico.

Particularidades de los archivos reales (verificadas en la descarga):

- El CSV del censo trae un encabezado UTF-8 con BOM, pero el cuerpo en
  latin-1 ("México" llega como bytes latin-1).
- Los totales por AGEB son las filas con ``NOM_LOC == "Total AGEB urbana"``.
- Los valores confidenciales aparecen como ``"*"``.
- La clave geográfica de una AGEB urbana es ``ENTIDAD + MUN + LOC + AGEB``
  (13 caracteres) y coincide con ``CVEGEO`` del Marco Geoestadístico.

Usado en: notebooks 01.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

_BOM = "﻿"


def _decode(raw: bytes) -> str:
    """Decodifica UTF-8 y, si falla, latin-1 (sin perder el BOM del encabezado)."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        return raw.decode("latin-1")


def read_inegi_csv(zip_path: Path, *, member_contains: str, **read_csv_kwargs) -> pd.DataFrame:
    """Lee el CSV de un ZIP del INEGI con su codificación mixta.

    Parameters
    ----------
    zip_path : Path
        Archivo ZIP descargado.
    member_contains : str
        Texto que identifica al CSV de datos dentro del ZIP.
    **read_csv_kwargs
        Argumentos extra para :func:`pandas.read_csv` (``usecols``...).

    Returns
    -------
    pd.DataFrame
        Todas las columnas como texto (las claves llevan ceros a la izquierda).
    """
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            name
            for name in zf.namelist()
            if member_contains in name and name.lower().endswith(".csv")
        ]
        if len(members) != 1:
            raise FileNotFoundError(f"Se esperaba 1 CSV con '{member_contains}', hay {members}")
        text = _decode(zf.read(members[0]))
    df = pd.read_csv(io.StringIO(text), dtype=str, **read_csv_kwargs)
    df.columns = [c.lstrip(_BOM).strip() for c in df.columns]
    return df


def read_ageb_population(zip_path: Path) -> pd.DataFrame:
    """Población total por AGEB urbana (Censo 2020).

    Parameters
    ----------
    zip_path : Path
        ZIP ``ageb_mza_urbana_<ent>_cpv2020_csv.zip``.

    Returns
    -------
    pd.DataFrame
        Columnas ``cvegeo`` (13 caracteres), ``cve_mun``, ``alcaldia`` y
        ``population`` (``NaN`` cuando el INEGI la reserva por confidencialidad).
    """
    df = read_inegi_csv(
        zip_path,
        member_contains="conjunto_de_datos_",
        usecols=["ENTIDAD", "MUN", "NOM_MUN", "LOC", "NOM_LOC", "AGEB", "MZA", "POBTOT"],
    )
    totals = df[df["NOM_LOC"].str.strip().eq("Total AGEB urbana")]
    return pd.DataFrame(
        {
            "cvegeo": totals["ENTIDAD"] + totals["MUN"] + totals["LOC"] + totals["AGEB"],
            "cve_mun": totals["MUN"],
            "alcaldia": totals["NOM_MUN"],
            "population": pd.to_numeric(totals["POBTOT"], errors="coerce"),
        }
    ).reset_index(drop=True)


def read_ageb_polygons(zip_path: Path, *, entity: str = "09"):
    """Polígonos de AGEB urbanas del Marco Geoestadístico (capa ``<ent>a.shp``).

    Parameters
    ----------
    zip_path : Path
        ZIP del Marco Geoestadístico de la entidad.
    entity : str, optional
        Clave de entidad de 2 dígitos (``"09"`` = Ciudad de México).

    Returns
    -------
    geopandas.GeoDataFrame
        Con ``CVEGEO`` (13 caracteres) en el CRS original del INEGI
        (Lambert cónica conforme, en metros).
    """
    import geopandas as gpd

    return gpd.read_file(f"zip://{Path(zip_path).as_posix()}!conjunto_de_datos/{entity}a.shp")
