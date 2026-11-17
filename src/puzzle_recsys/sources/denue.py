"""Lectura del DENUE y selección de restaurantes clientes posibles de Puzzle.

Usado en: notebooks 01.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from puzzle_recsys.sources.inegi import read_inegi_csv

#: Bandas de personal ocupado del DENUE, de menor a mayor.
STAFF_BANDS: tuple[str, ...] = (
    "0 a 5 personas",
    "6 a 10 personas",
    "11 a 30 personas",
    "31 a 50 personas",
    "51 a 100 personas",
    "101 a 250 personas",
    "251 y más personas",
)

_COLUMNS = [
    "id",
    "nom_estab",
    "codigo_act",
    "nombre_act",
    "per_ocu",
    "cve_ent",
    "cve_mun",
    "municipio",
    "cve_loc",
    "ageb",
    "latitud",
    "longitud",
    "fecha_alta",
]


def read_denue(zip_path: Path) -> pd.DataFrame:
    """Lee el DENUE de una entidad (``denue_<ent>_csv.zip``), solo las columnas útiles."""
    return read_inegi_csv(zip_path, member_contains="conjunto_de_datos/denue", usecols=_COLUMNS)


def select_restaurants(
    denue: pd.DataFrame,
    *,
    include_scian: Sequence[str],
    min_staff_band: str,
) -> pd.DataFrame:
    """Filtra el universo de restaurantes y lo lleva al contrato del proyecto.

    Parameters
    ----------
    denue : pd.DataFrame
        Salida de :func:`read_denue`.
    include_scian : Sequence[str]
        Códigos SCIAN de 6 dígitos que se consideran clientes posibles.
    min_staff_band : str
        Banda mínima de personal ocupado (una de :data:`STAFF_BANDS`).

    Returns
    -------
    pd.DataFrame
        Un restaurante por fila, sin coordenadas faltantes ni ids duplicados.

    Raises
    ------
    ValueError
        Si ``min_staff_band`` no es una banda conocida.
    """
    if min_staff_band not in STAFF_BANDS:
        raise ValueError(f"'{min_staff_band}' no es una banda de personal del DENUE")
    allowed_bands = STAFF_BANDS[STAFF_BANDS.index(min_staff_band) :]

    df = denue[denue["codigo_act"].isin(list(include_scian)) & denue["per_ocu"].isin(allowed_bands)]
    out = pd.DataFrame(
        {
            "restaurant_id": df["id"],
            "name": df["nom_estab"].str.strip(),
            "scian_code": df["codigo_act"],
            "scian_name": df["nombre_act"],
            "staff_band": df["per_ocu"],
            "cve_mun": df["cve_mun"],
            "alcaldia": df["municipio"],
            "ageb_cvegeo": df["cve_ent"] + df["cve_mun"] + df["cve_loc"] + df["ageb"],
            "lat": pd.to_numeric(df["latitud"], errors="coerce"),
            "lng": pd.to_numeric(df["longitud"], errors="coerce"),
            "registered_at": df["fecha_alta"],
        }
    )
    out = out.dropna(subset=["lat", "lng"]).drop_duplicates(subset="restaurant_id")
    return out.reset_index(drop=True)
