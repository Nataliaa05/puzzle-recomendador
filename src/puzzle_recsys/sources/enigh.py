"""Microdatos de la ENIGH (INEGI): gasto de los hogares en alimentos fuera de casa.

Variables del concentrado de hogares que se usan (todas trimestrales):

- ``ali_fuera``: gasto en alimentos y bebidas consumidos fuera del hogar.
- ``ing_cor``: ingreso corriente del hogar.
- ``tot_integ``: integrantes del hogar.
- ``factor``: **factor de expansión** (cuántos hogares representa cada registro).
  Cualquier remuestreo tiene que ponderarse por él; si no, el resultado no
  representa a la población sino a la muestra de la encuesta.

Usado en: notebooks 03.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from puzzle_recsys.sources.inegi import read_inegi_csv

_COLUMNS = ["folioviv", "foliohog", "ubica_geo", "tot_integ", "factor", "ing_cor", "ali_fuera"]


def read_enigh_households(zip_path: Path, *, entity: str) -> pd.DataFrame:
    """Hogares de una entidad con su gasto e ingreso per cápita (trimestral).

    Parameters
    ----------
    zip_path : Path
        ZIP ``enigh<año>_ns_concentradohogar_csv.zip``.
    entity : str
        Clave de entidad de 2 dígitos (``"09"`` = Ciudad de México).

    Returns
    -------
    pd.DataFrame
        ``household_id``, ``size``, ``weight`` (factor de expansión),
        ``income_pc`` y ``food_away_pc`` (pesos por persona por trimestre).
    """
    df = read_inegi_csv(zip_path, member_contains="concentradohogar", usecols=_COLUMNS)
    df = df[df["ubica_geo"].str.zfill(5).str[:2] == entity]
    size = pd.to_numeric(df["tot_integ"])
    out = pd.DataFrame(
        {
            "household_id": df["folioviv"] + "-" + df["foliohog"],
            "size": size,
            "weight": pd.to_numeric(df["factor"]),
            "income_pc": pd.to_numeric(df["ing_cor"]) / size,
            "food_away_pc": pd.to_numeric(df["ali_fuera"]) / size,
        }
    )
    return out.reset_index(drop=True)
