import zipfile
from pathlib import Path

import pandas as pd

from puzzle_recsys.sources.inegi import read_ageb_population, read_inegi_csv

HEADER = "ENTIDAD,NOM_ENT,MUN,NOM_MUN,LOC,NOM_LOC,AGEB,MZA,POBTOT\n"
ROWS = [
    "09,Ciudad de México,000,Total de la entidad,0000,Total de la entidad,0000,000,9209944\n",
    "09,Ciudad de México,002,Azcapotzalco,0001,Total AGEB urbana,0010,000,3183\n",
    "09,Ciudad de México,002,Azcapotzalco,0001,Azcapotzalco,0010,001,159\n",
    "09,Ciudad de México,015,Cuauhtémoc,0001,Total AGEB urbana,0021,000,*\n",
    "09,Ciudad de México,015,Cuauhtémoc,0001,Total AGEB urbana,0035,000,2500\n",
]


def _zip(tmp_path: Path) -> Path:
    # Encabezado UTF-8 con BOM + cuerpo latin-1: así vienen los archivos reales del INEGI.
    payload = b"\xef\xbb\xbf" + HEADER.encode() + "".join(ROWS).encode("latin-1")
    path = tmp_path / "censo.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("x/conjunto_de_datos/conjunto_de_datos_ageb_urbana_09_cpv2020.csv", payload)
        zf.writestr("x/diccionario_de_datos/dic.csv", "a,b\n")
    return path


def test_read_inegi_csv_handles_bom_and_latin1(tmp_path: Path) -> None:
    df = read_inegi_csv(_zip(tmp_path), member_contains="conjunto_de_datos_")
    assert df.columns[0] == "ENTIDAD"
    assert "Cuauhtémoc" in set(df["NOM_MUN"])


def test_ageb_population_keeps_only_ageb_totals(tmp_path: Path) -> None:
    pop = read_ageb_population(_zip(tmp_path))
    assert list(pop.columns) == ["cvegeo", "cve_mun", "alcaldia", "population"]
    assert len(pop) == 3
    assert pop["cvegeo"].str.len().eq(13).all()
    assert pop.loc[pop["cvegeo"] == "0900200010010", "population"].item() == 3183


def test_confidential_population_becomes_missing(tmp_path: Path) -> None:
    pop = read_ageb_population(_zip(tmp_path))
    value = pop.loc[pop["cvegeo"] == "0901500010021", "population"].item()
    assert pd.isna(value)
