import pandas as pd
import pytest

from puzzle_recsys.sources.denue import STAFF_BANDS, select_restaurants


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5", "5"],
            "nom_estab": [
                "TRATTORIA",
                "PUESTO",
                "AUTOSERVICIO",
                "SIN COORD",
                "TAQUERIA",
                "TAQUERIA",
            ],
            "codigo_act": ["722511", "722511", "722516", "722514", "722514", "722514"],
            "nombre_act": ["a"] * 6,
            "per_ocu": [
                "11 a 30 personas",
                "0 a 5 personas",
                "51 a 100 personas",
                "6 a 10 personas",
                "6 a 10 personas",
                "6 a 10 personas",
            ],
            "cve_ent": ["09"] * 6,
            "cve_mun": ["015", "015", "014", "002", "007", "007"],
            "municipio": [
                "Cuauhtémoc",
                "Cuauhtémoc",
                "Benito Juárez",
                "Azcapotzalco",
                "Iztapalapa",
                "Iztapalapa",
            ],
            "cve_loc": ["0001"] * 6,
            "ageb": ["0021", "0021", "0035", "0010", "4185", "4185"],
            "latitud": ["19.43", "19.43", "19.38", None, "19.31", "19.31"],
            "longitud": ["-99.15", "-99.15", "-99.16", "-99.18", "-99.06", "-99.06"],
            "fecha_alta": ["2010-07"] * 6,
        }
    )


def test_select_filters_scian_size_coords_and_duplicates() -> None:
    out = select_restaurants(
        _raw(), include_scian=["722511", "722514"], min_staff_band="6 a 10 personas"
    )
    assert out["restaurant_id"].tolist() == ["1", "5"]


def test_select_builds_contract_columns() -> None:
    out = select_restaurants(
        _raw(), include_scian=["722511", "722514"], min_staff_band="6 a 10 personas"
    )
    row = out.iloc[0]
    assert row["ageb_cvegeo"] == "0901500010021"
    assert row["lat"] == pytest.approx(19.43)
    assert row["staff_band"] == "11 a 30 personas"
    assert set(out.columns) >= {"restaurant_id", "name", "scian_code", "cve_mun", "alcaldia"}


def test_unknown_staff_band_raises() -> None:
    with pytest.raises(ValueError, match="banda"):
        select_restaurants(_raw(), include_scian=["722511"], min_staff_band="7 personas")


def test_staff_bands_are_ordered() -> None:
    assert STAFF_BANDS[0] == "0 a 5 personas"
    assert STAFF_BANDS[-1] == "251 y más personas"
