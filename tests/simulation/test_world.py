import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from puzzle_recsys.simulation.world import RESTAURANT_COLUMNS, build_restaurants, join_population


def _classified(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "restaurant_id": [str(i) for i in range(n)],
            "name": ["X"] * n,
            "scian_code": rng.choice(["722511", "722514"], size=n),
            "scian_name": ["a"] * n,
            "staff_band": ["6 a 10 personas"] * n,
            "cve_mun": rng.choice(["002", "015"], size=n),
            "alcaldia": ["A"] * n,
            "ageb_cvegeo": ["0900200010010"] * n,
            "lat": rng.uniform(19.2, 19.5, n),
            "lng": rng.uniform(-99.3, -99.0, n),
            "registered_at": ["2010-07"] * n,
            "cuisine_pred": ["mexicana"] * n,
            "cuisine_source": ["RULE"] * n,
            "cuisine_confidence": [1.0] * n,
            "masked_name": ["x"] * n,
        }
    )


def test_build_restaurants_returns_contract_table() -> None:
    out = build_restaurants(
        _classified(),
        n=50,
        strata=["scian_code", "cve_mun"],
        rng=np.random.default_rng(1),
        timezone="America/Mexico_City",
    )
    assert len(out) == 50
    assert list(out.columns) == list(RESTAURANT_COLUMNS)
    assert out["restaurant_id"].is_unique
    assert (out["timezone"] == "America/Mexico_City").all()


def test_join_population_keeps_matched_agebs_and_reports_lost() -> None:
    polys = gpd.GeoDataFrame(
        {"CVEGEO": ["A", "B", "C"], "CVE_MUN": ["002", "002", "015"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:6372",
    )
    pop = pd.DataFrame(
        {
            "cvegeo": ["A", "B", "Z"],
            "cve_mun": ["002"] * 3,
            "alcaldia": ["Az"] * 3,
            "population": [10, None, 5],
        }
    )
    layer, lost = join_population(polys, pop)
    assert layer["cvegeo"].tolist() == ["A"]  # B sin población, C sin censo
    assert lost == {
        "census_without_polygon": 1,
        "polygon_without_census": 1,
        "missing_population": 1,
    }
    assert layer.crs == polys.crs
