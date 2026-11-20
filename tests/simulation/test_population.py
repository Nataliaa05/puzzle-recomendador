import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from puzzle_recsys.config import load_simulation
from puzzle_recsys.simulation.population import (
    onboarding_answers,
    sample_budgets,
    sample_homes,
    sample_tastes,
    weighted_deciles,
)
from puzzle_recsys.sources.enigh import read_enigh_households


@pytest.fixture(scope="module")
def pop_cfg() -> dict:
    return load_simulation().population


def test_read_enigh_filters_entity_and_builds_per_capita(tmp_path: Path) -> None:
    csv = (
        "folioviv,foliohog,ubica_geo,tot_integ,factor,ing_cor,ali_fuera\n"
        "0900000001,1,09002,2,1000,60000,3000\n"
        "1500000002,1,15001,4,800,40000,2000\n"
        "0900000003,1,09015,4,1200,90000,0\n"
    )
    path = tmp_path / "enigh.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("concentradohogar.csv", csv)
    out = read_enigh_households(path, entity="09")
    assert len(out) == 2
    assert out.loc[0, "food_away_pc"] == 1500
    assert out.loc[0, "income_pc"] == 30000
    assert out["weight"].tolist() == [1000, 1200]


def test_weighted_deciles_split_population_weight_evenly() -> None:
    rng = np.random.default_rng(0)
    values = rng.lognormal(10, 1, 5000)
    weights = rng.integers(1, 3000, 5000)
    deciles = weighted_deciles(values, weights)
    shares = pd.Series(weights).groupby(deciles).sum() / weights.sum()
    assert set(deciles) == set(range(1, 11))
    assert np.allclose(shares, 0.1, atol=0.02)


def _ageb() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"cvegeo": ["A", "B"], "population": [9000, 1000]},
        geometry=[box(0, 0, 1000, 1000), box(5000, 5000, 6000, 6000)],
        crs="EPSG:6372",
    )


def test_homes_are_inside_their_ageb_and_follow_population() -> None:
    ageb = _ageb()
    homes = sample_homes(ageb, 2000, np.random.default_rng(1))
    share_a = (homes["cvegeo"] == "A").mean()
    assert 0.86 < share_a < 0.94
    projected = gpd.GeoSeries(
        gpd.points_from_xy(homes["lng"], homes["lat"]), crs="EPSG:4326"
    ).to_crs(ageb.crs)
    polygons = ageb.set_index("cvegeo").geometry
    inside = [
        polygons[c].buffer(1e-6).contains(Point(p.x, p.y))
        for c, p in zip(homes["cvegeo"], projected, strict=True)
    ]
    assert all(inside)


def _households(rng: np.random.Generator, n: int = 3000) -> pd.DataFrame:
    income = rng.lognormal(10, 0.8, n)
    return pd.DataFrame(
        {
            "weight": rng.integers(500, 2000, n),
            "income_pc": income,
            "food_away_pc": income * 0.04 * rng.lognormal(0, 0.3, n),
        }
    )


def test_budgets_and_frequency_grow_with_ses(pop_cfg) -> None:
    rng = np.random.default_rng(2)
    pct = np.linspace(0.01, 0.99, 4000)
    ticket, frequency = sample_budgets(pct, _households(rng), pop_cfg["budget"], rng)
    assert (ticket > 0).all() and (frequency > 0).all()
    low, high = pct < 0.2, pct > 0.8
    assert np.median(ticket[high]) > 1.5 * np.median(ticket[low])
    assert np.median(frequency[high]) > 1.5 * np.median(frequency[low])


def test_ticket_times_frequency_reproduces_spend(pop_cfg) -> None:
    """La descomposición no inventa ni pierde gasto: ticket·salidas·frecuencia = gasto."""
    cfg = {**pop_cfg["budget"], "smoothing_sigma": 0.0}
    households = _households(np.random.default_rng(3))
    pct = np.full(500, 0.5)
    ticket, frequency = sample_budgets(pct, households, cfg, np.random.default_rng(4))
    spend = ticket * cfg["outings_per_quarter"] * frequency
    assert np.isin(np.round(spend, 6), np.round(households["food_away_pc"].to_numpy(), 6)).all()


CUISINES = ["mexicana", "italiana", "japonesa", "tacos_tortas"]


@pytest.fixture(scope="module")
def tastes(pop_cfg):
    return sample_tastes(
        20_000,
        cuisines=CUISINES,
        cuisine_share=np.array([0.5, 0.2, 0.1, 0.2]),
        proteins=["BEEF", "PORK", "CHICKEN", "PLANT"],
        ingredients=["cebolla", "cilantro", "queso", "camaron"],
        allergens=list(pop_cfg["allergy_prevalence"]),
        cfg=pop_cfg,
        rng=np.random.default_rng(3),
    )


def test_cuisine_affinities_are_distributions(tastes) -> None:
    assert np.allclose(tastes.cuisine_affinity.sum(axis=1), 1.0)
    # el gusto promedio refleja la oferta de la ciudad
    assert tastes.cuisine_affinity.mean(axis=0).argmax() == 0


def test_allergy_prevalence_matches_config(tastes, pop_cfg) -> None:
    for j, allergen in enumerate(tastes.allergen_names):
        expected = pop_cfg["allergy_prevalence"][allergen]
        assert abs(tastes.allergies[:, j].mean() - expected) < 0.006


def test_vegans_are_also_vegetarian(tastes) -> None:
    vegan = tastes.diet == "vegano"
    assert vegan.any()
    assert set(np.unique(tastes.diet)) <= {"ninguna", "vegetariano", "vegano"}


def test_onboarding_never_invents_allergies(tastes, pop_cfg) -> None:
    answers = onboarding_answers(tastes, pop_cfg["onboarding"], np.random.default_rng(4))
    assert abs(answers["completed"].mean() - pop_cfg["onboarding"]["completion_rate"]) < 0.02
    assert answers.loc[~answers["completed"], "declared_spice"].isna().all()
    names = tastes.allergen_names
    for i, declared in zip(answers.index, answers["declared_allergies"], strict=True):
        true = {names[j] for j in np.flatnonzero(tastes.allergies[i])}
        assert set(declared) <= true
