"""Tests del motor sobre una corrida chica con los menús reales del notebook 02."""

import numpy as np
import pandas as pd
import pytest

from puzzle_recsys.catalog.library import load_library
from puzzle_recsys.config import load_paths, load_simulation
from puzzle_recsys.repository.parquet_repo import ParquetRepository, run_id_for
from puzzle_recsys.simulation.engine import (
    Agenda,
    available_at,
    build_world_index,
    simulate_visits,
)
from puzzle_recsys.simulation.population import sample_tastes

N_DINERS = 120
MONTHS = 2


@pytest.fixture(scope="module")
def run():
    paths, sim = load_paths(), load_simulation()
    repo = ParquetRepository(paths.synthetic / run_id_for(sim.run_name, sim.seed))
    if not (repo.root / "menu_items.parquet").exists():
        pytest.skip("requiere la corrida del notebook 02")
    tables = {
        n: repo.read_table(n)
        for n in (
            "restaurants",
            "canonical_dishes",
            "menu_items",
            "menu_item_attributes",
            "menu_item_price_history",
            "menu_item_availability_events",
        )
    }
    tables["ground_truth_menu_items"] = repo.ground_truth.read_table("menu_items")
    library = load_library(paths.external / "canonical_dishes.yaml")
    cuisines = sorted(tables["restaurants"]["cuisine_pred"].unique())
    rng = np.random.default_rng(0)
    index, _ = build_world_index(
        tables, library, cuisines, sim.choice, sim.calendar["timezone"], rng
    )

    def tastes(n, seed):
        return sample_tastes(
            n,
            cuisines=cuisines,
            cuisine_share=np.ones(len(cuisines)) / len(cuisines),
            proteins=[
                "BEEF",
                "PORK",
                "CHICKEN",
                "FISH",
                "SEAFOOD",
                "PLANT",
                "EGG",
                "CHEESE",
                "MIXED",
            ],
            ingredients=list(library.ingredients),
            allergens=list(library.allergens),
            cfg={
                **sim.population,
                "allergy_prevalence": {a: 0.15 for a in library.allergens},
                "diets": {"vegetariano": 0.15, "vegano": 0.05},
            },
            rng=np.random.default_rng(seed),
        )

    registered, anon = tastes(N_DINERS, 1), tastes(300, 2)
    r = tables["restaurants"].sample(N_DINERS, replace=True, random_state=3)
    homes = pd.DataFrame({"lat": r["lat"].to_numpy() + 0.005, "lng": r["lng"].to_numpy()})
    cfg = {
        k: getattr(sim, k)
        for k in ("visits", "choice", "sharing", "population", "ratings", "noise")
    }

    def go(seed):
        return simulate_visits(
            index,
            registered,
            homes,
            np.full(N_DINERS, 150.0),
            (anon, np.full(300, 150.0)),
            cfg,
            sim.calendar,
            months=MONTHS,
            rng=np.random.default_rng(seed),
        )

    return {"index": index, "out": go(7), "again": go(7), "registered": registered, "anon": anon}


def test_simulation_produces_visits(run) -> None:
    out = run["out"]
    assert len(out["dining_sessions"]) > N_DINERS
    assert len(out["order_items"]) > len(out["dining_sessions"])


def test_pax_equals_number_of_guests(run) -> None:
    out = run["out"]
    counts = out["guests"].groupby("session_id").size()
    pax = out["dining_sessions"].set_index("session_id")["pax"]
    assert (counts.reindex(pax.index) == pax).all()


def test_every_line_was_available_at_its_price(run) -> None:
    index, out = run["index"], run["out"]
    pos = {item: k for k, item in enumerate(index.item_ids)}
    lines = out["order_items"].merge(
        out["dining_sessions"][["session_id", "opened_at"]], on="session_id"
    )
    local = lines["opened_at"].dt.tz_convert("America/Mexico_City").dt.tz_localize(None)
    for item_id, t, price in zip(
        lines["menu_item_id"],
        local.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        lines["unit_price_cents"],
        strict=True,
    ):
        avail, prices = available_at(index, np.array([pos[item_id]]), int(t))
        assert len(avail) == 1, item_id
        assert round(prices[0] * 100) == price


def _tastes_of(run, ref: str):
    pool = run["registered"] if ref.startswith("d") else run["anon"]
    return pool, int(ref[1:])


def test_no_one_receives_their_allergens_or_breaks_their_diet(run) -> None:
    index, out = run["index"], run["out"]
    pos = {item: k for k, item in enumerate(index.item_ids)}
    lines = out["order_items"].merge(out["guests"], on=["guest_id", "session_id"])
    for _, line in lines[~lines["is_shared"]].iterrows():
        pool, j = _tastes_of(run, line["taste_ref"])
        k = pos[line["menu_item_id"]]
        assert not (index.items.allergens[k] & pool.allergies[j]).any()
        if pool.diet[j] == "vegetariano":
            assert index.items.vegetarian[k]
        if pool.diet[j] == "vegano":
            assert index.items.vegan[k]


def test_shared_dishes_are_only_shared_by_people_who_can_eat_them(run) -> None:
    """Un platillo para compartir se reparte solo entre quienes pueden comerlo (≥ 2 personas).

    Quien es alérgico (o tiene una dieta incompatible) queda FUERA del reparto: no se
    descarta el platillo para toda la mesa.
    """
    index, out = run["index"], run["out"]
    pos = {item: k for k, item in enumerate(index.item_ids)}
    ref_of = out["guests"].set_index("guest_id")["taste_ref"]
    table = out["guests"].groupby("session_id")["guest_id"].apply(list)
    shared = out["order_items"][out["order_items"]["is_shared"]]
    assert len(shared) > 0
    excluded_someone = 0
    for _, line in shared.iterrows():
        k = pos[line["menu_item_id"]]
        assert index.items.shareable[k]
        eaters = list(line["shared_by"])
        assert len(eaters) >= 2
        assert line["guest_id"] in eaters  # quien lo pide, lo come
        for guest in eaters:
            pool, j = _tastes_of(run, ref_of[guest])
            assert not (index.items.allergens[k] & pool.allergies[j]).any()
            if pool.diet[j] == "vegetariano":
                assert index.items.vegetarian[k]
            if pool.diet[j] == "vegano":
                assert index.items.vegan[k]
        excluded_someone += len(eaters) < len(table[line["session_id"]])
    assert excluded_someone > 0  # con 15 % de alérgicos, alguien debió quedar fuera


def test_same_seed_same_simulation(run) -> None:
    pd.testing.assert_frame_equal(run["out"]["order_items"], run["again"]["order_items"])


def test_visit_days_follow_the_weekday_profile() -> None:
    """Regresión: pasar al día siguiente al rechazar deformaba el perfil semanal."""
    from puzzle_recsys.simulation.engine import visit_days

    weights = np.array([0.10, 0.11, 0.12, 0.14, 0.19, 0.20, 0.14])
    start = pd.Timestamp("2025-01-06").value
    end = pd.Timestamp("2045-01-06").value
    days = visit_days(start, end, 4.0, 1.3, weights / weights.max(), np.random.default_rng(0))
    weekday = (np.asarray(days) // (86_400 * 10**9) + 3) % 7
    observed = np.bincount(weekday, minlength=7) / len(days)
    assert np.allclose(observed, weights / weights.sum(), atol=0.012)


def test_visit_days_keep_the_expected_frequency() -> None:
    from puzzle_recsys.simulation.engine import visit_days

    weights = np.array([0.10, 0.11, 0.12, 0.14, 0.19, 0.20, 0.14])
    start = pd.Timestamp("2025-01-06").value
    end = pd.Timestamp("2035-01-06").value
    days = visit_days(start, end, 10.0, 1.0, weights / weights.max(), np.random.default_rng(1))
    mean_gap = np.diff(days).mean() / (86_400 * 10**9)
    assert 9.0 < mean_gap < 11.5


def test_special_occasions_go_to_pricier_places(run) -> None:
    """Regresión: en la 1.ª versión las ocasiones iban a lugares 0.96x igual de caros."""
    sim = load_simulation()
    cfg = {
        k: getattr(sim, k)
        for k in ("visits", "choice", "sharing", "population", "ratings", "noise")
    }
    cfg["choice"] = {
        **cfg["choice"],
        "occasion": {**cfg["choice"]["occasion"], "probability": 0.5, "weekend_factor": 1.0},
    }
    index = run["index"]
    rng = np.random.default_rng(31)
    r = index.rest_latlng[rng.integers(0, len(index.rest_ids), N_DINERS)]
    homes = pd.DataFrame({"lat": np.degrees(r[:, 0]) + 0.004, "lng": np.degrees(r[:, 1])})
    out = simulate_visits(
        index,
        run["registered"],
        homes,
        np.full(N_DINERS, 250.0),
        (run["anon"], np.full(300, 250.0)),
        cfg,
        sim.calendar,
        months=4,
        rng=np.random.default_rng(32),
    )
    v = out["ground_truth_visits"].merge(out["dining_sessions"], on="session_id")
    level = dict(zip(index.rest_ids, index.rest_price_level, strict=True))
    v["level"] = v["restaurant_id"].map(level)
    by = v.groupby("occasion")["level"].median()
    assert by[True] > 1.15 * by[False]


def test_agenda_blocks_overlapping_times_in_both_directions() -> None:
    hour = 3600 * 10**9
    agenda = Agenda(window_ns=3 * hour)
    assert agenda.is_free(5, 10 * hour)
    agenda.book(5, 10 * hour)
    assert not agenda.is_free(5, 12 * hour)  # 2 h después: sigue en la mesa
    assert not agenda.is_free(5, 8 * hour)  # 2 h antes: ya no llegaría
    assert agenda.is_free(5, 13 * hour)  # justo en el límite
    assert agenda.is_free(6, 10 * hour)  # otra persona


def test_no_registered_diner_is_at_two_tables_at_once(run) -> None:
    """Regresión: un acompañante registrado aparecía en dos mesas con minutos de diferencia."""
    out = run["out"]
    window_h = load_simulation().visits["min_hours_between_visits"]
    x = (
        out["guests"]
        .dropna(subset=["diner_id"])
        .merge(out["dining_sessions"][["session_id", "opened_at"]], on="session_id")
        .sort_values(["diner_id", "opened_at"])
    )
    gap_h = x.groupby("diner_id")["opened_at"].diff().dt.total_seconds() / 3600
    assert (gap_h.dropna() >= window_h).all()


def test_no_diner_appears_twice_in_the_same_table(run) -> None:
    guests = run["out"]["guests"].dropna(subset=["diner_id"])
    assert not guests.duplicated(subset=["session_id", "diner_id"]).any()


def test_context_shapes_where_and_what_people_eat(run) -> None:
    """Regresión: la hora no influía en nada (EDA bivariado, notebook 07)."""
    from puzzle_recsys.features.calendar import WEEKEND_DAYS, period_index

    out, index = run["out"], run["index"]
    tz = load_simulation().calendar["timezone"]
    s = out["dining_sessions"].copy()
    local = s["opened_at"].dt.tz_convert(tz)
    s["period"] = period_index(local.dt.hour.to_numpy())
    s["weekend"] = local.dt.day_of_week.isin(WEEKEND_DAYS).to_numpy()
    cuisine = dict(zip(index.rest_ids, index.rest_cuisine_idx, strict=True))
    cafe = index.cuisines.index("cafeteria")
    s["cafe"] = s["restaurant_id"].map(cuisine) == cafe
    by_period = s.groupby("period")["cafe"].mean()
    assert by_period[0] > 2 * by_period[2]  # cafetería: desayuno >> cena

    pos = {m: k for k, m in enumerate(index.item_ids)}
    lines = out["order_items"].merge(s[["session_id", "period"]], on="session_id")
    k = lines["menu_item_id"].map(pos).to_numpy()
    alcohol = index.dish_period_bonus[k, 0] < -1
    drinks = index.items.course[k] == "DRINK"
    share = pd.Series(alcohol[drinks]).groupby(lines["period"].to_numpy()[drinks]).mean()
    assert share[0] < 0.5 * share[2]  # alcohol: casi nada en el desayuno

    assert s.loc[s["weekend"], "pax"].mean() > s.loc[~s["weekend"], "pax"].mean()
