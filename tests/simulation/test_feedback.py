import numpy as np
import pandas as pd
import pytest

from puzzle_recsys.config import load_simulation
from puzzle_recsys.simulation.feedback import simulate_ratings, stars_from_latent

# Se prueba la configuración REAL del proyecto (no una copia que se desincronice).
CFG = load_simulation().ratings


def test_stars_cover_1_to_5() -> None:
    cuts = np.array([-1.0, 0.0, 1.0, 2.0])
    assert stars_from_latent(np.array([-5, -0.5, 0.5, 1.5, 9]), cuts).tolist() == [1, 2, 3, 4, 5]


@pytest.fixture
def world():
    rng = np.random.default_rng(0)
    n = 20_000
    lines = pd.DataFrame(
        {
            "order_item_id": [f"l{i}" for i in range(n)],
            "session_id": [f"s{i // 4}" for i in range(n)],
            "guest_id": [f"s{i // 4}-g{i % 2}" for i in range(n)],
            "experienced": rng.normal(0, 1, n),
        }
    )
    guests = pd.DataFrame(
        {
            "guest_id": sorted(set(lines["guest_id"])),
        }
    )
    guests["session_id"] = guests["guest_id"].str.split("-").str[0]
    guests["diner_id"] = np.where(guests["guest_id"].str.endswith("g0"), "d1", None)
    sessions = pd.DataFrame(
        {
            "session_id": sorted(set(lines["session_id"])),
            "opened_at": pd.Timestamp("2025-03-01", tz="UTC"),
        }
    )
    return lines, guests, sessions


def test_only_registered_guests_rate(world) -> None:
    lines, guests, sessions = world
    out = simulate_ratings(lines, guests, sessions, CFG, np.random.default_rng(1))
    anon = set(guests.loc[guests["diner_id"].isna(), "guest_id"])
    assert not out["dish_ratings"]["guest_id"].isin(anon).any()
    assert not out["visit_ratings"]["guest_id"].isin(anon).any()


def test_extreme_experiences_are_rated_more_often(world) -> None:
    lines, guests, sessions = world
    out = simulate_ratings(lines, guests, sessions, CFG, np.random.default_rng(2))
    rated = lines["order_item_id"].isin(out["dish_ratings"]["order_item_id"])
    eligible = lines["guest_id"].str.endswith("g0")
    extreme = lines["experienced"].abs() > 1.5
    assert rated[eligible & extreme].mean() > 1.5 * rated[eligible & ~extreme].mean()


def test_observed_ratings_are_j_shaped(world) -> None:
    lines, guests, sessions = world
    out = simulate_ratings(lines, guests, sessions, CFG, np.random.default_rng(3))
    share = out["dish_ratings"]["stars"].value_counts(normalize=True)
    assert share[5] > share[4] > share[3]
    assert share[1] > share[2]


def test_rating_time_is_after_the_visit(world) -> None:
    lines, guests, sessions = world
    out = simulate_ratings(lines, guests, sessions, CFG, np.random.default_rng(4))
    assert (out["dish_ratings"]["created_at"] > sessions["opened_at"].iloc[0]).all()
