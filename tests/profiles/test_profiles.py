import numpy as np
import pandas as pd
import pytest

from puzzle_recsys.profiles.diner_profile import (
    cuisine_prior,
    point_in_time_profiles,
    recovery_by_history,
    visit_events,
)
from puzzle_recsys.profiles.table_profile import group_aggregate, table_profile

CUISINES = ["mexicana", "pizza", "tacos_tortas"]


@pytest.fixture
def world():
    sessions = pd.DataFrame(
        {
            "session_id": ["s1", "s2", "s3"],
            "opened_at": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-10"], utc=True),
            "cocina": ["mexicana", "pizza", "mexicana"],
            "ticket_por_persona": [100.0, 200.0, 400.0],
        }
    )
    guests = pd.DataFrame(
        {
            "guest_id": ["g1", "g2", "g3", "g4"],
            "session_id": ["s1", "s1", "s2", "s3"],
            "diner_id": ["d1", None, "d1", "d1"],
        }
    )
    menu_items = pd.DataFrame(
        {
            "menu_item_id": ["m1", "m2", "m3"],
            "course": ["MAIN", "DRINK", "STARTER"],
            "spice_level": [2, 0, 1],
        }
    )
    order_items = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1", "s2", "s3", "s3"],
            "guest_id": ["g1", "g1", "g2", "g3", "g4", "g4"],
            "menu_item_id": ["m1", "m2", "m3", "m1", "m1", "m3"],
            "is_shared": [False, False, True, False, False, False],
        }
    )
    return dict(sessions=sessions, guests=guests, menu_items=menu_items, order_items=order_items)


def test_visit_events_count_own_individual_lines_per_visit(world) -> None:
    ev = visit_events(**world).set_index("session_id")
    assert ev.loc["s1", "n__MAIN"] == 1 and ev.loc["s1", "n__DRINK"] == 1
    assert ev.loc["s1", "n__STARTER"] == 0  # la entrada compartida no es de nadie en particular
    assert bool(ev.loc["s1", "compartio"]) and not bool(ev.loc["s2", "compartio"])
    assert ev.loc["s3", "picante_suma"] == 3 and ev.loc["s3", "picante_n"] == 2
    assert ev.loc["s2", "log_ticket_pp"] == pytest.approx(np.log(200))


def test_profile_uses_only_previous_visits(world) -> None:
    ev = visit_events(**world)
    prior = pd.DataFrame([[0.5, 0.25, 0.25]], index=["d1"], columns=CUISINES)
    prof = point_in_time_profiles(ev, cuisines=CUISINES, prior=prior, prior_strength=2.0)
    prof = prof.set_index("session_id")

    first = prof.loc["s1"]
    assert first["visitas_previas"] == 0
    assert first["afinidad__mexicana"] == pytest.approx(0.5)  # sin historia: el prior
    assert np.isnan(first["habito__MAIN"]) and np.isnan(first["dias_desde_ultima"])

    third = prof.loc["s3"]  # ve s1 (mexicana) y s2 (pizza), no s3
    assert third["visitas_previas"] == 2
    assert third["afinidad__mexicana"] == pytest.approx((1 + 2 * 0.5) / (2 + 2))
    assert third["afinidad__pizza"] == pytest.approx((1 + 2 * 0.25) / (2 + 2))
    assert third["habito__MAIN"] == pytest.approx(1.0)  # 2 platos fuertes en 2 visitas
    assert third["habito__DRINK"] == pytest.approx(0.5)
    assert third["picante_medio"] == pytest.approx(4 / 3)  # m1, m2 y m1
    assert third["tasa_compartir"] == pytest.approx(0.5)
    assert third["log_ticket_medio"] == pytest.approx((np.log(100) + np.log(200)) / 2)
    assert third["dias_desde_ultima"] == pytest.approx(7.0)


def test_profile_does_not_change_when_the_current_visit_changes(world) -> None:
    """Prueba de fuga: lo que pasa en la visita no puede estar en su propio perfil."""
    prior = pd.DataFrame([[1 / 3] * 3], index=["d1"], columns=CUISINES)
    base = point_in_time_profiles(
        visit_events(**world), cuisines=CUISINES, prior=prior, prior_strength=1.0
    )
    oi = world["order_items"].copy()
    oi.loc[oi["session_id"] == "s3", "menu_item_id"] = "m2"
    changed = {
        **world,
        "order_items": oi,
        "sessions": world["sessions"].assign(
            cocina=["mexicana", "pizza", "tacos_tortas"], ticket_por_persona=[100.0, 200.0, 9e3]
        ),
    }
    other = point_in_time_profiles(
        visit_events(**changed), cuisines=CUISINES, prior=prior, prior_strength=1.0
    )
    row = base["session_id"] == "s3"
    cols = [c for c in base.columns if c not in ("diner_id", "session_id", "opened_at")]
    pd.testing.assert_frame_equal(base.loc[row, cols], other.loc[row, cols])


def test_cuisine_prior_blends_declared_favorites_with_the_population() -> None:
    profiles = pd.DataFrame(
        {
            "diner_id": ["d1", "d2"],
            "completed": [True, False],
            "favorite_cuisines": [["pizza", "desconocida"], []],
        }
    )
    share = pd.Series([0.6, 0.2, 0.2], index=CUISINES)
    out = cuisine_prior(profiles, share, declared_weight=0.5)
    assert out.loc["d2"].tolist() == pytest.approx([0.6, 0.2, 0.2])
    assert out.loc["d1", "pizza"] == pytest.approx(0.5 * 0.2 + 0.5 * 1.0)
    assert out.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_group_aggregate_gives_mean_min_and_blend() -> None:
    out = group_aggregate(
        pd.Series([1.0, 3.0, 2.0]), pd.Series(["t1", "t1", "t2"]), least_misery_weight=0.25
    )
    assert out.loc["t1"].tolist() == pytest.approx([2.0, 1.0, 0.75 * 2.0 + 0.25 * 1.0])
    assert out.loc["t2", "media"] == 2.0


def test_table_profile_counts_known_members_and_unites_allergies() -> None:
    guests = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1", "s2"],
            "diner_id": ["d1", "d2", None, "d1"],
        }
    )
    member = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2"],
            "diner_id": ["d1", "d2", "d1"],
            "tasa_compartir": [0.2, 0.6, np.nan],
        }
    )
    allergies = pd.Series({"d1": ["gluten"], "d2": ["lacteos", "gluten"]})
    out = table_profile(guests, member, allergies).set_index("session_id")
    assert out.loc["s1", "personas"] == 3 and out.loc["s1", "con_perfil"] == 2
    assert out.loc["s1", "pct_con_perfil"] == pytest.approx(2 / 3)
    assert out.loc["s1", "tasa_compartir_media"] == pytest.approx(0.4)
    assert out.loc["s1", "tasa_compartir_max"] == pytest.approx(0.6)
    assert out.loc["s1", "alergias"] == ["gluten", "lacteos"]
    assert np.isnan(out.loc["s2", "tasa_compartir_media"])  # primera visita: sin historia


def test_recovery_by_history_shows_signal_growing_with_visits() -> None:
    rng = np.random.default_rng(0)
    n = 3_000
    latent = rng.normal(size=n)
    prev = rng.integers(0, 20, n)
    noise = rng.normal(size=n) * 3 / np.sqrt(prev + 1)
    profiles = pd.DataFrame(
        {"diner_id": np.arange(n), "visitas_previas": prev, "obs": latent + noise}
    )
    truth = pd.DataFrame({"lat": latent}, index=np.arange(n))
    out = recovery_by_history(profiles, truth, [("obs", "lat")], bins=[0, 1, 5, 20])
    rho = out.set_index("visitas_previas")["rho"]
    assert rho.is_monotonic_increasing
    assert set(out.columns) >= {"observado", "latente", "visitas_previas", "n", "rho"}
