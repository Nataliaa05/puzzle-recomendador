import numpy as np
import pandas as pd
import pytest

from puzzle_recsys.validation.synthetic_checks import (
    gini,
    in_band,
    repeat_rate,
    top_share,
    total_variation,
)


def test_top_share_and_gini_on_known_cases() -> None:
    equal = pd.Series([10] * 10)
    assert top_share(equal, 0.2) == pytest.approx(0.2)
    assert gini(equal) == pytest.approx(0.0, abs=1e-9)
    skewed = pd.Series([100] + [0] * 9)
    assert top_share(skewed, 0.2) == pytest.approx(1.0)
    assert gini(skewed) > 0.85


def test_total_variation_bounds() -> None:
    p = np.array([0.5, 0.5])
    assert total_variation(p, p) == 0.0
    assert total_variation(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0


def test_repeat_rate_counts_reorders_of_the_same_canonical() -> None:
    orders = pd.DataFrame(
        {
            "diner": ["a", "a", "a", "b", "b"],
            "canonical": ["tacos", "tacos", "pizza", "sushi", "ramen"],
        }
    )
    # a: 3 pedidos, 1 repetido (el segundo tacos) · b: 2 pedidos, 0 repetidos → 1/5
    assert repeat_rate(orders, "diner", "canonical") == pytest.approx(0.2)


def test_in_band() -> None:
    assert in_band(0.5, (0.4, 0.6))
    assert not in_band(0.7, (0.4, 0.6))


def test_stylized_facts_on_a_tiny_world() -> None:
    from puzzle_recsys.validation.synthetic_checks import stylized_facts

    sessions = pd.DataFrame({"session_id": ["s1", "s2", "s3"], "restaurant_id": ["r1", "r1", "r2"]})
    guests = pd.DataFrame(
        {
            "guest_id": ["g1", "g2", "g3", "g4"],
            "session_id": ["s1", "s2", "s3", "s3"],
            "is_host": [True, True, True, False],
            "taste_ref": ["d1", "d1", "d2", "a1"],
        }
    )
    lines = pd.DataFrame(
        {
            "guest_id": ["g1", "g2", "g3", "g4"],
            "session_id": ["s1", "s2", "s3", "s3"],
            "menu_item_id": ["m1", "m1", "m2", "m3"],
            "is_shared": [False, False, False, True],
        }
    )
    facts = stylized_facts(
        sessions,
        guests,
        lines,
        canonical_of={"m1": "tacos", "m2": "sushi", "m3": "guacamole"},
        restaurant_cuisine={"r1": "mexicana", "r2": "japonesa"},
        favorite_cuisine={"d1": "mexicana", "d2": "italiana"},
    )
    assert facts["favorite_cuisine_share"] == pytest.approx(2 / 3)
    assert facts["repeat_rate_canonical"] == pytest.approx(1 / 3)  # d1 repite tacos
    assert facts["shared_table_rate"] == pytest.approx(1 / 3)


def test_stylized_facts_respect_the_time_window() -> None:
    from puzzle_recsys.validation.synthetic_checks import stylized_facts

    ts = pd.to_datetime(["2025-01-10", "2025-02-10", "2025-06-10"], utc=True)
    sessions = pd.DataFrame(
        {"session_id": ["s1", "s2", "s3"], "restaurant_id": ["r1"] * 3, "opened_at": ts}
    )
    guests = pd.DataFrame(
        {
            "guest_id": ["g1", "g2", "g3"],
            "session_id": ["s1", "s2", "s3"],
            "is_host": [True] * 3,
            "taste_ref": ["d1"] * 3,
        }
    )
    lines = pd.DataFrame(
        {
            "guest_id": ["g1", "g2", "g3"],
            "session_id": ["s1", "s2", "s3"],
            "menu_item_id": ["m1", "m2", "m1"],
            "is_shared": [False] * 3,
        }
    )
    kwargs = dict(
        canonical_of={"m1": "tacos", "m2": "sushi"},
        restaurant_cuisine={"r1": "mexicana"},
        favorite_cuisine={"d1": "mexicana"},
    )
    full = stylized_facts(sessions, guests, lines, **kwargs)
    window = stylized_facts(
        sessions, guests, lines, window_end=pd.Timestamp("2025-04-01", tz="UTC"), **kwargs
    )
    assert full["repeat_rate_canonical"] == pytest.approx(1 / 3)  # tacos se repite en junio
    assert window["repeat_rate_canonical"] == 0.0  # en la ventana aún no hay recompra


def test_repeat_rate_ignores_anonymous_companions() -> None:
    """Regresión: los perfiles anónimos se reutilizan entre mesas y simulaban 'recompra'."""
    from puzzle_recsys.validation.synthetic_checks import stylized_facts

    sessions = pd.DataFrame({"session_id": ["s1", "s2"], "restaurant_id": ["r1", "r1"]})
    guests = pd.DataFrame(
        {
            "guest_id": ["g1", "g2", "g3", "g4"],
            "session_id": ["s1", "s1", "s2", "s2"],
            "is_host": [True, False, True, False],
            "taste_ref": ["d1", "a7", "d2", "a7"],  # el mismo perfil anónimo en dos mesas
        }
    )
    lines = pd.DataFrame(
        {
            "guest_id": ["g1", "g2", "g3", "g4"],
            "session_id": ["s1", "s1", "s2", "s2"],
            "menu_item_id": ["m1", "m2", "m3", "m2"],
            "is_shared": [False] * 4,
        }
    )
    facts = stylized_facts(
        sessions,
        guests,
        lines,
        canonical_of={"m1": "tacos", "m2": "pizza", "m3": "sushi"},
        restaurant_cuisine={"r1": "mexicana"},
        favorite_cuisine={"d1": "mexicana", "d2": "mexicana"},
    )
    assert facts["repeat_rate_canonical"] == 0.0  # d1 y d2 nunca repiten
