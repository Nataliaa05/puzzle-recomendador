import numpy as np
import pytest

from puzzle_recsys.features.calendar import period_index
from puzzle_recsys.simulation.context import (
    course_rate,
    cuisine_period_log,
    dish_period_bonus,
    expected_pax_distribution,
    pax_probabilities,
)

CTX = {
    "weight": 1.0,
    "cuisine_period": {"cafeteria": [3.0, 0.8, 0.5]},
    "course_rate_period": {"desayuno": {"DRINK": 1.5, "STARTER": 0.5}},
    "breakfast_dishes": ["chilaquiles"],
    "breakfast_bonus": {"desayuno": 1.0, "comida": -0.3, "cena": -0.6},
    "alcohol_ingredients": ["cerveza"],
    "alcohol_bonus": {"desayuno": -2.5, "comida": 0.0, "cena": 0.5},
}
PARTY = {"large_from": 3, "weekend_factor": 1.5, "breakfast_factor": 0.5}


def test_period_index_matches_meal_schedule() -> None:
    assert period_index(np.array([8, 11, 12, 17, 18, 23, 2])).tolist() == [0, 0, 1, 1, 2, 2, 2]


def test_cuisine_period_log_defaults_to_neutral() -> None:
    out = cuisine_period_log(["mexicana", "cafeteria"], CTX)
    assert out.shape == (3, 2)
    assert np.allclose(out[:, 0], 0.0)  # sin multiplicador declarado: neutral
    assert out[0, 1] == pytest.approx(np.log(3.0))
    assert out[2, 1] < 0


def test_course_rate_scales_and_caps_probability() -> None:
    assert course_rate(0.4, "STARTER", 0, CTX) == pytest.approx(0.2)
    assert course_rate(0.9, "DRINK", 0, CTX) == pytest.approx(0.98)  # tope
    assert course_rate(0.4, "STARTER", 2, CTX) == pytest.approx(0.4)  # franja sin ajuste


class _Dish:
    def __init__(self, slug, ingredients):
        self.slug, self.ingredients = slug, ingredients


class _Library:
    dishes = (
        _Dish("chilaquiles", ["totopos"]),
        _Dish("cerveza", ["cerveza"]),
        _Dish("mole", ["mole"]),
    )


def test_dish_period_bonus_marks_breakfast_and_alcohol() -> None:
    out = dish_period_bonus(["mole", "chilaquiles", "cerveza", "chilaquiles"], _Library(), CTX)
    assert out.shape == (4, 3)
    assert out[0].tolist() == [0.0, 0.0, 0.0]
    assert out[1].tolist() == [1.0, -0.3, -0.6]
    assert out[2].tolist() == [-2.5, 0.0, 0.5]
    assert out[3].tolist() == out[1].tolist()


def test_pax_probabilities_shift_mass_to_large_tables_on_weekends_only() -> None:
    values, probs = np.array([1, 2, 3, 4]), np.array([0.25, 0.25, 0.25, 0.25])
    base = pax_probabilities(values, probs, period=1, weekend=False, cfg=PARTY)
    weekend = pax_probabilities(values, probs, period=1, weekend=True, cfg=PARTY)
    breakfast = pax_probabilities(values, probs, period=0, weekend=False, cfg=PARTY)
    assert np.allclose(base, probs)
    assert weekend.sum() == pytest.approx(1) and breakfast.sum() == pytest.approx(1)
    assert weekend[2:].sum() > 0.5 > breakfast[2:].sum()


def test_expected_pax_distribution_is_the_mixture_over_days_and_hours() -> None:
    values, probs = np.array([1, 2, 3, 4]), np.array([0.25, 0.25, 0.25, 0.25])
    weekdays = [1, 1, 1, 1, 0, 0, 0]  # solo entre semana
    hours = {"14": 1.0}  # solo comida
    out = expected_pax_distribution(values, probs, weekdays, hours, PARTY)
    assert np.allclose(out, probs)  # sin fin de semana ni desayuno: la base
    out2 = expected_pax_distribution(values, probs, [0, 0, 0, 0, 1, 1, 1], hours, PARTY)
    assert out2[2:].sum() > 0.5


def test_context_config_only_names_real_dishes_cuisines_and_ingredients() -> None:
    """Un error de dedo en la configuración dejaría el efecto apagado sin avisar."""
    from puzzle_recsys.catalog.library import load_library
    from puzzle_recsys.config import load_paths, load_simulation

    library = load_library(load_paths().external / "canonical_dishes.yaml")
    ctx = load_simulation().choice["context"]
    slugs = {d.slug for d in library.dishes}
    cuisines = {d.cuisine for d in library.dishes}
    assert set(ctx["breakfast_dishes"]) <= slugs
    assert set(ctx["cuisine_period"]) <= cuisines
    assert set(ctx["alcohol_ingredients"]) <= set(library.ingredients)
    assert all(len(v) == 3 and min(v) > 0 for v in ctx["cuisine_period"].values())
