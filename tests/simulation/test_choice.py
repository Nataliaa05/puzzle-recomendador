import numpy as np
import pytest

from puzzle_recsys.simulation.choice import (
    DinerVector,
    ItemFeatures,
    dish_utility,
    feasible,
    group_utility,
    gumbel_choice,
    restaurant_utility,
)

CFG = {
    "affinity_weight": 1.0,
    "protein_weight": 0.6,
    "ingredient_weight": 0.5,
    "spice_penalty": 1.2,
    "loyalty_weight": 0.4,
}


def _items() -> ItemFeatures:
    # 0: carbonara (italiana, cerdo, gluten+lácteos) · 1: ensalada (vegana) · 2: tacos (picante 3)
    return ItemFeatures(
        cuisine_idx=np.array([0, -1, 1]),
        protein_idx=np.array([1, -1, 1]),
        ingredients=np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=np.int8),
        spice=np.array([0, 0, 3]),
        course=np.array(["MAIN", "MAIN", "MAIN"]),
        shareable=np.array([False, True, True]),
        allergens=np.array([[True, True], [False, False], [False, False]]),
        vegetarian=np.array([False, True, False]),
        vegan=np.array([False, True, False]),
        appeal=np.zeros(3),
        quality=np.zeros(3),
    )


def _diner(**kw) -> DinerVector:
    base = dict(
        cuisine_affinity=np.array([0.8, 0.2]),
        protein_affinity=np.array([0.5, 0.5]),
        ingredient_pref=np.array([1, 0, -1]),
        spice_tolerance=1,
        price_sensitivity=1.0,
        budget=100.0,
        diet="ninguna",
        allergies=np.array([False, False]),
    )
    base.update(kw)
    return DinerVector(**base)


def test_allergens_are_a_hard_filter() -> None:
    mask = feasible(_items(), np.arange(3), _diner(allergies=np.array([True, False])))
    assert mask.tolist() == [False, True, True]


@pytest.mark.parametrize(
    ("diet", "expected"), [("vegano", [False, True, False]), ("vegetariano", [False, True, False])]
)
def test_diet_is_a_hard_filter(diet: str, expected: list[bool]) -> None:
    assert feasible(_items(), np.arange(3), _diner(diet=diet)).tolist() == expected


def test_spice_above_tolerance_lowers_utility() -> None:
    items, cand = _items(), np.array([2])
    prices, ref = np.array([100.0]), np.array([1.0])
    tolerant = dish_utility(items, cand, prices, ref, _diner(spice_tolerance=3), CFG)
    sensitive = dish_utility(items, cand, prices, ref, _diner(spice_tolerance=0), CFG)
    assert tolerant[0] - sensitive[0] == pytest.approx(CFG["spice_penalty"] * 3)


def test_price_far_from_ideal_is_less_attractive() -> None:
    items, cand = _items(), np.array([1, 1, 1])
    ideal = np.full(3, 200.0)
    u = dish_utility(
        items, cand, np.array([100.0, 200.0, 400.0]), ideal, _diner(price_sensitivity=2.0), CFG
    )
    assert u[1] > u[0]
    assert u[1] > u[2]
    assert u[0] == pytest.approx(u[2])  # simétrico en escala logarítmica


def test_budget_changes_which_dish_is_preferred() -> None:
    """Regresión: con -λ·log(precio/presupuesto) el presupuesto se cancelaba en el logit."""
    from puzzle_recsys.simulation.choice import ideal_prices

    items, cand = _items(), np.array([1, 1])
    prices = np.array([100.0, 300.0])
    typical, ref = np.full(2, 200.0), np.ones(2)

    def gap(budget: float) -> float:
        ideal = ideal_prices(typical, budget, ref, budget_weight=0.5)
        u = dish_utility(items, cand, prices, ideal, _diner(), CFG)
        return u[1] - u[0]  # ventaja del platillo caro

    assert gap(budget=600.0) > gap(budget=60.0)


def test_anchoring_to_an_expensive_menu() -> None:
    """Con el mismo presupuesto, en una carta cara el precio ideal sube."""
    from puzzle_recsys.simulation.choice import ideal_prices

    ref = np.ones(1)
    cheap = ideal_prices(np.array([120.0]), 150.0, ref, budget_weight=0.5)
    fancy = ideal_prices(np.array([600.0]), 150.0, ref, budget_weight=0.5)
    assert fancy[0] > 2 * cheap[0]


def test_gumbel_choice_matches_logit_probabilities() -> None:
    rng = np.random.default_rng(0)
    u = np.array([0.0, 1.0, 2.0])
    picks = np.bincount([gumbel_choice(u, 1.0, rng) for _ in range(20_000)], minlength=3) / 20_000
    expected = np.exp(u) / np.exp(u).sum()
    assert np.allclose(picks, expected, atol=0.015)


def test_group_utility_least_misery() -> None:
    m = np.array([[3.0, 1.0], [-3.0, 1.0]])
    assert group_utility(m, 0.0).tolist() == [0.0, 1.0]
    assert group_utility(m, 1.0).tolist() == [-3.0, 1.0]


def test_distance_sensitivity_penalizes_far_restaurants() -> None:
    u = restaurant_utility(
        np.array([0, 0]),
        np.array([1.0, 5.0]),
        np.array([100.0, 100.0]),
        np.array([0, 0]),
        _diner(),
        distance_sensitivity=0.5,
        variety_seeking=0.0,
        cfg=CFG,
    )
    assert u[0] - u[1] == pytest.approx(2.0)


def test_shared_utility_only_counts_people_who_can_eat_it() -> None:
    from puzzle_recsys.simulation.choice import shared_group_utility

    # 3 personas x 2 platillos. Platillo 0: la persona 2 es alérgica (no elegible).
    u = np.array([[2.0, 1.0], [2.0, 1.0], [-9.0, 1.0]])
    eligible = np.array([[True, True], [True, True], [False, True]])
    out = shared_group_utility(u, eligible, least_misery_weight=0.0, exclusion_penalty=0.0)
    assert out[0] == pytest.approx(2.0)  # el -9 de quien no lo come NO cuenta
    assert out[1] == pytest.approx(1.0)


def test_exclusion_has_a_social_cost() -> None:
    from puzzle_recsys.simulation.choice import shared_group_utility

    u = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    eligible = np.array([[True, True], [True, True], [False, True]])
    out = shared_group_utility(u, eligible, least_misery_weight=0.0, exclusion_penalty=0.6)
    assert out[0] == pytest.approx(1.0 - 0.6 / 3)  # 1 de 3 queda fuera
    assert out[1] == pytest.approx(1.0)


def test_a_shared_dish_needs_at_least_two_people() -> None:
    from puzzle_recsys.simulation.choice import shared_group_utility

    u = np.array([[5.0], [5.0], [5.0]])
    eligible = np.array([[True], [False], [False]])
    out = shared_group_utility(u, eligible, least_misery_weight=0.3, exclusion_penalty=0.5)
    assert np.isneginf(out[0])
