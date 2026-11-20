from types import SimpleNamespace

import numpy as np

from puzzle_recsys.simulation.availability import demand_driven_sold_out_rates


def _index(appeal, rest_items):
    items = SimpleNamespace(appeal=np.asarray(appeal, float), quality=np.zeros(len(appeal)))
    return SimpleNamespace(items=items, rest_items=[np.asarray(r) for r in rest_items])


def test_popular_dishes_sell_out_more_and_mean_rate_is_kept() -> None:
    index = _index([2.0, 0.0, -1.0, -2.0], [[0, 1, 2, 3]])
    rates = demand_driven_sold_out_rates(index, 0.02, max_rate=0.5)
    assert rates[0] > rates[1] > rates[2] > rates[3]
    assert np.isclose(rates.mean(), 0.02)


def test_rates_are_relative_within_each_restaurant() -> None:
    # Mismo atractivo relativo en dos restaurantes → mismas tasas, aunque el nivel cambie.
    index = _index([1.0, 0.0, 6.0, 5.0], [[0, 1], [2, 3]])
    rates = demand_driven_sold_out_rates(index, 0.02, max_rate=0.5)
    assert np.allclose(rates[:2], rates[2:])


def test_max_rate_caps_the_most_popular_dish() -> None:
    index = _index([8.0, 0.0, 0.0, 0.0], [[0, 1, 2, 3]])
    rates = demand_driven_sold_out_rates(index, 0.05, max_rate=0.15)
    assert rates.max() <= 0.15
