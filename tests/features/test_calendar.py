import pandas as pd

from puzzle_recsys.features.calendar import MEAL_PERIODS, meal_period


def test_meal_period_follows_mexican_meal_schedule() -> None:
    hours = pd.Series([8, 11, 12, 16, 17, 18, 22, 3])
    out = meal_period(hours)
    assert out.tolist() == [
        "desayuno",
        "desayuno",
        "comida",
        "comida",
        "comida",
        "cena",
        "cena",
        "cena",  # madrugada: se agrupa con la cena (último servicio)
    ]
    assert list(out.cat.categories) == list(MEAL_PERIODS)
