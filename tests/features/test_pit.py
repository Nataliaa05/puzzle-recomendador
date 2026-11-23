import numpy as np
import pandas as pd

from puzzle_recsys.features.pit import count_before, state_at, to_seconds


def _t(*stamps: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(stamps), utc=True, format="ISO8601"))


def test_count_before_counts_only_strictly_earlier_events_of_the_same_key() -> None:
    events = pd.DataFrame(
        {
            "item": ["a", "a", "a", "b"],
            "t": _t("2025-01-01", "2025-01-02", "2025-01-03", "2025-01-01"),
        }
    )
    queries = pd.DataFrame(
        {
            "item": ["a", "a", "b", "c"],
            "t": _t("2025-01-02", "2025-01-05", "2025-01-01", "2025-01-09"),
        }
    )
    out = count_before(events, queries, keys=["item"])
    assert out.tolist() == [1, 3, 0, 0]  # el evento del mismo instante NO cuenta


def test_count_before_supports_weights_and_several_keys() -> None:
    events = pd.DataFrame(
        {
            "diner": ["d1", "d1", "d1", "d2"],
            "dish": ["x", "x", "y", "x"],
            "t": _t("2025-01-01", "2025-01-02", "2025-01-02", "2025-01-01"),
            "w": [1.0, 2.0, 5.0, 7.0],
        }
    )
    queries = pd.DataFrame(
        {"diner": ["d1", "d2"], "dish": ["x", "x"], "t": _t("2025-02-01", "2025-02-01")}
    )
    out = count_before(events, queries, keys=["diner", "dish"], weight="w")
    assert out.tolist() == [3.0, 7.0]


def test_state_at_returns_the_latest_value_at_or_before_the_query() -> None:
    events = pd.DataFrame(
        {
            "item": ["a", "a", "a"],
            "t": _t("2025-01-01", "2025-01-05", "2025-01-06"),
            "available": [True, False, True],
        }
    )
    queries = pd.DataFrame(
        {
            "item": ["a", "a", "a", "a", "z"],
            "t": _t("2024-12-31", "2025-01-01", "2025-01-05 12:00", "2025-01-07", "2025-01-07"),
        }
    )
    out = state_at(events, queries, keys=["item"], value="available")
    assert out.iloc[0] is None or pd.isna(out.iloc[0])  # antes de existir
    assert out.iloc[1:4].tolist() == [True, False, True]  # el evento del mismo instante SÍ aplica
    assert pd.isna(out.iloc[4])


def test_to_seconds_is_timezone_safe() -> None:
    utc = pd.Series(pd.to_datetime(["2025-01-01 06:00"], utc=True))
    local = utc.dt.tz_convert("America/Mexico_City")
    assert np.array_equal(to_seconds(utc), to_seconds(local))
