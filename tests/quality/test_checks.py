import numpy as np
import pandas as pd
import pytest

from puzzle_recsys.quality.checks import (
    completeness,
    duplicate_keys,
    missingness_test,
    orphan_keys,
    out_of_range,
    price_consistency,
    robust_outliers,
)


def test_completeness_is_share_of_non_null() -> None:
    df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
    out = completeness(df)
    assert out["a"] == 0.5
    assert out["b"] == 1.0


def test_duplicate_keys_counts_extra_rows() -> None:
    df = pd.DataFrame({"id": ["x", "x", "y", "z", "z", "z"]})
    assert duplicate_keys(df, ["id"]) == 3


def test_orphan_keys_ignores_nulls() -> None:
    child = pd.DataFrame({"session_id": ["s1", "s2", None, "s9"]})
    parent = pd.DataFrame({"session_id": ["s1", "s2"]})
    assert orphan_keys(child, "session_id", parent, "session_id") == 1


def test_out_of_range() -> None:
    assert out_of_range(pd.Series([0, 1, 5, 6, None]), 1, 5) == 2


def test_robust_outliers_flags_giant_tables_only() -> None:
    pax = pd.Series([2, 2, 3, 4, 2, 1, 3, 2, 25, 4, 2, 3])
    flags = robust_outliers(pax)
    assert flags.tolist().count(True) == 1
    assert flags.iloc[8]


def test_price_consistency_uses_price_valid_at_order_time() -> None:
    history = pd.DataFrame(
        {
            "menu_item_id": ["m1", "m1"],
            "price_cents": [10000, 12000],
            "valid_from": pd.to_datetime(["2025-01-01", "2025-03-01"], utc=True),
        }
    )
    lines = pd.DataFrame(
        {
            "menu_item_id": ["m1", "m1", "m1"],
            "unit_price_cents": [10000, 12000, 10000],
            "created_at": pd.to_datetime(["2025-02-01", "2025-04-01", "2025-04-02"], utc=True),
        }
    )
    assert price_consistency(lines, history) == pytest.approx(2 / 3)


def test_missingness_detects_dependence_on_an_observed_variable() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(0, 1, n)
    missing = rng.random(n) < np.where(x > 0, 0.6, 0.1)  # falta más cuando x es alto
    df = pd.DataFrame({"x": x, "target": np.where(missing, np.nan, 1.0)})
    assert missingness_test(df, "target", "x") < 0.001
    random_missing = rng.random(n) < 0.3
    df2 = pd.DataFrame({"x": x, "target": np.where(random_missing, np.nan, 1.0)})
    assert missingness_test(df2, "target", "x") > 0.01


def test_price_consistency_tolerates_different_time_resolutions() -> None:
    """Regresión: Parquet devolvió una tabla en µs y otra en ns y merge_asof falló."""
    history = pd.DataFrame(
        {
            "menu_item_id": ["m1"],
            "price_cents": [10000],
            "valid_from": pd.to_datetime(["2025-01-01"], utc=True).as_unit("us"),
        }
    )
    lines = pd.DataFrame(
        {
            "menu_item_id": ["m1"],
            "unit_price_cents": [10000],
            "created_at": pd.to_datetime(["2025-02-01"], utc=True).as_unit("ns"),
        }
    )
    assert price_consistency(lines, history) == 1.0
