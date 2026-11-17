import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from puzzle_recsys.simulation.distributions import allocate_proportional, stratified_sample

sizes_strategy = st.lists(st.integers(min_value=1, max_value=60), min_size=1, max_size=8)


@given(sizes=sizes_strategy, data=st.data())
@settings(max_examples=200, deadline=None)
def test_allocation_sums_to_n_and_respects_capacity(sizes: list[int], data: st.DataObject) -> None:
    n = data.draw(st.integers(min_value=0, max_value=sum(sizes)))
    alloc = allocate_proportional(np.array(sizes), n)
    assert alloc.sum() == n
    assert (alloc >= 0).all()
    assert (alloc <= np.array(sizes)).all()


def test_allocation_is_proportional_when_capacity_allows() -> None:
    alloc = allocate_proportional(np.array([600, 300, 100]), 100)
    assert alloc.tolist() == [60, 30, 10]


def test_allocation_rejects_n_above_population() -> None:
    with pytest.raises(ValueError, match="mayor"):
        allocate_proportional(np.array([2, 3]), 6)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(1000),
            "scian": ["722511"] * 700 + ["722514"] * 300,
            "alcaldia": (["002", "015"] * 500),
        }
    )


def test_stratified_sample_preserves_proportions_and_uniqueness() -> None:
    df = _frame()
    out = stratified_sample(df, 100, ["scian"], np.random.default_rng(0))
    assert len(out) == 100
    assert out["id"].is_unique
    assert (out["scian"] == "722511").sum() == 70


def test_stratified_sample_is_reproducible_with_same_seed() -> None:
    df = _frame()
    a = stratified_sample(df, 50, ["scian", "alcaldia"], np.random.default_rng(42))
    b = stratified_sample(df, 50, ["scian", "alcaldia"], np.random.default_rng(42))
    assert a["id"].tolist() == b["id"].tolist()


def test_make_rng_streams_are_reproducible_and_independent() -> None:
    from puzzle_recsys.simulation.distributions import make_rng

    a1 = make_rng(7, "menus").random(5)
    a2 = make_rng(7, "menus").random(5)
    b = make_rng(7, "world").random(5)
    np.testing.assert_array_equal(a1, a2)
    assert not np.allclose(a1, b)
