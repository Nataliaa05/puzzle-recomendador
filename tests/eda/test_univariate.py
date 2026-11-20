import numpy as np
import pandas as pd
import pytest

from puzzle_recsys.eda.univariate import describe_numeric, frequency_table, ticket_per_session


def test_describe_numeric_flags_right_skew_and_log_helps() -> None:
    rng = np.random.default_rng(0)
    x = pd.Series(rng.lognormal(5, 0.8, 20_000))
    out = describe_numeric(x)
    assert out["n"] == 20_000
    assert out["asimetria"] > 1
    assert abs(out["asimetria_log"]) < 0.2  # el log de una lognormal es normal
    assert out["mediana"] < out["media"]  # cola derecha jala la media
    assert out["p_normalidad"] < 0.001


def test_describe_numeric_does_not_reject_a_normal_sample() -> None:
    rng = np.random.default_rng(1)
    out = describe_numeric(pd.Series(rng.normal(0, 1, 3000)))
    assert abs(out["asimetria"]) < 0.2
    assert out["p_normalidad"] > 0.01


def test_describe_numeric_ignores_nulls_and_skips_log_for_non_positive() -> None:
    out = describe_numeric(pd.Series([-1.0, 0.0, 1.0, 2.0, None, 3.0, 4.0, 5.0, 6.0]))
    assert out["n"] == 8
    assert np.isnan(out["asimetria_log"])


def test_frequency_table_groups_tail_into_other() -> None:
    s = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"] + ["d"])
    out = frequency_table(s, top=2)
    assert out.index.tolist() == ["a", "b", "otras"]
    assert out["n"].tolist() == [5, 3, 2]
    assert out["%"].sum() == pytest.approx(100)
    assert out["% acumulado"].iloc[-1] == pytest.approx(100)


def test_frequency_table_counts_missing_explicitly() -> None:
    out = frequency_table(pd.Series(["a", None, "a"]))
    assert out.loc["(faltante)", "n"] == 1


def test_ticket_per_session_sums_quantity_times_price() -> None:
    lines = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2"],
            "quantity": [2, 1, 3],
            "unit_price_cents": [10_000, 5_000, 2_000],
        }
    )
    out = ticket_per_session(lines)
    assert out.loc["s1"] == 250.0  # en pesos
    assert out.loc["s2"] == 60.0


def test_frequency_table_cumulative_follows_requested_order() -> None:
    """Regresión: al reordenar (días, estrellas) el acumulado seguía el orden por frecuencia."""
    s = pd.Series([1] * 1 + [2] * 3 + [3] * 6)
    out = frequency_table(s, order=[1, 2, 3])
    assert out.index.tolist() == [1, 2, 3]
    assert out["% acumulado"].tolist() == pytest.approx([10, 40, 100])
