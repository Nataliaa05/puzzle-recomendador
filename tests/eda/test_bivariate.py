import numpy as np
import pandas as pd
import pytest

from puzzle_recsys.eda.bivariate import cramers_v, effect_label, kruskal_effect, spearman_pairs


def test_spearman_pairs_detects_monotonic_relation() -> None:
    rng = np.random.default_rng(0)
    x = rng.lognormal(0, 1, 2000)
    y = np.exp(x) + rng.normal(0, 0.1, 2000)
    df = pd.DataFrame({"x": x, "y": y, "z": rng.normal(size=2000)})
    out = spearman_pairs(df).set_index(["a", "b"])
    assert out.loc[("x", "y"), "rho"] > 0.95  # monótona aunque no lineal
    assert abs(out.loc[("x", "z"), "rho"]) < 0.1
    assert len(out) == 3  # pares sin repetir


def test_kruskal_effect_size_grows_with_group_separation() -> None:
    rng = np.random.default_rng(1)
    g = np.repeat(["a", "b", "c"], 1000)
    weak = pd.DataFrame({"v": rng.normal(0, 1, 3000) + (g == "c") * 0.1, "g": g})
    strong = pd.DataFrame({"v": rng.normal(0, 1, 3000) + (g == "c") * 3, "g": g})
    w, s = kruskal_effect(weak, "v", "g"), kruskal_effect(strong, "v", "g")
    assert s["epsilon2"] > 0.4 > w["epsilon2"]
    assert 0 <= w["epsilon2"] <= 1
    assert s["p"] < 0.001


def test_cramers_v_is_one_for_perfect_association_and_near_zero_for_independence() -> None:
    rng = np.random.default_rng(2)
    a = pd.Series(rng.choice(["x", "y", "z"], 5000))
    assert cramers_v(a, a.map({"x": 1, "y": 2, "z": 3}))["V"] == pytest.approx(1, abs=0.01)
    b = pd.Series(rng.choice(["p", "q"], 5000))
    assert cramers_v(a, b)["V"] < 0.05


def test_effect_label_uses_cohen_style_thresholds() -> None:
    assert effect_label(0.05, "rho") == "despreciable"
    assert effect_label(0.35, "rho") == "moderado"
    assert effect_label(0.6, "rho") == "fuerte"
    assert effect_label(0.02, "epsilon2") == "pequeño"
