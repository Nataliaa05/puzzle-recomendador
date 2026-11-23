import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from puzzle_recsys.explainability.ground_truth_check import (
    observed_signs,
    true_components,
    within_query_spread,
)
from puzzle_recsys.explainability.shap_analysis import group_importance, tree_shap


def test_true_components_follow_the_simulated_utility() -> None:
    cands = pd.DataFrame(
        {"query_id": [0, 0], "diner_id": ["d1", "d1"], "menu_item_id": ["m1", "m2"]}
    )
    tastes = pd.DataFrame(
        {
            "cuisine_affinity": [np.array([0.75, 0.25])],
            "protein_affinity": [np.array([0.5, 0.5])],
            "ingredient_pref": [np.array([1, -1, 0])],
            "spice_tolerance": [1],
        },
        index=["d1"],
    )
    items = pd.DataFrame(
        {
            "cuisine_idx": [0, -1],
            "protein_idx": [1, -1],
            "ingredients": [[0, 2], [1]],
            "spice": [3, 0],
            "appeal": [0.2, -0.1],
            "quality": [0.1, 0.0],
        },
        index=["m1", "m2"],
    )
    w = {
        "affinity_weight": 1.0,
        "protein_weight": 0.6,
        "ingredient_weight": 0.5,
        "spice_penalty": 1.2,
    }
    out = true_components(cands, tastes, items, w).set_index("menu_item_id")
    assert out.loc["m1", "cocina"] == pytest.approx(np.log(2 * 0.75))
    assert out.loc["m2", "cocina"] == 0.0  # sin cocina: neutral
    assert out.loc["m1", "proteina"] == pytest.approx(0.6 * np.log(2 * 0.5), abs=1e-6)
    assert out.loc["m1", "ingredientes"] == pytest.approx(0.5 * 1)
    assert out.loc["m2", "ingredientes"] == pytest.approx(-0.5)
    assert out.loc["m1", "picante"] == pytest.approx(-1.2 * 2)
    assert out.loc["m1", "atractivo_calidad"] == pytest.approx(0.3)


def test_within_query_spread_ignores_what_is_constant_in_a_query() -> None:
    df = pd.DataFrame(
        {"query_id": [0, 0, 1, 1], "fijo": [5, 5, 1, 1], "varia": [0.0, 2.0, 1.0, 3.0]}
    )
    out = within_query_spread(df, ["fijo", "varia"])
    assert out["fijo"] == 0.0 and out["varia"] == pytest.approx(1.0)


def test_observed_signs_read_the_direction_of_each_feature() -> None:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {"sube": rng.normal(size=500), "baja": rng.normal(size=500), "nada": rng.normal(size=500)}
    )
    shap = pd.DataFrame(
        {"sube": x["sube"] * 2, "baja": -x["baja"], "nada": rng.normal(0, 1e-6, 500) * 0}
    )
    out = observed_signs(shap, x).set_index("feature")["signo"]
    assert out["sube"] == "+" and out["baja"] == "-" and out["nada"] == "0"


def test_tree_shap_values_add_up_to_the_prediction() -> None:
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = (x["a"] > 0).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1}, lgb.Dataset(x, y), num_boost_round=20
    )
    shap, base = tree_shap(booster, x)
    assert np.allclose(shap.sum(axis=1) + base, booster.predict(x, raw_score=True))
    imp = group_importance(shap, {"A": ["a"], "B": ["b"]})
    assert imp["A"] > imp["B"]


def test_truth_items_uses_the_simulator_indices() -> None:
    from puzzle_recsys.explainability.ground_truth_check import truth_items

    menu = pd.DataFrame(
        {
            "menu_item_id": ["m1", "m2"],
            "canonical_true": ["tacos", "agua"],
            "protein": ["PORK", "NONE"],
            "spice_level": [2, 0],
        }
    )
    out = truth_items(
        menu,
        pd.Series({"tacos": "tacos_tortas", "agua": "general"}),
        pd.DataFrame({"menu_item_id": ["m1", "m1"], "value": ["cerdo", "tortilla"]}),
        pd.Series({"tacos": 0.5, "agua": 0.0}),
        pd.Series({"m1": 0.1, "m2": -0.2}),
        cuisines=["mexicana", "tacos_tortas"],
        ingredient_names=["tortilla", "cerdo"],
    )
    assert out.loc["m1", "cuisine_idx"] == 1 and out.loc["m2", "cuisine_idx"] == -1
    assert out.loc["m1", "protein_idx"] == 1 and out.loc["m2", "protein_idx"] == -1
    assert out.loc["m1", "ingredients"] == [1, 0] and out.loc["m2", "ingredients"] == []
    assert out.loc["m1", "appeal"] == 0.5 and out.loc["m2", "quality"] == -0.2


def test_relative_to_query_removes_what_is_shared_by_the_whole_query() -> None:
    from puzzle_recsys.explainability.shap_analysis import relative_to_query, top_reasons

    values = pd.DataFrame({"constante": [-1.0, -1.0, 2.0, 2.0], "distingue": [0.5, -0.5, 0.0, 1.0]})
    out = relative_to_query(values, np.array([0, 0, 1, 1]))
    assert (out["constante"] == 0).all()
    assert out["distingue"].tolist() == [0.5, -0.5, -0.5, 0.5]
    assert top_reasons(pd.Series({"a": 0.1, "b": -0.3, "c": 0.2}), n=2) == "b (-0.30), c (+0.20)"
