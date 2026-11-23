import numpy as np
import pandas as pd
import pytest

from puzzle_recsys.catalog.dish_homologation import (
    DishHomologator,
    accuracy_by_group,
    clean_dish_name,
    simulate_confirmation,
    split_aliases,
    threshold_table,
    uses_unseen_alias,
)

STOP = ["de la casa", "especial", "premium"]


def test_clean_dish_name_removes_menu_filler_but_never_empties_a_name() -> None:
    assert clean_dish_name("Tacos al Pastor ESPECIAL", STOP) == "tacos al pastor"
    assert clean_dish_name("Sopa de la casa", STOP) == "sopa"
    assert clean_dish_name("Especial", STOP) == "especial"  # solo relleno: se conserva


def test_split_aliases_keeps_every_canonical_name_known() -> None:
    aliases = pd.DataFrame(
        {"canonical_id": ["a", "a", "b", "b"], "alias": ["A1", "A2", "B1", "B2"]}
    )
    known, unseen = split_aliases(aliases, known_share=0.5, rng=np.random.default_rng(0))
    assert len(known) + len(unseen) == 4
    assert set(known["alias"]).isdisjoint(unseen["alias"])


@pytest.fixture
def reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_id": ["tacos_pastor", "pizza_mexicana", "huevos_rancheros", "limonada"],
            "name": ["Tacos al pastor", "Pizza mexicana", "Huevos rancheros", "Limonada"],
            "course": ["MAIN", "MAIN", "MAIN", "DRINK"],
            "cuisine": ["tacos_tortas", "pizza", "mexicana", "general"],
        }
    )


def test_homologator_matches_noisy_names_within_the_same_course(reference) -> None:
    model = DishHomologator(stop_words=STOP).fit(reference)
    out = model.predict(
        pd.DataFrame(
            {
                "name": ["TACOS DE PASTOR especial", "Limonda", "Limonada"],
                "course": ["MAIN", "DRINK", "MAIN"],
                "cuisine": ["tacos_tortas", "general", "mexicana"],
            }
        )
    )
    assert out["canonical_pred"].tolist()[:2] == ["tacos_pastor", "limonada"]
    assert out["canonical_pred"].iloc[2] != "limonada"  # nunca cruza de tipo de plato
    assert out["confianza"].between(0, 1).all()


def test_restaurant_cuisine_breaks_ties_between_similar_names(reference) -> None:
    query = pd.DataFrame({"name": ["Ranchera"], "course": ["MAIN"], "cuisine": ["pizza"]})
    plain = DishHomologator(stop_words=STOP, cuisine_weight=0.0).fit(reference).predict(query)
    hinted = DishHomologator(stop_words=STOP, cuisine_weight=1.0).fit(reference).predict(query)
    assert plain["canonical_pred"].iloc[0] == "huevos_rancheros"
    assert hinted["canonical_pred"].iloc[0] == "pizza_mexicana"


def test_confirmation_is_decided_per_restaurant_and_fixes_reviewed_menus() -> None:
    pred = pd.Series(["a", "x", "b", "y"])
    truth = pd.Series(["a", "b", "b", "c"])
    restaurants = pd.Series(["r1", "r1", "r2", "r2"])
    out = simulate_confirmation(pred, truth, restaurants, rate=1.0, rng=np.random.default_rng(0))
    assert out["canonical_pred"].tolist() == ["a", "b", "b", "c"]
    assert out["canonical_source"].tolist() == ["CONFIRMED", "MANUAL", "CONFIRMED", "MANUAL"]
    none = simulate_confirmation(pred, truth, restaurants, rate=0.0, rng=np.random.default_rng(0))
    assert none["canonical_pred"].tolist() == pred.tolist()
    assert (none["canonical_source"] == "SUGGESTED").all()
    half = simulate_confirmation(
        pd.Series(["x"] * 400),
        pd.Series(["y"] * 400),
        pd.Series(np.repeat(np.arange(100), 4)),
        rate=0.5,
        rng=np.random.default_rng(1),
    )
    by_restaurant = half.groupby(np.repeat(np.arange(100), 4))["canonical_source"].nunique()
    assert (by_restaurant == 1).all()  # el restaurante revisa todo su menú o nada


def test_accuracy_by_group_reports_each_group_and_the_total() -> None:
    out = accuracy_by_group(
        pd.Series(["a", "b", "c", "d"]),
        pd.Series(["a", "x", "c", "d"]),
        pd.Series(["g1", "g1", "g2", "g2"]),
    ).set_index("grupo")
    assert out.loc["g1", "exactitud"] == 0.5
    assert out.loc["g2", "exactitud"] == 1.0
    assert out.loc["total", "n"] == 4
    assert out.loc["total", "exactitud"] == 0.75


def test_uses_unseen_alias_detects_alias_inside_a_decorated_name() -> None:
    out = uses_unseen_alias(
        pd.Series(["AL PASTOR especial", "Tacos al pastor", "Limonda"]), ["al pastor", "limon"]
    )
    assert out.tolist() == [True, True, False]  # "Limonda" no contiene "limon" como palabra


def test_threshold_table_trades_coverage_for_accuracy() -> None:
    conf = pd.Series([0.9, 0.8, 0.4, 0.2])
    hit = pd.Series([True, True, False, True])
    out = threshold_table(conf, hit, [0.0, 0.5]).set_index("umbral")
    assert out.loc[0.0, "cobertura"] == 1.0 and out.loc[0.0, "exactitud"] == 0.75
    assert out.loc[0.5, "cobertura"] == 0.5 and out.loc[0.5, "exactitud"] == 1.0
