import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from puzzle_recsys.catalog.library import load_library
from puzzle_recsys.config import load_paths, load_simulation
from puzzle_recsys.simulation.menus import (
    build_menus,
    compose_menu,
    evolve_menus,
    psychological_round,
    vary_name,
)


@pytest.fixture(scope="module")
def library():
    return load_library(load_paths().external / "canonical_dishes.yaml")


@pytest.fixture(scope="module")
def frames(library) -> dict[str, pd.DataFrame]:
    return library.to_frames()


@pytest.fixture(scope="module")
def cfg() -> dict:
    sim = load_simulation()
    return {"supply": sim.supply, "calendar": sim.calendar}


@given(prices=st.lists(st.floats(min_value=1, max_value=2000), min_size=1, max_size=50))
@settings(max_examples=100, deadline=None)
def test_psychological_prices_end_in_0_5_or_9(prices: list[float]) -> None:
    out = psychological_round(np.array(prices), np.random.default_rng(0), nine_ending_rate=0.5)
    assert (out > 0).all()
    assert set(np.unique(out % 10)).issubset({0, 5, 9})


def test_compose_menu_has_mains_no_duplicates_and_own_cuisine_majority(frames, cfg) -> None:
    dishes = frames["canonical_dishes"]
    rng = np.random.default_rng(1)
    for cuisine in ["italiana", "tacos_tortas", "cafeteria", "peruana"]:
        menu = compose_menu(cuisine, dishes, size=25, supply=cfg["supply"], rng=rng)
        chosen = dishes.set_index("canonical_id").loc[menu]
        assert len(menu) == len(set(menu))
        assert (chosen["course"] == "MAIN").sum() >= 1
        assert chosen["cuisine"].isin([cuisine, "general"]).mean() >= 0.75


def test_vary_name_respects_variation_rate() -> None:
    rng = np.random.default_rng(2)
    noise = {"alias": 0.6, "decorate": 0.3, "typo": 0.1}
    kinds = [
        vary_name("Tacos al pastor", ["Orden de pastor"], 0.6, noise, rng)[1] for _ in range(2000)
    ]
    share_canonical = kinds.count("CANONICAL") / len(kinds)
    assert 0.35 < share_canonical < 0.45
    assert {"ALIAS", "DECORATED", "TYPO"} <= set(kinds)


@pytest.fixture(scope="module")
def menus(library, cfg) -> dict[str, pd.DataFrame]:
    restaurants = pd.DataFrame(
        {
            "restaurant_id": ["r1", "r2", "r3"],
            "cuisine_pred": ["italiana", "mariscos", "mexicana"],
            "ses_z": [1.0, 0.0, -1.0],
        }
    )
    return build_menus(restaurants, library, cfg["supply"], np.random.default_rng(3))


def test_item_allergens_match_item_ingredients(menus, library) -> None:
    attrs = menus["menu_item_attributes"]
    for item_id, group in attrs.groupby("menu_item_id"):
        ingredients = group.loc[group["kind"] == "ingredient", "value"].tolist()
        expected = sorted({a for i in ingredients for a in library.ingredients[i].allergens})
        actual = sorted(group.loc[group["kind"] == "allergen", "value"])
        assert actual == expected, item_id


def test_item_ingredients_are_subset_of_canonical(menus, frames) -> None:
    truth = menus["ground_truth_menu_items"].set_index("menu_item_id")["canonical_true"]
    canon = frames["canonical_dish_ingredients"].groupby("canonical_id")["ingredient"].apply(set)
    attrs = menus["menu_item_attributes"]
    ings = attrs[attrs["kind"] == "ingredient"].groupby("menu_item_id")["value"].apply(set)
    for item_id, items in ings.items():
        assert items <= canon[truth[item_id]]


def test_richer_area_has_higher_prices(menus, frames) -> None:
    items = menus["menu_items"].merge(menus["ground_truth_menu_items"], on="menu_item_id")
    base = frames["canonical_dishes"].set_index("canonical_id")["base_price"]
    items["ratio"] = items["price_cents"] / 100 / items["canonical_true"].map(base)
    by_restaurant = items.groupby("restaurant_id")["ratio"].median()
    assert by_restaurant["r1"] > by_restaurant["r3"]


def test_build_menus_is_reproducible(library, cfg) -> None:
    restaurants = pd.DataFrame(
        {"restaurant_id": ["a"], "cuisine_pred": ["japonesa"], "ses_z": [0.0]}
    )
    a = build_menus(restaurants, library, cfg["supply"], np.random.default_rng(9))
    b = build_menus(restaurants, library, cfg["supply"], np.random.default_rng(9))
    pd.testing.assert_frame_equal(a["menu_items"], b["menu_items"])


@pytest.fixture(scope="module")
def evolved(menus, library, cfg) -> dict[str, pd.DataFrame]:
    restaurants = pd.DataFrame(
        {
            "restaurant_id": ["r1", "r2", "r3"],
            "cuisine_pred": ["italiana", "mariscos", "mexicana"],
            "ses_z": [1.0, 0.0, -1.0],
        }
    )
    return evolve_menus(
        menus,
        restaurants,
        library,
        cfg["supply"],
        cfg["calendar"],
        months=6,
        rng=np.random.default_rng(4),
    )


def test_price_history_intervals_are_contiguous(evolved) -> None:
    ph = evolved["menu_item_price_history"].sort_values(["menu_item_id", "valid_from"])
    for _, g in ph.groupby("menu_item_id"):
        assert g["valid_to"].iloc[-1] is pd.NaT or pd.isna(g["valid_to"].iloc[-1])
        assert (g["valid_to"].iloc[:-1].to_numpy() == g["valid_from"].iloc[1:].to_numpy()).all()
        assert g["price_cents"].is_monotonic_increasing


def test_availability_events_are_well_formed(evolved) -> None:
    ev = evolved["menu_item_availability_events"].sort_values(["menu_item_id", "occurred_at"])
    items = evolved["menu_items"].set_index("menu_item_id")
    for item_id, g in ev.groupby("menu_item_id"):
        reasons = g["reason"].tolist()
        assert reasons[0] == "CREATED"
        assert reasons.count("SOLD_OUT") == reasons.count("RESTOCKED")
        archived = items.loc[item_id, "archived_at"]
        if pd.notna(archived):
            assert reasons[-1] == "ARCHIVED"
            assert g["occurred_at"].max() == archived


def test_evolution_adds_and_archives_items(evolved, menus) -> None:
    items = evolved["menu_items"]
    assert len(items) > len(menus["menu_items"])
    assert items["archived_at"].notna().any()
    assert (
        items["archived_at"].dropna() > items.loc[items["archived_at"].notna(), "created_at"]
    ).all()


# ---------------------------------------------- regresiones (bugs encontrados en nb 02)


def test_no_price_change_after_archive(evolved) -> None:
    ph = evolved["menu_item_price_history"].merge(
        evolved["menu_items"][["menu_item_id", "archived_at"]], on="menu_item_id"
    )
    archived = ph[ph["archived_at"].notna()]
    assert (archived["valid_from"] <= archived["archived_at"]).all()


def test_typo_always_changes_the_normalized_name(library) -> None:
    from puzzle_recsys.catalog.cuisine_imputation import normalize_name

    rng = np.random.default_rng(5)
    noise = {"typo": 1.0}
    for dish in library.dishes:
        for _ in range(20):
            text, kind = vary_name(dish.name, [], 1.0, noise, rng)
            assert kind == "TYPO"
            assert normalize_name(text) != normalize_name(dish.name), (dish.name, text)


@pytest.fixture(scope="module")
def chain_world(library, cfg):
    restaurants = pd.DataFrame(
        {
            "restaurant_id": ["t1", "t2", "t3", "solo"],
            "cuisine_pred": ["familiar_cadena"] * 3 + ["italiana"],
            "ses_z": [1.0, -1.0, 0.0, 0.0],
            "staff_band": ["31 a 50 personas"] * 3 + ["6 a 10 personas"],
            "chain": ["toks", "toks", "toks", None],
        }
    )
    menus = build_menus(restaurants, library, cfg["supply"], np.random.default_rng(11))
    evolved = evolve_menus(
        menus,
        restaurants,
        library,
        cfg["supply"],
        cfg["calendar"],
        months=6,
        rng=np.random.default_rng(12),
    )
    return restaurants, menus, evolved


def _menu_signature(tables, rid, active_only=False):
    items = tables["menu_items"].merge(tables["ground_truth_menu_items"], on="menu_item_id")
    items = items[items["restaurant_id"] == rid]
    if active_only:
        items = items[items["archived_at"].isna()]
    return sorted(zip(items["canonical_true"], items["name"], items["price_cents"], strict=True))


def test_chain_branches_share_the_same_menu(chain_world) -> None:
    _, menus, _ = chain_world
    assert (
        _menu_signature(menus, "t1") == _menu_signature(menus, "t2") == _menu_signature(menus, "t3")
    )


def test_chain_branches_evolve_together(chain_world) -> None:
    _, _, evolved = chain_world
    active = [
        sorted(c for c, _, _ in _menu_signature(evolved, rid, active_only=True))
        for rid in ("t1", "t2", "t3")
    ]
    assert active[0] == active[1] == active[2]


def test_menu_size_follows_staff_band(chain_world, cfg) -> None:
    _, menus, _ = chain_world
    sizes = menus["menu_items"].groupby("restaurant_id").size()
    by_band = cfg["supply"]["menu_size_by_staff_band"]
    assert sizes["solo"] <= by_band["6 a 10 personas"]["max"]
    assert sizes["t1"] > sizes["solo"]


def test_no_price_change_after_archive_under_stress(library, cfg) -> None:
    """Tasas altas de bajas y de cambios de precio: fuerza la coincidencia en el mismo mes."""
    supply = {
        **cfg["supply"],
        "evolution": {**cfg["supply"]["evolution"], "archive_rate": 0.3, "price_change_rate": 0.6},
    }
    restaurants = pd.DataFrame(
        {
            "restaurant_id": [f"r{i}" for i in range(6)],
            "cuisine_pred": ["mexicana", "italiana", "mariscos", "pizza", "china", "cafeteria"],
            "ses_z": [0.0] * 6,
        }
    )
    menus = build_menus(restaurants, library, supply, np.random.default_rng(21))
    evolved = evolve_menus(
        menus,
        restaurants,
        library,
        supply,
        cfg["calendar"],
        months=6,
        rng=np.random.default_rng(22),
    )
    ph = evolved["menu_item_price_history"].merge(
        evolved["menu_items"][["menu_item_id", "archived_at"]], on="menu_item_id"
    )
    archived = ph[ph["archived_at"].notna()]
    assert (archived["valid_from"] <= archived["archived_at"]).all()
