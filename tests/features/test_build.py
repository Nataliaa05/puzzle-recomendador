import numpy as np
import pandas as pd
import pytest

from puzzle_recsys.features.build import (
    candidates,
    diner_features,
    drop_allergens,
    for_you_queries,
    item_features,
    label,
    sharing_features,
    table_features,
)


def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


T0, T1, T2 = _t("2025-01-01"), _t("2025-01-10 14:00"), _t("2025-01-20 14:00")


@pytest.fixture
def world():
    menu = pd.DataFrame(
        {
            "menu_item_id": ["m1", "m2", "m3", "m4"],
            "restaurant_id": ["r1"] * 4,
            "course": ["MAIN", "MAIN", "STARTER", "MAIN"],
            "canonical_pred": ["c_a", "c_b", "c_s", "c_a"],
            "cocina_platillo": ["mexicana", "pizza", "mexicana", "mexicana"],
            "protein": ["BEEF", "CHEESE", "NONE", "BEEF"],
            "spice_level": [2, 0, 1, 2],
            "is_shareable": [False, False, True, False],
            "alergenos": [[], ["gluten"], [], []],
            "created_at": [T0, T0, T0, _t("2025-01-15")],
        }
    )
    availability = pd.DataFrame(
        {
            "menu_item_id": ["m1", "m2", "m3", "m4"],
            "t": [T0, T0, T0, _t("2025-01-15")],
            "is_available": [True, True, True, True],
        }
    )
    prices = pd.DataFrame(
        {
            "menu_item_id": ["m1", "m2", "m3", "m4", "m1"],
            "t": [T0, T0, T0, _t("2025-01-15"), _t("2025-01-15")],
            "price_cents": [10_000, 20_000, 8_000, 12_000, 11_000],
        }
    )
    # Visita s1 (d1 + anónimo): d1 pide m1 y la mesa comparte m3. Visita s2: d1 pide m1.
    lines = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2"],
            "guest_id": ["g1", "g1", "g3"],
            "diner_id": ["d1", "d1", "d1"],
            "restaurant_id": ["r1", "r1", "r1"],
            "menu_item_id": ["m1", "m3", "m1"],
            "is_shared": [False, True, False],
            "t": [T1, T1, T2],
        }
    ).merge(menu[["menu_item_id", "course", "canonical_pred", "protein"]], on="menu_item_id")
    return dict(menu=menu, availability=availability, prices=prices, lines=lines)


def test_for_you_queries_are_one_per_guest_visit_and_course_with_own_lines(world) -> None:
    q = for_you_queries(world["lines"])
    assert sorted(zip(q["session_id"], q["course"], strict=True)) == [
        ("s1", "MAIN"),
        ("s2", "MAIN"),
    ]
    assert q["query_id"].is_unique


def test_candidates_are_the_available_menu_of_that_course_at_that_time(world) -> None:
    q = for_you_queries(world["lines"])
    c = candidates(
        q, world["menu"], world["availability"], world["prices"], by=["restaurant_id", "course"]
    )
    got = c.groupby("session_id")["menu_item_id"].apply(sorted).to_dict()
    assert got == {"s1": ["m1", "m2"], "s2": ["m1", "m2", "m4"]}  # m4 aún no existía en s1
    price = c.set_index(["session_id", "menu_item_id"])["precio"]
    assert price["s1", "m1"] == 100.0 and price["s2", "m1"] == 110.0  # precio vigente


def test_drop_allergens_removes_dishes_with_a_declared_allergen(world) -> None:
    q = for_you_queries(world["lines"])
    c = candidates(
        q, world["menu"], world["availability"], world["prices"], by=["restaurant_id", "course"]
    )
    out = drop_allergens(
        c,
        world["menu"].set_index("menu_item_id")["alergenos"],
        pd.Series({"d1": ["gluten"]}),
        who="diner_id",
    )
    assert "m2" not in set(out["menu_item_id"])


def test_label_marks_what_the_guest_ordered_individually(world) -> None:
    q = for_you_queries(world["lines"])
    c = candidates(
        q, world["menu"], world["availability"], world["prices"], by=["restaurant_id", "course"]
    )
    own = world["lines"][~world["lines"]["is_shared"]]
    y = label(c, own, keys=["guest_id", "menu_item_id"]).set_index(["session_id", "menu_item_id"])[
        "y"
    ]
    assert y["s1", "m1"] == 1 and y["s1", "m2"] == 0 and y["s2", "m4"] == 0


def test_item_features_only_count_earlier_orders(world) -> None:
    q = for_you_queries(world["lines"])
    c = candidates(
        q, world["menu"], world["availability"], world["prices"], by=["restaurant_id", "course"]
    )
    f = item_features(c, world["lines"], world["menu"]).set_index(["session_id", "menu_item_id"])
    assert f.loc[("s1", "m1"), "pedidos_previos_platillo"] == 0
    assert f.loc[("s2", "m1"), "pedidos_previos_platillo"] == 1
    assert f.loc[("s2", "m4"), "pedidos_previos_platillo"] == 0
    assert f.loc[("s2", "m4"), "pedidos_previos_restaurante"] == 1  # m1 en s1 (MAIN de r1)
    assert f.groupby(level=0)["precio_relativo"].sum().abs().max() < 1e-9  # centrado por consulta
    assert f.loc[("s2", "m4"), "dias_en_menu"] == pytest.approx(5 + 14 / 24)
    assert (f["share_restaurante"] > 0).all()  # suavizado: nunca exactamente 0


def test_diner_features_use_only_the_diner_history_before_the_visit(world) -> None:
    q = for_you_queries(world["lines"])
    c = candidates(
        q, world["menu"], world["availability"], world["prices"], by=["restaurant_id", "course"]
    )
    c = item_features(c, world["lines"], world["menu"])
    profiles = pd.DataFrame(
        {
            "diner_id": ["d1", "d1"],
            "session_id": ["s1", "s2"],
            "visitas_previas": [0, 1],
            "picante_medio": [np.nan, 2.0],
            "log_ticket_medio": [np.nan, np.log(100)],
            "habito__MAIN": [np.nan, 1.0],
            "afinidad__mexicana": [0.5, 0.7],
            "afinidad__pizza": [0.5, 0.3],
        }
    )
    onboarding = pd.DataFrame(
        {
            "diner_id": ["d1"],
            "completed": [True],
            "declared_spice": [3.0],
            "favorite_cuisines": [["pizza"]],
        }
    )
    own = world["lines"][~world["lines"]["is_shared"]]
    f = diner_features(c, own, profiles, onboarding).set_index(["session_id", "menu_item_id"])
    assert f.loc[("s1", "m1"), "veces_pidio_canonico"] == 0
    assert f.loc[("s2", "m1"), "veces_pidio_canonico"] == 1
    assert f.loc[("s2", "m4"), "veces_pidio_canonico"] == 1  # mismo canónico, otro platillo
    assert f.loc[("s2", "m4"), "veces_pidio_platillo"] == 0
    assert f.loc[("s2", "m1"), "dif_picante"] == pytest.approx(0.0)
    assert f.loc[("s2", "m1"), "habito_tipo"] == 1.0
    assert f.loc[("s2", "m2"), "afinidad_cocina_platillo"] == pytest.approx(0.3)
    assert bool(f.loc[("s2", "m2"), "cocina_declarada"]) and not bool(
        f.loc[("s2", "m1"), "cocina_declarada"]
    )
    assert f.loc[("s2", "m1"), "share_proteina"] == pytest.approx(1.0)
    assert np.isnan(f.loc[("s1", "m1"), "share_proteina"])  # sin historia


def test_table_features_aggregate_members_and_flag_excluded_people(world) -> None:
    sessions = pd.DataFrame({"session_id": ["s1"], "restaurant_id": ["r1"], "t": [T2]})
    c = candidates(
        sessions.assign(query_id=0),
        world["menu"],
        world["availability"],
        world["prices"],
        by=["restaurant_id"],
        shareable_only=True,
    )
    c = item_features(c, world["lines"], world["menu"])
    members = pd.DataFrame({"session_id": ["s1", "s1"], "diner_id": ["d1", "d2"]})
    profiles = pd.DataFrame(
        {"diner_id": ["d1", "d2"], "session_id": ["s1", "s1"], "picante_medio": [0.0, 2.0]}
    )
    allergies = pd.Series({"d1": [], "d2": ["gluten"]})
    item_allergens = pd.Series({"m3": ["gluten"]})
    f = table_features(c, members, profiles, allergies, item_allergens, world["lines"]).iloc[0]
    assert f["menu_item_id"] == "m3"
    assert f["pct_excluidos"] == pytest.approx(0.5)
    assert f["dif_picante_media"] == pytest.approx(0.0)  # 1 - 0 y 1 - 2
    assert f["dif_picante_max"] == pytest.approx(1.0)  # al menos tolerante le pica 1 nivel de más
    assert f["veces_mesa_pidio_canonico"] == 1  # d1 lo pidió (compartido) en s1, antes de T2


def test_sharing_features_count_earlier_shared_orders(world) -> None:
    q = pd.DataFrame(
        {
            "query_id": [0, 1],
            "session_id": ["s1", "s2"],
            "menu_item_id": ["m3", "m3"],
            "canonical_pred": ["c_s", "c_s"],
            "t": [T1, T2],
        }
    )
    f = sharing_features(q, world["lines"]).set_index("session_id")
    assert f.loc["s1", "veces_compartido_previo"] == 0  # el compartido de s1 no se ve en s1
    assert f.loc["s2", "veces_compartido_previo"] == 1
    assert 0 < f.loc["s2", "tasa_compartido_canonico"] < 1


def test_without_history_erases_personal_history_but_keeps_onboarding() -> None:
    from puzzle_recsys.features.build import without_history

    df = pd.DataFrame(
        {
            "diner_id": ["d1", "d1"],
            "cocina_platillo": ["mexicana", "pizza"],
            "visitas_previas": [10, 10],
            "habito_tipo": [1.0, 1.0],
            "picante_medio": [2.0, 2.0],
            "log_ticket_medio": [5.0, 5.0],
            "tasa_compartir": [0.3, 0.3],
            "dias_desde_ultima": [3.0, 3.0],
            "share_proteina": [0.5, 0.2],
            "dif_picante": [0.0, 1.0],
            "ajuste_precio": [0.1, 0.2],
            "veces_pidio_canonico": [4, 0],
            "veces_pidio_platillo": [2, 0],
            "afinidad_cocina_platillo": [0.8, 0.1],
            "picante_declarado": [2.0, 2.0],
            "visitas_previas_restaurante": [3, 3],
        }
    )
    prior = pd.DataFrame({"mexicana": [0.6], "pizza": [0.4]}, index=["d1"])
    out = without_history(df, prior)
    assert (out["visitas_previas"] == 0).all() and (out["veces_pidio_canonico"] == 0).all()
    assert out["habito_tipo"].isna().all() and out["dif_picante"].isna().all()
    assert out["afinidad_cocina_platillo"].tolist() == [0.6, 0.4]  # vuelve al prior del onboarding
    assert out["picante_declarado"].tolist() == [2.0, 2.0]  # el onboarding se conserva
    assert out["primera_vez_restaurante"].all()


def test_as_new_restaurant_erases_local_popularity_only() -> None:
    from puzzle_recsys.features.build import as_new_restaurant

    df = pd.DataFrame(
        {
            "pedidos_previos_platillo": [10, 3],
            "pedidos_previos_restaurante": [13, 13],
            "share_restaurante": [0.3, 0.1],
            "share_global_canonico": [0.2, 0.05],
        }
    )
    out = as_new_restaurant(df)
    assert (out["pedidos_previos_restaurante"] == 0).all()
    share = out["share_restaurante"]
    assert (share == share.iloc[0]).all()  # sin historia local: todos iguales
    assert out["share_global_canonico"].tolist() == [0.2, 0.05]  # lo global sigue
