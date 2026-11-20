import pandas as pd
import pytest

from puzzle_recsys.eda.frames import diner_frame, item_frame, session_frame

TZ = "America/Mexico_City"


@pytest.fixture
def tiny():
    restaurants = pd.DataFrame(
        {
            "restaurant_id": ["r1", "r2"],
            "cuisine_pred": ["mexicana", "pizza"],
            "alcaldia": ["Coyoacán", "Tlalpan"],
            "staff_band": ["6 a 10 personas", "11 a 30 personas"],
        }
    )
    menu_items = pd.DataFrame(
        {
            "menu_item_id": ["m1", "m2", "m3"],
            "restaurant_id": ["r1", "r1", "r2"],
            "course": ["MAIN", "MAIN", "MAIN"],
            "protein": ["BEEF", "CHICKEN", "CHEESE"],
            "spice_level": [2, 0, 1],
            "is_shareable": [False, False, True],
            "price_cents": [10_000, 20_000, 15_000],
        }
    )
    sessions = pd.DataFrame(
        {
            "session_id": ["s1", "s2", "s3"],
            "restaurant_id": ["r1", "r2", "r1"],
            # 20:00 UTC = 14:00 en CDMX (comida); 02:00 UTC = 20:00 del día anterior (cena)
            "opened_at": pd.to_datetime(
                ["2025-01-06 20:00", "2025-01-11 02:00", "2025-01-13 20:00"], utc=True
            ),
            "pax": [2, 1, 1],
        }
    )
    guests = pd.DataFrame(
        {
            "guest_id": ["g1", "g2", "g3", "g4"],
            "session_id": ["s1", "s1", "s2", "s3"],
            "diner_id": ["d1", None, "d1", "d1"],
            "is_host": [True, False, True, True],
        }
    )
    order_items = pd.DataFrame(
        {
            "order_item_id": ["o1", "o2", "o3", "o4", "o5"],
            "session_id": ["s1", "s1", "s1", "s2", "s3"],
            "guest_id": ["g1", "g2", "g1", "g3", "g4"],
            "menu_item_id": ["m1", "m2", "m2", "m3", "m1"],
            "quantity": [1, 1, 1, 1, 1],
            "unit_price_cents": [10_000, 20_000, 20_000, 15_000, 10_000],
            "is_shared": [False, False, True, False, False],
        }
    )
    ratings = pd.DataFrame({"order_item_id": ["o1", "o5"], "stars": [5, 3]})
    profiles = pd.DataFrame(
        {
            "diner_id": ["d1"],
            "completed": [True],
            "favorite_cuisines": [["mexicana", "pizza", "tacos"]],
            "declared_spice": pd.array([2.0], dtype="Float64"),
        }
    )
    return dict(
        restaurants=restaurants,
        menu_items=menu_items,
        sessions=sessions,
        guests=guests,
        order_items=order_items,
        ratings=ratings,
        profiles=profiles,
    )


def test_session_frame_one_row_per_table_with_local_time(tiny) -> None:
    out = session_frame(
        tiny["sessions"], tiny["order_items"], tiny["restaurants"], tiny["menu_items"], TZ
    ).set_index("session_id")
    assert len(out) == 3
    assert out.loc["s1", "ticket"] == 500.0
    assert out.loc["s1", "ticket_por_persona"] == 250.0
    assert out.loc["s1", "franja"] == "comida"
    assert out.loc["s2", "franja"] == "cena"
    assert out.loc["s2", "dia"] == 4  # viernes local (sábado en UTC)
    assert out.loc["s1", "cocina"] == "mexicana"
    assert out.loc["s1", "nivel_precio"] == 150.0  # mediana del menú de r1


def test_item_frame_price_is_relative_to_same_course_in_same_restaurant(tiny) -> None:
    out = item_frame(tiny["menu_items"], tiny["order_items"], tiny["ratings"], tiny["restaurants"])
    out = out.set_index("menu_item_id")
    assert out.loc["m1", "precio_relativo"] == pytest.approx(100 / 150)
    assert out.loc["m3", "precio_relativo"] == 1.0
    assert out.loc["m1", "veces_pedido"] == 2
    assert out.loc["m1", "estrellas_medias"] == 4.0
    assert pd.isna(out.loc["m3", "estrellas_medias"])


def test_diner_frame_uses_own_individual_lines_only(tiny) -> None:
    sf = session_frame(
        tiny["sessions"], tiny["order_items"], tiny["restaurants"], tiny["menu_items"], TZ
    )
    out = diner_frame(
        tiny["guests"], sf, tiny["order_items"], tiny["menu_items"], tiny["profiles"]
    ).set_index("diner_id")
    assert out.loc["d1", "visitas"] == 3
    assert out.loc["d1", "restaurantes_distintos"] == 2
    # Líneas propias e individuales: m1 (2), m3 (1), m1 (2); la compartida m2 no cuenta.
    assert out.loc["d1", "picante_pedido"] == pytest.approx(5 / 3)
    assert out.loc["d1", "cocina_mas_visitada"] == "mexicana"
    assert out.loc["d1", "share_cocina_top"] == pytest.approx(2 / 3)
    assert out.loc["d1", "cocina_favorita_declarada"] == "mexicana"
    assert out.loc["d1", "picante_declarado"] == 2.0
