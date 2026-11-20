"""Tablas de análisis, una por **unidad**: mesa, comensal, platillo.

Cada relación del EDA se mide en la unidad donde tiene sentido. Por ejemplo,
"¿quien sale más gasta menos por visita?" es una pregunta por comensal, y
"¿el precio afecta la popularidad?" es una pregunta por platillo. Mezclar
unidades cambia la pregunta sin avisar.

Usado en: notebooks 07, 08.
"""

from __future__ import annotations

import pandas as pd

from puzzle_recsys.features.calendar import meal_period


def session_frame(
    sessions: pd.DataFrame,
    order_items: pd.DataFrame,
    restaurants: pd.DataFrame,
    menu_items: pd.DataFrame,
    tz: str,
) -> pd.DataFrame:
    """Una fila por mesa: ticket, hora local, franja, día y datos del restaurante.

    ``nivel_precio`` es la mediana de precios del menú del restaurante (pesos).
    ``dia`` va de 0 (lunes) a 6 (domingo), en hora local.
    """
    amount = order_items["quantity"] * order_items["unit_price_cents"] / 100
    ticket = amount.groupby(order_items["session_id"]).sum()
    local = sessions["opened_at"].dt.tz_convert(tz)
    level = menu_items.groupby("restaurant_id")["price_cents"].median() / 100
    rest = restaurants.set_index("restaurant_id")
    out = sessions[["session_id", "restaurant_id", "pax"]].copy()
    out["ticket"] = out["session_id"].map(ticket).fillna(0.0)
    out["ticket_por_persona"] = out["ticket"] / out["pax"]
    out["hora"] = local.dt.hour.to_numpy()
    out["franja"] = meal_period(out["hora"])
    out["dia"] = local.dt.day_of_week.to_numpy()
    out["fin_de_semana"] = out["dia"] >= 4  # viernes a domingo
    out["cocina"] = out["restaurant_id"].map(rest["cuisine_pred"])
    out["alcaldia"] = out["restaurant_id"].map(rest["alcaldia"])
    out["personal"] = out["restaurant_id"].map(rest["staff_band"])
    out["nivel_precio"] = out["restaurant_id"].map(level)
    return out


def item_frame(
    menu_items: pd.DataFrame,
    order_items: pd.DataFrame,
    ratings: pd.DataFrame,
    restaurants: pd.DataFrame,
) -> pd.DataFrame:
    """Una fila por platillo del menú: atributos, precio relativo, popularidad y estrellas.

    ``precio_relativo`` = precio / mediana del mismo tipo de plato en el mismo
    restaurante: compara un platillo con sus **alternativas reales**, no con
    platillos de otros lugares.
    """
    out = menu_items[
        ["menu_item_id", "restaurant_id", "course", "protein", "spice_level", "is_shareable"]
    ].copy()
    out["precio"] = menu_items["price_cents"] / 100
    course_median = out.groupby(["restaurant_id", "course"])["precio"].transform("median")
    out["precio_relativo"] = out["precio"] / course_median
    out["veces_pedido"] = (
        out["menu_item_id"].map(order_items["menu_item_id"].value_counts()).fillna(0).astype(int)
    )
    stars = ratings.merge(order_items[["order_item_id", "menu_item_id"]], on="order_item_id")
    out["estrellas_medias"] = out["menu_item_id"].map(stars.groupby("menu_item_id")["stars"].mean())
    out["cocina"] = out["restaurant_id"].map(restaurants.set_index("restaurant_id")["cuisine_pred"])
    return out


def diner_frame(
    guests: pd.DataFrame,
    sessions: pd.DataFrame,
    order_items: pd.DataFrame,
    menu_items: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Una fila por comensal registrado: comportamiento observado y onboarding declarado.

    ``sessions`` es la salida de :func:`session_frame`. El picante pedido usa solo
    las líneas **propias e individuales** (una compartida no revela el gusto de
    cada uno).
    """
    seated = guests.dropna(subset=["diner_id"]).merge(
        sessions[["session_id", "restaurant_id", "cocina", "ticket_por_persona"]], on="session_id"
    )
    by = seated.groupby("diner_id")
    out = pd.DataFrame(
        {
            "visitas": by.size(),
            "restaurantes_distintos": by["restaurant_id"].nunique(),
            "ticket_por_persona_mediano": by["ticket_por_persona"].median(),
        }
    )
    cuisine_counts = seated.groupby(["diner_id", "cocina"]).size()
    top = cuisine_counts.sort_values(ascending=False).groupby(level=0).head(1)
    out["cocina_mas_visitada"] = pd.Series(
        top.index.get_level_values(1), index=top.index.get_level_values(0)
    )
    out["share_cocina_top"] = (
        pd.Series(top.to_numpy(), index=top.index.get_level_values(0)) / out["visitas"]
    )

    own = order_items.loc[~order_items["is_shared"], ["guest_id", "menu_item_id"]]
    own = own.merge(guests[["guest_id", "diner_id"]], on="guest_id").dropna(subset=["diner_id"])
    own = own.merge(menu_items[["menu_item_id", "spice_level"]], on="menu_item_id")
    out["picante_pedido"] = own.groupby("diner_id")["spice_level"].mean()

    prof = profiles.set_index("diner_id")
    out["onboarding_completo"] = prof["completed"].reindex(out.index).fillna(False).astype(bool)
    out["picante_declarado"] = prof["declared_spice"].reindex(out.index).astype(float)
    favorite = prof["favorite_cuisines"].map(lambda f: f[0] if len(f) else None)
    out["cocina_favorita_declarada"] = favorite.reindex(out.index)
    return out.reset_index(names="diner_id")
