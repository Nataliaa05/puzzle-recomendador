"""Construcción de las tablas de features "para ti" y "para la mesa" (punto en el tiempo).

Una fila = un **candidato** (platillo) dentro de una **consulta**:

- "Para ti": consulta = una persona registrada, en una visita, para un tipo de
  plato. Candidatos = platillos de ese tipo **disponibles a esa hora** en el
  restaurante, sin los alérgenos que declaró. Etiqueta = lo pidió para sí.
- "Para la mesa": consulta = una mesa. Candidatos = platillos **compartibles**
  disponibles. Etiqueta = la mesa lo compartió.

Todas las features usan solo información **anterior** a la visita
(:mod:`puzzle_recsys.features.pit`) y el canónico **predicho** (notebook 09).

Usado en: notebooks 10.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from puzzle_recsys.features.pit import count_before, state_at

# Suavizado de las proporciones de popularidad: (conteo + 1) / (total + K).
_SMOOTH_K = 20.0


def for_you_queries(lines: pd.DataFrame) -> pd.DataFrame:
    """Consultas "para ti": una por (invitado registrado, visita, tipo de plato) con pedido propio.

    ``lines`` necesita ``session_id``, ``guest_id``, ``diner_id``, ``restaurant_id``,
    ``course``, ``is_shared`` y ``t`` (apertura de la mesa). Solo cuentan las líneas
    **propias e individuales**: así cada consulta tiene al menos un positivo.
    """
    own = lines[~lines["is_shared"] & lines["diner_id"].notna()]
    q = own.drop_duplicates(["guest_id", "course"])[
        ["guest_id", "session_id", "diner_id", "restaurant_id", "course", "t"]
    ].reset_index(drop=True)
    q.insert(0, "query_id", np.arange(len(q), dtype=np.int64))
    return q


def candidates(
    queries: pd.DataFrame,
    menu: pd.DataFrame,
    availability: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    by: Sequence[str],
    shareable_only: bool = False,
) -> pd.DataFrame:
    """Platillos del menú disponibles a la hora de cada consulta, con su precio vigente (pesos).

    ``by``: columnas que unen consulta y menú (restaurante y tipo de plato para
    "para ti"; solo restaurante para "para la mesa"). ``availability`` y
    ``prices`` son eventos (``menu_item_id``, ``t``, ``is_available`` /
    ``price_cents``); se toma el último estado **hasta** la hora de la consulta.
    """
    pool = menu[menu["is_shareable"]] if shareable_only else menu
    cols = list(dict.fromkeys([*by, "menu_item_id"]))
    c = queries.merge(pool[cols], on=list(by))
    available = state_at(availability, c, keys=["menu_item_id"], value="is_available")
    c = c[available.fillna(False).astype(bool).to_numpy()].reset_index(drop=True)
    price = state_at(prices, c, keys=["menu_item_id"], value="price_cents")
    c["precio"] = price.astype(float).to_numpy() / 100
    return c


def drop_allergens(
    cands: pd.DataFrame, item_allergens: pd.Series, person_allergies: pd.Series, *, who: str
) -> pd.DataFrame:
    """Quita los candidatos con algún alérgeno declarado por la persona de la consulta.

    Es un **filtro duro** previo al modelo: la seguridad alimentaria no se deja
    a un ranking.
    """
    mine = cands[who].map(person_allergies)
    theirs = cands["menu_item_id"].map(item_allergens)
    clash = [
        bool(set(a) & set(b)) if isinstance(a, list | np.ndarray) and len(a) else False
        for a, b in zip(mine, theirs, strict=True)
    ]
    return cands[~np.asarray(clash, dtype=bool)].reset_index(drop=True)


def label(cands: pd.DataFrame, positives: pd.DataFrame, *, keys: Sequence[str]) -> pd.DataFrame:
    """Agrega ``y`` = 1 si (``keys``) aparece en ``positives``, 0 si no."""
    hit = positives[list(keys)].drop_duplicates().assign(y=1)
    out = cands.merge(hit, on=list(keys), how="left")
    out["y"] = out["y"].fillna(0).astype(np.int8)
    return out


def item_features(cands: pd.DataFrame, lines: pd.DataFrame, menu: pd.DataFrame) -> pd.DataFrame:
    """Atributos del platillo, precio relativo y popularidad **a la fecha**.

    - ``precio_relativo``: log del precio menos la media del log en la consulta
      (¿es de lo caro o de lo barato de **este** menú?).
    - ``pedidos_previos_platillo``: veces que se pidió antes, por cualquiera.
    - ``pedidos_previos_restaurante``: pedidos previos del restaurante en ese tipo
      de plato (decide si su popularidad es confiable).
    - ``share_restaurante``: proporción suavizada de los pedidos previos del
      restaurante en ese tipo de plato. ``share_global_canonico``: lo mismo para el
      canónico en **toda** la plataforma (respaldo cuando el restaurante es nuevo).
    - ``dias_en_menu``: antigüedad del platillo (detecta platillos nuevos).
    """
    attrs = ["canonical_pred", "cocina_platillo", "protein", "spice_level", "is_shareable"]
    have = [a for a in attrs if a not in cands.columns]
    c = cands.merge(menu[["menu_item_id", *have, "created_at"]], on="menu_item_id")
    if "course" not in c.columns:
        c = c.merge(menu[["menu_item_id", "course"]], on="menu_item_id")
    c["log_precio"] = np.log(c["precio"])
    c["precio_relativo"] = c["log_precio"] - c.groupby("query_id")["log_precio"].transform("mean")
    c["n_candidatos"] = c.groupby("query_id")["menu_item_id"].transform("size")

    ev = lines[["menu_item_id", "restaurant_id", "course", "canonical_pred", "t"]]
    c["pedidos_previos_platillo"] = count_before(ev, c, keys=["menu_item_id"]).to_numpy()
    rest = count_before(ev, c, keys=["restaurant_id", "course"]).to_numpy()
    c["pedidos_previos_restaurante"] = rest
    c["share_restaurante"] = (c["pedidos_previos_platillo"] + 1) / (rest + _SMOOTH_K)
    canon = count_before(ev, c, keys=["canonical_pred"]).to_numpy()
    course = count_before(ev, c, keys=["course"]).to_numpy()
    c["share_global_canonico"] = (canon + 1) / (course + _SMOOTH_K)
    c["dias_en_menu"] = (c["t"] - c["created_at"]).dt.total_seconds() / 86_400
    return c.drop(columns="created_at")


def diner_features(
    cands: pd.DataFrame,
    own_lines: pd.DataFrame,
    profiles: pd.DataFrame,
    onboarding: pd.DataFrame,
) -> pd.DataFrame:
    """Perfil a la fecha de la persona y su **coincidencia** con cada candidato.

    ``profiles`` es el perfil a la fecha (notebook 09) por (``diner_id``,
    ``session_id``); ``own_lines`` son las líneas propias e individuales con
    ``diner_id``, ``canonical_pred``, ``menu_item_id``, ``protein`` y ``t``.
    """
    keep = ["visitas_previas", "picante_medio", "log_ticket_medio"]
    keep += [c for c in ("dias_desde_ultima", "tasa_compartir") if c in profiles.columns]
    c = cands.merge(
        profiles[["diner_id", "session_id", *keep]], on=["diner_id", "session_id"], how="left"
    )

    habit_cols = [col for col in profiles.columns if col.startswith("habito__")]
    habits = profiles.melt(["diner_id", "session_id"], habit_cols, "course", "habito_tipo")
    habits["course"] = habits["course"].str.removeprefix("habito__")
    c = c.merge(habits, on=["diner_id", "session_id", "course"], how="left")

    aff_cols = [col for col in profiles.columns if col.startswith("afinidad__")]
    aff = profiles.melt(
        ["diner_id", "session_id"], aff_cols, "cocina_platillo", "afinidad_cocina_platillo"
    )
    aff["cocina_platillo"] = aff["cocina_platillo"].str.removeprefix("afinidad__")
    c = c.merge(aff, on=["diner_id", "session_id", "cocina_platillo"], how="left")

    ev = own_lines[["diner_id", "canonical_pred", "menu_item_id", "protein", "t"]]
    c["veces_pidio_canonico"] = count_before(ev, c, keys=["diner_id", "canonical_pred"]).to_numpy()
    c["veces_pidio_platillo"] = count_before(ev, c, keys=["diner_id", "menu_item_id"]).to_numpy()
    prot = count_before(ev, c, keys=["diner_id", "protein"]).to_numpy()
    total = count_before(ev, c, keys=["diner_id"]).to_numpy()
    c["share_proteina"] = np.where(total > 0, prot / np.maximum(total, 1), np.nan)
    c["dif_picante"] = c["spice_level"] - c["picante_medio"]
    c["ajuste_precio"] = c["log_precio"] - c["log_ticket_medio"]

    ob = onboarding.set_index("diner_id")
    declared = c["diner_id"].map(ob["declared_spice"]).astype(float)
    c["picante_declarado"] = declared.to_numpy()
    c["dif_picante_declarado"] = c["spice_level"] - c["picante_declarado"]
    favorites = c["diner_id"].map(ob["favorite_cuisines"])
    c["cocina_declarada"] = [
        isinstance(f, list | np.ndarray) and cz in list(f)
        for f, cz in zip(favorites, c["cocina_platillo"], strict=True)
    ]
    c["onboarding_completo"] = c["diner_id"].map(ob["completed"]).fillna(False).astype(bool)
    return c


def table_features(
    cands: pd.DataFrame,
    members: pd.DataFrame,
    profiles: pd.DataFrame,
    allergies: pd.Series,
    item_allergens: pd.Series,
    lines: pd.DataFrame,
) -> pd.DataFrame:
    """Coincidencia del platillo compartible con **la mesa** (comensales registrados).

    ``cands`` ya pasó por :func:`item_features` (trae ``canonical_pred`` y ``spice_level``).

    - ``pct_excluidos``: proporción de los registrados que no pueden comerlo por
      alergia (el platillo se comparte solo entre quienes sí pueden).
    - ``dif_picante_media`` / ``dif_picante_max``: picante del platillo menos el
      habitual de cada uno; el máximo es la **miseria mínima** (al menos tolerante).
    - ``veces_mesa_pidio_canonico``: veces que los registrados de la mesa
      pidieron ese canónico antes (propio o compartido).
    """
    c = cands
    pairs = c[
        ["query_id", "session_id", "menu_item_id", "canonical_pred", "spice_level", "t"]
    ].merge(members[["session_id", "diner_id"]], on="session_id")
    pairs = pairs.merge(
        profiles[["diner_id", "session_id", "picante_medio"]],
        on=["diner_id", "session_id"],
        how="left",
    )
    mine = pairs["diner_id"].map(allergies)
    theirs = pairs["menu_item_id"].map(item_allergens)
    pairs["excluido"] = [
        bool(set(a) & set(b))
        if isinstance(a, list | np.ndarray) and isinstance(b, list | np.ndarray)
        else False
        for a, b in zip(mine, theirs, strict=True)
    ]
    pairs["dif"] = pairs["spice_level"] - pairs["picante_medio"]
    ev = lines.dropna(subset=["diner_id"])[["diner_id", "canonical_pred", "t"]]
    pairs["previos"] = count_before(ev, pairs, keys=["diner_id", "canonical_pred"]).to_numpy()
    agg = pairs.groupby(["query_id", "menu_item_id"]).agg(
        pct_excluidos=("excluido", "mean"),
        dif_picante_media=("dif", "mean"),
        dif_picante_max=("dif", "max"),
        veces_mesa_pidio_canonico=("previos", "sum"),
    )
    return c.join(agg, on=["query_id", "menu_item_id"])


def sharing_features(cands: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Historia de **compartir** de cada candidato, a la fecha.

    - ``veces_compartido_previo``: veces que una mesa compartió este platillo antes.
    - ``tasa_compartido_canonico``: de los pedidos previos del canónico en toda la
      plataforma, qué proporción fue compartida (suavizada).
    """
    ev = lines[["menu_item_id", "canonical_pred", "is_shared", "t"]]
    shared = ev[ev["is_shared"]]
    c = cands.copy()
    c["veces_compartido_previo"] = count_before(shared, c, keys=["menu_item_id"]).to_numpy()
    canon_shared = count_before(shared, c, keys=["canonical_pred"]).to_numpy()
    canon_all = count_before(ev, c, keys=["canonical_pred"]).to_numpy()
    c["tasa_compartido_canonico"] = (canon_shared + 1) / (canon_all + _SMOOTH_K)
    return c
