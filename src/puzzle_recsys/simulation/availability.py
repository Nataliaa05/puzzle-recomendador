"""Agotados dependientes de la demanda.

En el notebook 02 los agotados eran exógenos (misma tasa para todo). Aquí se
regeneran **antes** de simular las visitas con una tasa por platillo
proporcional a su demanda latente dentro del restaurante: ``softmax(atractivo
+ calidad)``, que es lo mismo que mueve la elección. Así se agota lo que más se
pide, sin circularidad (no depende de las visitas realizadas) y la tasa media
sigue siendo ``daily_sold_out_rate``.

Usado en: notebooks 03.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from puzzle_recsys.simulation.engine import WorldIndex, _local_ns
from puzzle_recsys.simulation.menus import _availability_events


def demand_driven_sold_out_rates(
    index: WorldIndex, base_rate: float, max_rate: float
) -> np.ndarray:
    """Tasa diaria de agotado por platillo, ∝ su demanda latente en su restaurante (con tope)."""
    score = index.items.appeal + index.items.quality
    rates = np.full(len(score), base_rate)
    for cand in index.rest_items:
        if len(cand) == 0:
            continue
        p = np.exp(score[cand] - score[cand].max())
        p /= p.sum()
        rates[cand] = np.minimum(max_rate, base_rate * len(cand) * p)
    return rates


def regenerate_availability(
    index: WorldIndex,
    menu_items: pd.DataFrame,
    base_rate: float,
    max_rate: float,
    end: pd.Timestamp,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Nueva tabla ``menu_item_availability_events`` con agotados según la demanda."""
    rates = demand_driven_sold_out_rates(index, base_rate, max_rate)
    items = menu_items.set_index("menu_item_id").loc[index.item_ids].reset_index()
    local = items.assign(
        created_at=items["created_at"].dt.tz_convert(end.tz),
        archived_at=items["archived_at"].dt.tz_convert(end.tz),
    )
    return _availability_events(local, end, rates, rng)


def attach_availability(index: WorldIndex, events: pd.DataFrame, tz: str) -> None:
    """Reemplaza en el índice los intervalos de agotado por los de ``events``."""
    pos = {item: k for k, item in enumerate(index.item_ids)}
    ev = events[events["reason"].isin(["SOLD_OUT", "RESTOCKED"])]
    ev = ev.assign(k=ev["menu_item_id"].map(pos), t=_local_ns(ev["occurred_at"], tz)).sort_values(
        ["k", "t"]
    )
    index.soldout_from.clear()
    index.soldout_to.clear()
    for k, group in ev.groupby("k", sort=False):
        index.soldout_from[int(k)] = group.loc[group["reason"] == "SOLD_OUT", "t"].to_numpy()
        index.soldout_to[int(k)] = group.loc[group["reason"] == "RESTOCKED", "t"].to_numpy()
