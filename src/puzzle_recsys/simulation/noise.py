"""Ruido del mundo real sobre las tablas del contrato (la verdad se guarda aparte).

- **Pedidos del mesero sin comensal:** en Puzzle, ``OrderItem.guestId`` es
  opcional; cuando el mesero carga el pedido sin asignarlo, la línea queda sin
  dueño. Se marca ``source = "WAITER"`` y ``guest_id = None``.
- **Cuentas duplicadas:** una persona se registra dos veces (correo y
  celular). Parte de sus visitas quedan en la segunda cuenta. Para el modelo
  son dos comensales con historial partido.

Las mesas gigantes (``outlier_party``) se generan en el motor, porque afectan
a la elección. Los faltantes del onboarding (MCAR: cada comensal lo completa con probabilidad
fija) viven en ``population``.

Usado en: notebooks 03, 05 (calidad de datos: aquí hay algo que encontrar).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def unassign_waiter_orders(
    order_items: pd.DataFrame, rate: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quita el comensal de una fracción de líneas (pedidos cargados por el mesero).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Las líneas con ``source`` y ``guest_id`` modificados, y la verdad
        (``order_item_id``, ``guest_id_true``).
    """
    out = order_items.copy()
    truth = out[["order_item_id", "guest_id"]].rename(columns={"guest_id": "guest_id_true"})
    hit = rng.random(len(out)) < rate
    out["source"] = np.where(hit, "WAITER", "QR")
    out.loc[hit, "guest_id"] = None
    return out, truth


def duplicate_accounts(
    diners: pd.DataFrame,
    guests: pd.DataFrame,
    rate: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Crea cuentas duplicadas y les traspasa la mitad de las visitas del original.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``diners`` con las cuentas nuevas, ``guests`` con ``diner_id``
        reasignado, y la verdad (``diner_id`` duplicado → ``diner_id_true``).
    """
    originals = diners["diner_id"].to_numpy()[rng.random(len(diners)) < rate]
    new_ids = [f"{d}-dup" for d in originals]
    extra = diners[diners["diner_id"].isin(originals)].copy()
    extra["diner_id"] = [f"{d}-dup" for d in extra["diner_id"]]
    out_guests = guests.copy()
    for original, dup in zip(originals, new_ids, strict=True):
        mine = out_guests.index[out_guests["diner_id"] == original]
        moved = mine[rng.random(len(mine)) < 0.5]
        out_guests.loc[moved, "diner_id"] = dup
    truth = pd.DataFrame({"diner_id": new_ids, "diner_id_true": originals})
    return pd.concat([diners, extra], ignore_index=True), out_guests, truth


def apply_noise(
    tables: Mapping[str, pd.DataFrame], cfg: Mapping[str, Any], rng: np.random.Generator
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Aplica todo el ruido configurado y devuelve (contrato, verdad)."""
    lines, lines_truth = unassign_waiter_orders(
        tables["order_items"], cfg["waiter_unassigned_rate"], rng
    )
    diners, guests, dup_truth = duplicate_accounts(
        tables["diners"], tables["guests"], cfg["duplicate_account_rate"], rng
    )
    contract = {**tables, "order_items": lines, "diners": diners, "guests": guests}
    truth = {"order_item_guests": lines_truth, "duplicate_accounts": dup_truth}
    return contract, truth
