"""Separación entre el contrato (lo que vería Puzzle) y la verdad oculta.

Regla anti-fuga (ADR 0002 y 0003): ninguna columna de verdad puede quedar en
una tabla del contrato. Si una se cuela, el modelo aprende de la respuesta y
todas las métricas mienten. :func:`assert_no_truth_leak` lo verifica antes de
guardar.

Usado en: notebooks 03.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from puzzle_recsys.simulation.population import TASTE_COURSES, DinerTastes

#: Columnas que solo pueden vivir en ``ground_truth/``.
TRUTH_COLUMNS: frozenset[str] = frozenset(
    {
        "utility",
        "experienced",
        "taste_ref",
        "canonical_true",
        "name_variant",
        "price_multiplier",
        "appeal",
        "quality",
        "restaurant_utility",
        "ses_z",
        "shared_by",
    }
)


def tastes_frame(tastes: DinerTastes, diner_ids: Sequence[str]) -> pd.DataFrame:
    """Gustos latentes en una tabla (vectores como listas): verdad oculta."""
    return pd.DataFrame(
        {
            "diner_id": list(diner_ids),
            "cuisine_affinity": list(tastes.cuisine_affinity),
            "protein_affinity": list(tastes.protein_affinity),
            "ingredient_pref": list(tastes.ingredient_pref),
            "spice_tolerance": tastes.spice_tolerance,
            "price_sensitivity": tastes.price_sensitivity,
            "distance_sensitivity": tastes.distance_sensitivity,
            "variety_seeking": tastes.variety_seeking,
            "sharer": tastes.sharer,
            **{
                f"habit_{c.lower()}": tastes.course_habit[:, j] for j, c in enumerate(TASTE_COURSES)
            },
            "diet": tastes.diet,
            "allergies": [
                [tastes.allergen_names[j] for j, has in enumerate(row) if has]
                for row in tastes.allergies
            ],
        }
    )


def split_truth(
    order_items: pd.DataFrame, guests: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separa las columnas de verdad de las líneas de pedido y de los invitados.

    Returns
    -------
    tuple
        ``order_items`` y ``guests`` del contrato, y sus verdades
        (``order_item_utilities`` y ``guest_tastes``).
    """
    line_truth = [c for c in ("utility", "experienced", "shared_by") if c in order_items]
    truth_lines = order_items[["order_item_id", *line_truth]]
    truth_guests = guests[["guest_id", "taste_ref"]]
    return (
        order_items.drop(columns=line_truth),
        guests.drop(columns=["taste_ref"]),
        truth_lines,
        truth_guests,
    )


def assert_no_truth_leak(contract: Mapping[str, pd.DataFrame]) -> None:
    """Falla si alguna tabla del contrato contiene una columna de verdad oculta."""
    leaks = {
        name: sorted(set(table.columns) & TRUTH_COLUMNS)
        for name, table in contract.items()
        if set(table.columns) & TRUTH_COLUMNS
    }
    if leaks:
        raise ValueError(f"Fuga de verdad oculta en el contrato: {leaks}")


def tastes_from_frame(
    frame: pd.DataFrame,
    *,
    cuisines: Sequence[str],
    proteins: Sequence[str],
    ingredients: Sequence[str],
    allergens: Sequence[str],
) -> DinerTastes:
    """Inversa de :func:`tastes_frame`: reconstruye los gustos desde ``ground_truth``.

    Permite volver a simular con **exactamente** la misma población (p. ej. para
    calibrar o hacer réplicas Monte Carlo) sin duplicar el código que la generó.
    """
    import numpy as np

    allergen_pos = {a: j for j, a in enumerate(allergens)}
    allergies = np.zeros((len(frame), len(allergens)), dtype=bool)
    for i, names in enumerate(frame["allergies"]):
        for name in names:
            allergies[i, allergen_pos[name]] = True
    return DinerTastes(
        cuisine_names=tuple(cuisines),
        protein_names=tuple(proteins),
        ingredient_names=tuple(ingredients),
        allergen_names=tuple(allergens),
        cuisine_affinity=np.vstack(frame["cuisine_affinity"].to_numpy()),
        protein_affinity=np.vstack(frame["protein_affinity"].to_numpy()),
        ingredient_pref=np.vstack(frame["ingredient_pref"].to_numpy()).astype(np.int8),
        spice_tolerance=frame["spice_tolerance"].to_numpy(),
        price_sensitivity=frame["price_sensitivity"].to_numpy(),
        distance_sensitivity=frame["distance_sensitivity"].to_numpy(),
        variety_seeking=frame["variety_seeking"].to_numpy(),
        sharer=frame["sharer"].to_numpy(),
        course_habit=np.column_stack(
            [frame[f"habit_{c.lower()}"].to_numpy() for c in TASTE_COURSES]
        ),
        diet=frame["diet"].to_numpy(),
        allergies=allergies,
    )
