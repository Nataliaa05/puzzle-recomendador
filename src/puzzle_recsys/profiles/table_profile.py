"""Perfil de la **mesa**: lo que se sabe de quienes se sientan juntos.

La lista "para la mesa" no ordena para una persona sino para un grupo. Dos
reglas de agregación:

- **Promedio:** lo que le gusta a la mesa en general.
- **Miseria mínima:** lo que le gusta al **menos** contento; evita recomendar
  algo que alguien detesta. Se combinan con un peso (``least_misery_weight``).

Las **alergias** no se promedian: se toma la **unión**, porque un platillo con un
alérgeno de cualquier persona de la mesa se filtra antes del modelo.

Usado en: notebooks 09, 10.
"""

from __future__ import annotations

import pandas as pd


def group_aggregate(
    values: pd.Series, groups: pd.Series, *, least_misery_weight: float
) -> pd.DataFrame:
    """Media, mínimo y su combinación ``(1 - w)·media + w·mínimo`` por grupo."""
    by = pd.Series(values.to_numpy(), index=groups.to_numpy()).groupby(level=0)
    out = pd.DataFrame({"media": by.mean(), "minimo": by.min()})
    out["combinado"] = (1 - least_misery_weight) * out["media"] + least_misery_weight * out[
        "minimo"
    ]
    return out


def table_profile(
    guests: pd.DataFrame, member_profiles: pd.DataFrame, allergies: pd.Series
) -> pd.DataFrame:
    """Una fila por mesa: cuántos tienen perfil, gusto por compartir y alergias.

    ``member_profiles`` son los perfiles **a la fecha** de los comensales sentados
    (``session_id``, ``diner_id``, ``tasa_compartir``); ``allergies`` es la lista
    declarada de cada comensal, indexada por ``diner_id``.
    """
    by = guests.groupby("session_id")
    out = pd.DataFrame({"personas": by.size(), "con_perfil": by["diner_id"].count()})
    out["pct_con_perfil"] = out["con_perfil"] / out["personas"]
    share = member_profiles.groupby("session_id")["tasa_compartir"]
    out["tasa_compartir_media"] = share.mean()
    out["tasa_compartir_max"] = share.max()
    seated = guests.dropna(subset=["diner_id"])
    union = (
        seated.assign(a=seated["diner_id"].map(allergies))
        .explode("a")
        .dropna(subset=["a"])
        .groupby("session_id")["a"]
        .agg(lambda a: sorted(set(a)))
    )
    out["alergias"] = union.reindex(out.index)
    out["alergias"] = out["alergias"].map(lambda a: a if isinstance(a, list) else [])
    return out.reset_index()
