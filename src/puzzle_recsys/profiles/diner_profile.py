"""Perfil del comensal **a la fecha**: en cada visita, solo con sus visitas anteriores.

Si el perfil de una visita incluyera lo que se pidió en esa misma visita, el
modelo "vería la respuesta" al entrenar (fuga de información) y en producción
rendiría mucho peor. Por eso cada perfil es una suma acumulada **menos la visita
actual**, ordenada por fecha.

Qué contiene:

- ``visitas_previas`` y ``dias_desde_ultima``.
- ``afinidad__<cocina>``: proporción suavizada de visitas a cada cocina. Sin
  historia es el **prior** (población + cocinas declaradas en el onboarding); con
  historia se mueve hacia lo observado:
  ``(visitas a c + s·prior_c) / (visitas + s)``.
- ``habito__<tipo>``: líneas propias e individuales por visita de cada tipo de plato.
- ``picante_medio``, ``log_ticket_medio`` y ``tasa_compartir``.

Usado en: notebooks 09, 10.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

COURSES = ("MAIN", "DRINK", "STARTER", "SIDE", "DESSERT")


def visit_events(
    sessions: pd.DataFrame,
    guests: pd.DataFrame,
    order_items: pd.DataFrame,
    menu_items: pd.DataFrame,
    *,
    courses: Sequence[str] = COURSES,
) -> pd.DataFrame:
    """Una fila por (comensal registrado, visita) con lo que hizo en esa visita.

    ``sessions`` necesita ``session_id``, ``opened_at``, ``cocina`` y
    ``ticket_por_persona``. Las líneas compartidas no cuentan para los hábitos
    ni el picante (no revelan el gusto de cada uno), pero marcan ``compartio``.
    """
    seated = guests.dropna(subset=["diner_id"])[["guest_id", "session_id", "diner_id"]].merge(
        sessions[["session_id", "opened_at", "cocina", "ticket_por_persona"]], on="session_id"
    )
    own = order_items.loc[~order_items["is_shared"], ["guest_id", "menu_item_id"]].merge(
        menu_items[["menu_item_id", "course", "spice_level"]], on="menu_item_id"
    )
    counts = pd.crosstab(own["guest_id"], own["course"]).reindex(
        columns=list(courses), fill_value=0
    )
    counts.columns = [f"n__{c}" for c in counts.columns]
    spice = own.groupby("guest_id")["spice_level"].agg(picante_suma="sum", picante_n="size")
    out = seated.join(counts, on="guest_id").join(spice, on="guest_id")
    fill = [*counts.columns, "picante_suma", "picante_n"]
    out[fill] = out[fill].fillna(0).astype(int)
    # Una mesa sin líneas tiene ticket 0: no es un gasto observado, queda como faltante.
    out["log_ticket_pp"] = np.log(out["ticket_por_persona"].where(out["ticket_por_persona"] > 0))
    shared = set(order_items.loc[order_items["is_shared"], "session_id"])
    out["compartio"] = out["session_id"].isin(shared)
    return out.drop(columns=["guest_id", "ticket_por_persona"]).reset_index(drop=True)


def cuisine_prior(
    profiles: pd.DataFrame, population_share: pd.Series, *, declared_weight: float = 0.5
) -> pd.DataFrame:
    """Prior de afinidad por cocina para cada comensal (arranque en frío).

    Sin onboarding es la proporción de la población (``population_share``). Con
    onboarding se mezcla con un reparto uniforme entre las cocinas declaradas:
    ``(1 - w)·población + w·declaradas``. Las cocinas declaradas que no existen
    en ``population_share`` se ignoran.
    """
    cuisines = list(population_share.index)
    base = population_share.to_numpy(dtype=float)
    rows = []
    for completed, favorites in zip(
        profiles["completed"], profiles["favorite_cuisines"], strict=True
    ):
        known = [c for c in (favorites if completed else []) if c in population_share.index]
        if not known:
            rows.append(base)
            continue
        declared = np.isin(cuisines, known) / len(known)
        rows.append((1 - declared_weight) * base + declared_weight * declared)
    return pd.DataFrame(rows, index=profiles["diner_id"].to_numpy(), columns=cuisines)


def point_in_time_profiles(
    events: pd.DataFrame,
    *,
    cuisines: Sequence[str],
    prior: pd.DataFrame,
    prior_strength: float,
    courses: Sequence[str] = COURSES,
) -> pd.DataFrame:
    """Perfil de cada comensal **antes** de cada una de sus visitas.

    ``events`` es la salida de :func:`visit_events`; ``prior`` tiene una fila por
    comensal y una columna por cocina (salida de :func:`cuisine_prior`). Un
    comensal sin fila en ``prior`` recibe el prior promedio.
    """
    ev = events.sort_values(["diner_id", "opened_at"], kind="stable").reset_index(drop=True)
    by = ev.groupby("diner_id", sort=False)
    prev = by.cumcount().to_numpy()

    def before(frame: pd.DataFrame) -> pd.DataFrame:
        """Suma acumulada por comensal sin la visita actual."""
        return frame.groupby(ev["diner_id"], sort=False).cumsum() - frame

    visited = pd.get_dummies(ev["cocina"]).reindex(columns=list(cuisines), fill_value=False)
    seen = before(visited.astype(float)).to_numpy()
    p = prior.reindex(ev["diner_id"]).fillna(prior.mean()).to_numpy()
    affinity = (seen + prior_strength * p) / (prev[:, None] + prior_strength)

    num_cols = [f"n__{c}" for c in courses]
    stats = ev[[*num_cols, "picante_suma", "picante_n", "compartio"]].astype(float)
    stats["log_ticket_pp"] = ev["log_ticket_pp"].fillna(0.0)
    stats["ticket_n"] = ev["log_ticket_pp"].notna().astype(float)
    acc = before(stats)
    visits = np.where(prev > 0, prev, np.nan)

    out = ev[["diner_id", "session_id", "opened_at"]].copy()
    out["visitas_previas"] = prev
    out["dias_desde_ultima"] = (
        ev["opened_at"] - by["opened_at"].shift()
    ).dt.total_seconds().to_numpy() / 86_400
    for j, c in enumerate(cuisines):
        out[f"afinidad__{c}"] = affinity[:, j]
    for c in courses:
        out[f"habito__{c}"] = acc[f"n__{c}"].to_numpy() / visits
    out["picante_medio"] = acc["picante_suma"] / acc["picante_n"].replace(0, np.nan)
    out["log_ticket_medio"] = acc["log_ticket_pp"] / acc["ticket_n"].replace(0, np.nan)
    out["tasa_compartir"] = acc["compartio"].to_numpy() / visits
    return out


def recovery_by_history(
    profiles: pd.DataFrame,
    truth: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    *,
    bins: Sequence[float],
) -> pd.DataFrame:
    """Spearman entre cada rasgo del perfil y su rasgo latente, por visitas previas.

    Muestra cuánta historia hace falta para que el perfil refleje el gusto real.
    ``truth`` está indexado por ``diner_id``; ``bins`` son los bordes de los
    tramos de ``visitas_previas`` (cerrados a la izquierda).
    """
    edges = list(bins)
    labels = [
        str(int(lo))
        if hi - lo == 1
        else f"{int(lo)}+"
        if np.isinf(hi)
        else f"{int(lo)}-{int(hi) - 1}"
        for lo, hi in itertools.pairwise(edges)
    ]
    tramo = pd.cut(profiles["visitas_previas"], edges, right=False, labels=labels)
    rows = []
    for obs, lat in pairs:
        frame = pd.DataFrame(
            {
                "o": profiles[obs].to_numpy(),
                "l": truth[lat].reindex(profiles["diner_id"]).to_numpy(),
                "t": tramo.to_numpy(),
            }
        ).dropna()
        for label, g in frame.groupby("t", observed=True, sort=False):
            rho = float(stats.spearmanr(g["o"], g["l"])[0]) if len(g) > 2 else np.nan
            rows.append(
                {
                    "observado": obs,
                    "latente": lat,
                    "visitas_previas": label,
                    "n": len(g),
                    "rho": rho,
                }
            )
    out = pd.DataFrame(rows)
    out["visitas_previas"] = pd.Categorical(out["visitas_previas"], categories=labels, ordered=True)
    return out.sort_values(["observado", "visitas_previas"]).reset_index(drop=True)
