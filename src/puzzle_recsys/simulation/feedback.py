"""Calificaciones de platillos y visitas: logit ordenado con faltantes no aleatorios (MNAR).

- **Estrellas:** ``latente = utilidad vivida + ruido logístico``; los puntos de
  corte son cuantiles de la distribución de utilidades vividas
  (``ratings.cutpoints_quantiles``), así las estrellas se reparten como se
  configura y no dependen de la escala de la utilidad.
- **Quién califica (MNAR):** la probabilidad de calificar crece con lo
  **extremo** de la experiencia (``mnar_extremity_weight``): califica más quien
  quedó muy contento o muy molesto. Por eso lo observado toma forma de J aunque
  lo vivido sea simétrico. Solo pueden calificar los comensales con cuenta
  (``diner_id``): la pantalla de calificación vive en la app.

Usado en: notebooks 03, 05 (diagnóstico de faltantes: aquí MNAR es verificable
porque se simuló; con datos reales no se distingue de MAR).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

_LOGISTIC_SCALE = 0.3
_RATING_DELAY_NS = 2 * 3_600 * 10**9


def stars_from_latent(latent: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    """Logit ordenado: estrellas 1..5 según cuántos puntos de corte supera el latente."""
    return 1 + np.searchsorted(np.sort(cutpoints), latent, side="right")


def _rates(experienced: np.ndarray, base_rate: float, weight: float) -> np.ndarray:
    """Probabilidad de calificar: base x (1 + peso·|z|), truncada a 1."""
    z = (experienced - experienced.mean()) / (experienced.std() + 1e-9)
    return np.minimum(1.0, base_rate * (1 + weight * np.abs(z)))


def simulate_ratings(
    order_items: pd.DataFrame,
    guests: pd.DataFrame,
    sessions: pd.DataFrame,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Genera ``dish_ratings`` y ``visit_ratings`` a partir de la utilidad vivida.

    Parameters
    ----------
    order_items : pd.DataFrame
        Líneas con ``experienced`` (verdad oculta), ``guest_id``, ``session_id``.
    guests : pd.DataFrame
        Con ``guest_id`` y ``diner_id`` (``None`` = anónimo, no califica).
    sessions : pd.DataFrame
        Con ``session_id`` y ``opened_at`` (UTC).
    cfg : Mapping
        Bloque ``ratings`` de la configuración.
    rng : np.random.Generator
        Generador aleatorio.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``dish_ratings`` (``order_item_id``, ``guest_id``, ``stars``, ``created_at``)
        y ``visit_ratings`` (``session_id``, ``guest_id``, ``stars``, ``created_at``).
    """
    registered = guests.loc[guests["diner_id"].notna(), "guest_id"]
    lines = order_items[order_items["guest_id"].isin(registered)].copy()
    opened = sessions.set_index("session_id")["opened_at"]
    cutpoints = np.quantile(order_items["experienced"], cfg["cutpoints_quantiles"])

    latent = lines["experienced"].to_numpy() + rng.logistic(0.0, _LOGISTIC_SCALE, len(lines))
    rate = _rates(
        lines["experienced"].to_numpy(), cfg["dish_base_rate"], cfg["mnar_extremity_weight"]
    )
    rated = rng.random(len(lines)) < rate
    dish = pd.DataFrame(
        {
            "order_item_id": lines["order_item_id"].to_numpy()[rated],
            "guest_id": lines["guest_id"].to_numpy()[rated],
            "stars": stars_from_latent(latent[rated], cutpoints),
            "created_at": lines["session_id"].map(opened).to_numpy()[rated]
            + pd.Timedelta(_RATING_DELAY_NS),
        }
    )

    per_visit = lines.groupby(["session_id", "guest_id"], as_index=False)["experienced"].mean()
    latent_v = per_visit["experienced"].to_numpy() + rng.logistic(
        0.0, _LOGISTIC_SCALE, len(per_visit)
    )
    rate_v = _rates(
        per_visit["experienced"].to_numpy(), cfg["visit_base_rate"], cfg["mnar_extremity_weight"]
    )
    rated_v = rng.random(len(per_visit)) < rate_v
    visit = pd.DataFrame(
        {
            "session_id": per_visit["session_id"].to_numpy()[rated_v],
            "guest_id": per_visit["guest_id"].to_numpy()[rated_v],
            "stars": stars_from_latent(latent_v[rated_v], cutpoints),
            "created_at": per_visit["session_id"].map(opened).to_numpy()[rated_v]
            + pd.Timedelta(_RATING_DELAY_NS),
        }
    )
    return {
        "dish_ratings": dish.reset_index(drop=True),
        "visit_ratings": visit.reset_index(drop=True),
    }
