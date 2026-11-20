"""Métricas para validar que los datos sintéticos se parezcan a la realidad.

Dos tipos de comprobación:

- **Hechos estilizados:** patrones que muestra cualquier dato real de
  restaurantes (cola larga de popularidad, picos de fin de semana,
  calificaciones en J, recompra de favoritos). Se miden y se comparan contra
  **bandas plausibles** declaradas en ``config/validation.yaml``.
- **Ajuste a lo configurado:** la distribución simulada contra la que se pidió
  (variación total para categóricas; KS y Wasserstein para continuas).

Usado en: notebooks 04.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def top_share(counts: pd.Series, fraction: float) -> float:
    """Proporción del total que concentra la fracción ``fraction`` más alta.

    Ejemplo: ``top_share(pedidos_por_platillo, 0.2)`` = qué parte de los
    pedidos se lleva el 20 % de los platillos más pedidos (regla 80/20).
    """
    values = np.sort(np.asarray(counts, dtype=float))[::-1]
    k = max(1, round(len(values) * fraction))
    total = values.sum()
    return float(values[:k].sum() / total) if total else 0.0


def gini(counts: pd.Series) -> float:
    """Coeficiente de Gini (0 = todo igual de popular; cerca de 1 = todo en uno)."""
    values = np.sort(np.asarray(counts, dtype=float))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    cum = np.cumsum(values)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Distancia de variación total entre dos distribuciones discretas (0 a 1)."""
    p = np.asarray(p, dtype=float) / np.sum(p)
    q = np.asarray(q, dtype=float) / np.sum(q)
    return float(0.5 * np.abs(p - q).sum())


def repeat_rate(orders: pd.DataFrame, who: str, what: str) -> float:
    """Proporción de pedidos que repiten algo que esa persona ya pidió antes.

    Es la señal que un recomendador puede aprender: si nadie repite nada, el
    historial no predice; si todos repiten todo, el problema es trivial.
    """
    return float(orders.duplicated(subset=[who, what]).mean())


def in_band(value: float, band: Sequence[float]) -> bool:
    """``value`` dentro del intervalo cerrado ``[band[0], band[1]]``."""
    return band[0] <= value <= band[1]


def stylized_facts(
    sessions: pd.DataFrame,
    guests: pd.DataFrame,
    order_items: pd.DataFrame,
    *,
    canonical_of: Mapping[str, str],
    restaurant_cuisine: Mapping[str, str],
    favorite_cuisine: Mapping[str, str],
    window_end: pd.Timestamp | None = None,
) -> dict[str, float]:
    """Hechos estilizados de una corrida (salida de ``simulate_visits``).

    Las métricas están definidas para ser **comparables entre escalas**:

    - La concentración se mide sobre **platillos canónicos** (~220), no sobre
      platillos servidos (~36,000): con menos pedidos, la popularidad de miles
      de platillos se ve plana por puro azar.
    - Todo se mide dentro de la misma **ventana de tiempo** (``window_end``):
      la recompra crece con el horizonte (más tiempo, más oportunidades).

    Parameters
    ----------
    sessions, guests, order_items : pd.DataFrame
        Tablas de la corrida. ``guests`` debe traer ``taste_ref`` (verdad).
    canonical_of : Mapping[str, str]
        ``menu_item_id`` → platillo canónico verdadero.
    restaurant_cuisine : Mapping[str, str]
        ``restaurant_id`` → cocina.
    favorite_cuisine : Mapping[str, str]
        ``taste_ref`` del comensal → su cocina de mayor afinidad latente.
    window_end : pd.Timestamp, optional
        Solo se usan las visitas anteriores a esta fecha.
    """
    if window_end is not None:
        sessions = sessions[sessions["opened_at"] < window_end]
        guests = guests[guests["session_id"].isin(sessions["session_id"])]
        order_items = order_items[order_items["session_id"].isin(sessions["session_id"])]
    lines = order_items.merge(guests[["guest_id", "taste_ref"]], on="guest_id")
    lines = lines.assign(canonical=lines["menu_item_id"].map(canonical_of))
    # Solo comensales registrados ("d…"): los perfiles anónimos ("a…") se reutilizan
    # entre mesas distintas y simulaban recompra (bug de la 1.ª versión).
    own = lines[~lines["is_shared"] & lines["taste_ref"].str.startswith("d")]
    hosts = guests[guests["is_host"]].merge(
        sessions[["session_id", "restaurant_id"]], on="session_id"
    )
    visits = hosts.groupby("taste_ref").size()
    fav = hosts["taste_ref"].map(favorite_cuisine) == hosts["restaurant_id"].map(restaurant_cuisine)
    return {
        "top20_share_dishes": top_share(lines["canonical"].value_counts(), 0.2),
        "favorite_cuisine_share": float(fav.mean()),
        "repeat_rate_canonical": repeat_rate(own, "taste_ref", "canonical"),
        "shared_table_rate": float(
            order_items.loc[order_items["is_shared"], "session_id"].nunique() / len(sessions)
        ),
        "visits_p90_over_p50": float(visits.quantile(0.9) / visits.median()),
    }
