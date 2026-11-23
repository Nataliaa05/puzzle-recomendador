"""Consultas **punto en el tiempo**: contar o leer el estado *antes* de una fecha.

Casi todas las features dependen de la historia: "¿cuántas veces se pidió este
platillo **antes** de esta visita?", "¿estaba disponible **a esa hora**?",
"¿cuánto costaba **ese día**?". Hacerlo con ``merge`` + filtro de fechas explota
en memoria (millones de candidatos por millones de pedidos). Aquí se resuelve con
búsqueda binaria:

1. Las claves (p. ej. platillo, o comensal + canónico) se convierten en enteros.
2. Cada evento se codifica como ``clave · rango + segundos`` y se ordena.
3. Para cada consulta, ``searchsorted`` encuentra cuántos eventos de su clave
   quedan antes de su fecha. Es O((eventos + consultas) · log eventos).

Convenciones: ``count_before`` cuenta eventos **estrictamente anteriores** (lo
que pasó en la misma visita no se ve); ``state_at`` toma el último estado
**hasta** la fecha incluida (un platillo creado a las 6:00 está disponible a
las 6:00).

Usado en: notebooks 10.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def to_seconds(times: pd.Series) -> np.ndarray:
    """Segundos desde 1970 (UTC), sin importar la zona horaria de la serie."""
    t = pd.to_datetime(times)
    if t.dt.tz is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    return t.to_numpy().astype("datetime64[s]").astype(np.int64)


def _encode(
    events: pd.DataFrame, queries: pd.DataFrame, keys: Sequence[str], time: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Códigos combinados ``clave · rango + segundos`` de eventos y consultas."""
    both = pd.concat([events[list(keys)], queries[list(keys)]], ignore_index=True)
    codes = both.groupby(list(keys), sort=False, dropna=False).ngroup().to_numpy(np.int64)
    te, tq = to_seconds(events[time]), to_seconds(queries[time])
    t0 = min(te.min(initial=0), tq.min(initial=0)) - 1
    span = int(max(te.max(initial=0), tq.max(initial=0)) - t0 + 2)
    ce = codes[: len(events)] * span + (te - t0)
    cq_key = codes[len(events) :] * span
    return ce, cq_key, tq - t0, span


def count_before(
    events: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    keys: Sequence[str],
    time: str = "t",
    weight: str | None = None,
) -> pd.Series:
    """Cuenta los eventos (o suma ``weight``) de la misma clave anteriores a cada consulta."""
    ce, cq_key, tq, _ = _encode(events, queries, keys, time)
    order = np.argsort(ce, kind="stable")
    sorted_codes = ce[order]
    start = np.searchsorted(sorted_codes, cq_key, side="left")
    stop = np.searchsorted(sorted_codes, cq_key + tq, side="left")
    if weight is None:
        return pd.Series(stop - start, index=queries.index)
    w = np.concatenate([[0.0], np.cumsum(events[weight].to_numpy(dtype=float)[order])])
    return pd.Series(w[stop] - w[start], index=queries.index)


def state_at(
    events: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    keys: Sequence[str],
    value: str,
    time: str = "t",
) -> pd.Series:
    """Último valor de ``value`` con la misma clave en o antes de cada consulta (NaN si no hay)."""
    ce, cq_key, tq, _ = _encode(events, queries, keys, time)
    order = np.argsort(ce, kind="stable")
    sorted_codes = ce[order]
    start = np.searchsorted(sorted_codes, cq_key, side="left")
    last = np.searchsorted(sorted_codes, cq_key + tq, side="right") - 1
    found = last >= start
    values = events[value].to_numpy()[order]
    out = pd.Series(np.full(len(queries), None, dtype=object), index=queries.index)
    out[found] = values[last[found]]
    return out.infer_objects()
