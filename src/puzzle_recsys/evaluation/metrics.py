"""Métricas de ranking por consulta: NDCG@k, HitRate@k, Recall@k, MAP@k y cobertura.

Relevancia **binaria** (lo pidió o no; notebook 06). Las métricas son
**condicionales**: solo hay consultas con al menos un positivo ("dado que pidió
postre, ¿acertamos cuál?").

- **HitRate@k:** ¿hay al menos un positivo en el top k? La más fácil de explicar.
- **Recall@k:** qué fracción de los positivos quedó en el top k.
- **NDCG@k:** premia poner los positivos **arriba**: un acierto en el lugar 1 vale
  más que en el 3 (descuento ``1 / log2(lugar + 1)``), normalizado por el mejor orden posible.
- **MAP@k:** promedio de la precisión en cada lugar donde hay un positivo.

**Empates:** si dos candidatos tienen el mismo score (muy común en popularidad:
platillos sin pedidos previos), el orden se decide con un número aleatorio
reproducible. Si se desempatara por el orden de las filas, el positivo podría
quedar arriba "por suerte" y las métricas saldrían infladas.

Usado en: notebooks 11 a 15.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _ranked(df: pd.DataFrame, group: str, score: str, seed: int) -> pd.DataFrame:
    tiebreak = np.random.default_rng(seed).random(len(df))
    order = np.lexsort((tiebreak, -df[score].to_numpy(dtype=float), df[group].to_numpy()))
    out = df.iloc[order]
    rank = out.groupby(group, sort=False).cumcount().to_numpy() + 1
    return out.assign(_rank=rank)


def rank_metrics(
    df: pd.DataFrame,
    *,
    ks: Sequence[int] = (3, 5),
    group: str = "query_id",
    label: str = "y",
    score: str = "score",
    seed: int = 0,
) -> pd.DataFrame:
    """Métricas de cada consulta (una fila por consulta)."""
    r = _ranked(df[[group, label, score]], group, score, seed)
    rel = r[label].to_numpy(dtype=float)
    rank = r["_rank"].to_numpy()
    g = r[group].to_numpy()
    hits_so_far = pd.Series(rel).groupby(g, sort=False).cumsum().to_numpy()
    discount = 1 / np.log2(rank + 1)
    max_k = max(ks)
    ideal = np.concatenate([[0.0], np.cumsum(1 / np.log2(np.arange(2, max_k + 2)))])

    out = pd.DataFrame(index=pd.unique(g))
    out.index.name = group
    total_pos = pd.Series(rel, index=g).groupby(level=0, sort=False).sum()
    for k in ks:
        top = rank <= k
        frame = pd.DataFrame(
            {
                "g": g,
                "hit": rel * top,
                "dcg": rel * discount * top,
                "ap": rel * top * hits_so_far / rank,
            }
        )
        agg = frame.groupby("g", sort=False)[["hit", "dcg", "ap"]].sum().reindex(out.index)
        n = total_pos.reindex(out.index).to_numpy()
        idcg = ideal[np.minimum(n, k).astype(int)]
        out[f"ndcg@{k}"] = agg["dcg"].to_numpy() / idcg
        out[f"hit@{k}"] = (agg["hit"].to_numpy() > 0).astype(float)
        out[f"recall@{k}"] = agg["hit"].to_numpy() / n
        out[f"map@{k}"] = agg["ap"].to_numpy() / np.minimum(n, k)
    return out.reset_index()


def summarize(per_query: pd.DataFrame, *, by: str | Sequence[str] | None = None) -> pd.DataFrame:
    """Promedio de cada métrica (todas las consultas pesan igual), en total o por segmento."""
    metrics = [c for c in per_query.columns if "@" in c]
    if by is None:
        out = per_query[metrics].mean().to_frame().T
        out.insert(0, "consultas", len(per_query))
        return out
    g = per_query.groupby(by, observed=True)
    out = g[metrics].mean()
    out.insert(0, "consultas", g.size())
    return out.reset_index()


def catalog_coverage(
    df: pd.DataFrame,
    *,
    k: int,
    group: str = "query_id",
    item: str = "menu_item_id",
    score: str = "score",
    seed: int = 0,
) -> float:
    """Fracción de los platillos candidatos que llega al top k en al menos una consulta.

    Un ranking que siempre recomienda lo mismo tiene cobertura baja: los platillos
    menos conocidos nunca se muestran (sesgo de popularidad).
    """
    r = _ranked(df[[group, item, score]], group, score, seed)
    shown = r.loc[r["_rank"] <= k, item].nunique()
    return shown / df[item].nunique()
