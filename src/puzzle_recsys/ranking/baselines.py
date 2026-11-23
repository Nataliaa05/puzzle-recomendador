"""Líneas base del ranking: popularidad con respaldo y similitud por contenido.

**Popularidad (jerarquía de respaldo).** Si el restaurante tiene al menos ``N``
pedidos previos en ese tipo de plato, se ordena por lo más pedido **ahí**; si no
(restaurante nuevo o con poca historia), por lo más pedido de ese canónico en
**toda la plataforma**. ``N`` se elige en validación (notebook 11).

**Contenido.** Puntaje armado a mano con el perfil a la fecha: cuánto coincide
el platillo con lo que la persona ya pidió (canónico, proteína, cocina, picante y
precio). No aprende nada de los datos: sirve para saber cuánto aporta "aprender".

Usado en: notebooks 11, 15.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def popularity_score(df: pd.DataFrame, *, min_orders: int) -> pd.Series:
    """Popularidad del restaurante si tiene ``min_orders`` pedidos previos; si no, la global."""
    trust = df["pedidos_previos_restaurante"] >= min_orders
    return df["share_restaurante"].where(trust, df["share_global_canonico"])


def content_score(df: pd.DataFrame) -> pd.Series:
    """Coincidencia platillo-perfil; un dato faltante no suma ni resta.

    ``log(1 + veces que pidió el canónico) + proporción de esa proteína + afinidad
    por la cocina - |diferencia de picante| / 3 - |ajuste de precio|``
    """
    terms = [
        np.log1p(df["veces_pidio_canonico"].astype(float)),
        df["share_proteina"],
        df["afinidad_cocina_platillo"],
        -df["dif_picante"].abs() / 3,
        -df["ajuste_precio"].abs(),
    ]
    return sum(t.fillna(0.0) for t in terms)


def random_score(df: pd.DataFrame, *, seed: int) -> pd.Series:
    """Orden al azar: el piso que cualquier modelo debe superar."""
    return pd.Series(np.random.default_rng(seed).random(len(df)), index=df.index)
