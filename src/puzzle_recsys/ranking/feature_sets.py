"""Qué columnas entran a cada modelo y cómo se preparan.

Una sola definición para la regresión logística y para LightGBM, así los dos
modelos compiten con **la misma información**.

- ``numeric``: entran tal cual (con NaN; cada modelo decide cómo tratarlos).
- ``raw_counts``: conteos con cola larga; entran como ``log_<col> = log(1 + col)``.
- ``categorical``: one-hot (logística) o categoría nativa (LightGBM).
- ``target_encoded``: alta cardinalidad; target encoding fuera de pliegue.

Usado en: notebooks 11 a 14.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSet:
    """Columnas de un modelo, agrupadas por cómo se preprocesan."""

    numeric: tuple[str, ...]
    raw_counts: tuple[str, ...]
    categorical: tuple[str, ...]
    target_encoded: tuple[str, ...]

    @property
    def logged(self) -> tuple[str, ...]:
        """Nombres de los conteos ya transformados con ``log1p``."""
        return tuple(f"log_{c}" for c in self.raw_counts)

    @property
    def dense(self) -> tuple[str, ...]:
        """Columnas numéricas finales (numéricas + conteos en log)."""
        return (*self.numeric, *self.logged)

    @property
    def columns(self) -> tuple[str, ...]:
        """Todas las columnas que recibe el modelo."""
        return (*self.dense, *self.categorical, *self.target_encoded)


_DISH = (
    "log_precio",
    "precio_relativo",
    "spice_level",
    "is_shareable",
    "n_candidatos",
    "share_restaurante",
    "share_global_canonico",
)
_COUNTS = ("pedidos_previos_platillo", "pedidos_previos_restaurante", "dias_en_menu")
_CATEGORICAL = ("course", "protein", "franja", "cocina_platillo")

FOR_YOU = FeatureSet(
    numeric=(
        *_DISH,
        "habito_tipo",
        "picante_medio",
        "log_ticket_medio",
        "tasa_compartir",
        "dias_desde_ultima",
        "afinidad_cocina_platillo",
        "share_proteina",
        "dif_picante",
        "ajuste_precio",
        "picante_declarado",
        "dif_picante_declarado",
        "cocina_declarada",
        "onboarding_completo",
        "fin_de_semana",
        "pax",
    ),
    raw_counts=(*_COUNTS, "veces_pidio_canonico", "veces_pidio_platillo", "visitas_previas"),
    categorical=_CATEGORICAL,
    target_encoded=("canonical_pred",),
)

FOR_TABLE = FeatureSet(
    numeric=(
        *_DISH,
        "tasa_compartido_canonico",
        "pct_excluidos",
        "dif_picante_media",
        "dif_picante_max",
        "personas",
        "pct_con_perfil",
        "tasa_compartir_media",
        "tasa_compartir_max",
        "fin_de_semana",
    ),
    raw_counts=(*_COUNTS, "veces_compartido_previo", "veces_mesa_pidio_canonico"),
    categorical=_CATEGORICAL,
    target_encoded=("canonical_pred",),
)


def model_frame(df: pd.DataFrame, features: FeatureSet = FOR_YOU) -> pd.DataFrame:
    """Columnas listas para el modelo: conteos en log, booleanos como 0/1, NaN intactos."""
    out = pd.DataFrame(index=df.index)
    for c in features.numeric:
        out[c] = df[c].astype(float)
    for c in features.raw_counts:
        out[f"log_{c}"] = np.log1p(df[c].astype(float).clip(lower=0))
    for c in (*features.categorical, *features.target_encoded):
        out[c] = df[c].astype(str)
    keep = [c for c in ("query_id", "y", "menu_item_id") if c in df.columns]
    return pd.concat([df[keep], out], axis=1)
