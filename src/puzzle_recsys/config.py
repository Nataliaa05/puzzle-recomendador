"""Carga y validación de la configuración del proyecto.

Toda la configuración vive en ``config/*.yaml``.

El código no tiene rutas ni parámetros escritos a mano. Los bloques críticos se validan con
pydantic, de modo que un error tipográfico falla al cargar y no a mitad de una simulación.

Usado en: todos los notebooks y pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "config"


def load_yaml(name: str) -> dict[str, Any]:
    """Lee ``config/<name>.yaml``.

    Parameters
    ----------
    name : str
        Nombre del archivo sin extensión (``"simulation"``, ``"models"``...).

    Returns
    -------
    dict[str, Any]
        Contenido del YAML.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No existe la configuración: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class Paths(BaseModel):
    """Rutas absolutas del proyecto (resueltas desde ``config/paths.yaml``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw: Path
    external: Path
    interim: Path
    synthetic: Path
    processed: Path
    data_model: Path
    models: Path
    figures: Path


def load_paths() -> Paths:
    """Devuelve las rutas del proyecto como rutas absolutas.

    Las rutas del YAML son relativas a la raíz del proyecto, así los
    notebooks funcionan sin importar desde qué carpeta se abran.
    """
    raw = load_yaml("paths")
    return Paths(**{key: (PROJECT_ROOT / value).resolve() for key, value in raw.items()})


class SimulationScale(BaseModel):
    """Tamaño del mundo simulado."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_restaurants: PositiveInt
    n_diners: PositiveInt
    months: PositiveInt


class SimulationConfig(BaseModel):
    """Configuración de la simulación.

    ``seed`` y ``scale`` se validan estrictamente; el resto de los bloques
    (distribuciones, ruido, calibración...) se valida en el módulo que los
    usa, para no duplicar su contrato aquí.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    seed: int
    run_name: str = Field(min_length=1)
    metro_area: str | None
    scale: SimulationScale


def load_simulation() -> SimulationConfig:
    """Carga y valida ``config/simulation.yaml``."""
    return SimulationConfig(**load_yaml("simulation"))
