"""Adaptador Parquet del puerto ``Repository`` (datos sintéticos del curso).

Estructura de una corrida::

    data/synthetic/<run_id>/
    ├── <tabla>.parquet        ← contrato (lo que vería Puzzle en producción)
    ├── ground_truth/          ← verdad oculta: SOLO la lee la evaluación
    └── config/                ← copia de la configuración usada (reproducibilidad)

Usado en: notebooks 02-15.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def run_id_for(run_name: str, seed: int) -> str:
    """Identificador estable de una corrida: ``<run_name>-<seed>``."""
    return f"{run_name}-{seed}"


class ParquetRepository:
    """Lee y escribe tablas como ``<root>/<nombre>.parquet``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def ground_truth(self) -> ParquetRepository:
        """Repositorio de la verdad oculta (carpeta separada a propósito)."""
        return ParquetRepository(self.root / "ground_truth")

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.parquet"

    def read_table(self, name: str) -> pd.DataFrame:
        """Lee una tabla; falla con un mensaje claro si no existe."""
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(f"No existe la tabla '{name}' en {self.root}")
        return pd.read_parquet(path)

    def write_table(self, name: str, df: pd.DataFrame) -> None:
        """Escribe (o reemplaza) una tabla."""
        self.root.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._path(name), index=False)


def start_run(
    synthetic_dir: Path, run_id: str, *, config_files: Iterable[Path]
) -> ParquetRepository:
    """Prepara la carpeta de una corrida y guarda una copia de su configuración.

    Parameters
    ----------
    synthetic_dir : Path
        Carpeta ``data/synthetic``.
    run_id : str
        Identificador de la corrida (ver :func:`run_id_for`).
    config_files : Iterable[Path]
        Archivos de configuración a copiar (``simulation.yaml``, catálogos...).

    Returns
    -------
    ParquetRepository
        Repositorio con raíz en ``<synthetic_dir>/<run_id>``.
    """
    repo = ParquetRepository(Path(synthetic_dir) / run_id)
    snapshot = repo.root / "config"
    snapshot.mkdir(parents=True, exist_ok=True)
    for file in config_files:
        shutil.copy2(file, snapshot / Path(file).name)
    return repo
