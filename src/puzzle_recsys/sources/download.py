"""Descarga idempotente y validada de fuentes públicas.

El servidor del INEGI responde ``Content-Type: text/html`` incluso cuando
entrega un ZIP, y ante errores puede devolver una página HTML con código
200. Por eso la validez se decide por la firma del archivo (``PK``), no por
el encabezado HTTP.

Usado en: notebooks 01.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Any

from puzzle_recsys.config import load_paths, load_yaml

ZIP_MAGIC: bytes = b"PK\x03\x04"
_USER_AGENT = "puzzle-recsys/0.1 (proyecto academico)"


class InvalidDownloadError(RuntimeError):
    """El archivo descargado no tiene la firma esperada."""


def _has_magic(path: Path, magic: bytes) -> bool:
    with path.open("rb") as fh:
        return fh.read(len(magic)) == magic


def download_file(
    url: str,
    dest: Path,
    *,
    magic: bytes = ZIP_MAGIC,
    timeout: float = 120.0,
) -> Path:
    """Descarga ``url`` a ``dest`` si no existe ya un archivo válido.

    Escribe primero a ``<dest>.part`` y solo renombra si la firma es
    correcta, de modo que una descarga cortada nunca queda como válida.

    Parameters
    ----------
    url : str
        Dirección del archivo.
    dest : Path
        Ruta final.
    magic : bytes, optional
        Firma esperada al inicio del archivo (por defecto, ZIP).
    timeout : float, optional
        Segundos de espera de la conexión.

    Returns
    -------
    Path
        ``dest``.

    Raises
    ------
    InvalidDownloadError
        Si el contenido descargado no empieza con ``magic``.
    """
    if dest.exists() and _has_magic(dest, magic):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        if not _has_magic(part, magic):
            raise InvalidDownloadError(f"{url} no devolvió el archivo esperado")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    return dest


def download_source(
    name: str,
    *,
    raw_dir: Path | None = None,
    sources: dict[str, Any] | None = None,
) -> Path:
    """Descarga una fuente declarada en ``config/sources.yaml``.

    Parameters
    ----------
    name : str
        Clave de la fuente (``"denue"``, ``"census_ageb"``, ``"marco_geo"``).
    raw_dir : Path, optional
        Carpeta destino; por defecto ``data/raw``.
    sources : dict, optional
        Catálogo de fuentes; por defecto se lee del YAML.

    Returns
    -------
    Path
        Ruta del archivo descargado.
    """
    sources = sources if sources is not None else load_yaml("sources")
    raw_dir = raw_dir if raw_dir is not None else load_paths().raw
    spec = sources[name]
    return download_file(spec["url"], raw_dir / spec["file"])
