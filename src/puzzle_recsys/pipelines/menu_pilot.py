"""Piloto de menús web: cuántos restaurantes publican un menú legible con precios.

Uso::

    python -m puzzle_recsys.pipelines.menu_pilot --n 50

Toma restaurantes **de la muestra** de la simulación (no de todo el universo,
para medir el rendimiento en lo que realmente se usaría) que tengan un sitio
web válido en el DENUE, los visita con :class:`PoliteFetcher` (respeta
``robots.txt`` y pausa entre peticiones) y guarda:

- ``data/interim/menus_pilot/results.parquet``: un renglón por restaurante con
  su estado (``MENU_TEXT``, ``MENU_PDF``, ``MENU_IMAGES``, ``NO_MENU_LINK``…).
- ``data/interim/menus_pilot/items.parquet``: platillos y precios extraídos
  (SIN verificar: pasan a la base real solo tras revisión).

Usado en: notebooks 02 (piloto de menús reales).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from puzzle_recsys.config import load_paths, load_simulation
from puzzle_recsys.simulation.distributions import make_rng
from puzzle_recsys.sources.inegi import read_inegi_csv
from puzzle_recsys.sources.menus_web import PoliteFetcher, normalize_website, pilot_restaurant


def select_pilot(
    restaurants: pd.DataFrame, websites: pd.Series, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Elige ``n`` restaurantes con sitio web válido (al azar, reproducible).

    Parameters
    ----------
    restaurants : pd.DataFrame
        Muestra de la simulación (``restaurant_id``, ``name``, ``cuisine_pred``…).
    websites : pd.Series
        Campo ``www`` del DENUE indexado por ``restaurant_id``.
    n : int
        Tamaño del piloto.
    rng : np.random.Generator
        Generador aleatorio.

    Returns
    -------
    pd.DataFrame
        Restaurantes elegidos con su ``website`` normalizado.
    """
    out = restaurants.assign(
        website=restaurants["restaurant_id"].map(websites).map(normalize_website)
    )
    valid = out[out["website"].notna()]
    picks = rng.choice(len(valid), size=min(n, len(valid)), replace=False)
    return valid.iloc[np.sort(picks)].reset_index(drop=True)


def run(n: int, delay_s: float) -> Path:
    """Ejecuta el piloto y devuelve la carpeta de resultados."""
    paths, sim = load_paths(), load_simulation()
    restaurants = pd.read_parquet(paths.interim / "world" / "restaurants.parquet")
    www = read_inegi_csv(
        paths.raw / "denue_09_csv.zip",
        member_contains="conjunto_de_datos/denue",
        usecols=["id", "www"],
    ).set_index("id")["www"]
    pilot = select_pilot(restaurants, www, n, make_rng(sim.seed, "menu_pilot"))

    fetcher = PoliteFetcher(delay_s=delay_s)
    results, items = [], []
    for k, row in enumerate(pilot.itertuples(index=False), start=1):
        res = pilot_restaurant(fetcher, row.restaurant_id, row.website)
        print(
            f"[{k:>3}/{len(pilot)}] {res['status']:<15} {res['n_items']:>4} · {row.name[:40]}",
            flush=True,
        )
        items += [
            {"restaurant_id": row.restaurant_id, "item_name": name, "price_mxn": price}
            for name, price in res.pop("items")
        ]
        results.append(
            {
                **res,
                "name": row.name,
                "cuisine_pred": row.cuisine_pred,
                "staff_band": row.staff_band,
            }
        )

    out_dir = paths.interim / "menus_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(out_dir / "results.parquet", index=False)
    pd.DataFrame(items, columns=["restaurant_id", "item_name", "price_mxn"]).to_parquet(
        out_dir / "items.parquet", index=False
    )
    return out_dir


def main() -> None:
    """Punto de entrada de la línea de comandos."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=50, help="restaurantes del piloto")
    parser.add_argument(
        "--delay", type=float, default=2.0, help="segundos entre peticiones por dominio"
    )
    args = parser.parse_args()
    print(run(args.n, args.delay))


if __name__ == "__main__":
    main()
