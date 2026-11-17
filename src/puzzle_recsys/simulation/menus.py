"""Menús de los restaurantes: composición, nombres, precios y evolución en el tiempo.

Unidad de decisión: el **dueño del menú**. Es la cadena si el restaurante es
una sucursal (todas comparten carta, nombres, precios y cambios, como una
``MenuTemplate`` de Puzzle) o el propio restaurante si es independiente.

Qué se simula y por qué:

- **Composición:** cada menú mezcla platillos de su cocina, del pool general
  (bebidas y postres comunes) y una fracción pequeña de otras cocinas. Su
  tamaño depende del personal ocupado del restaurante (DENUE).
- **Nombres con ruido:** la carta rara vez usa el nombre canónico ("Orden de
  pastor", "TACOS AL PASTRO"). Ese ruido es lo que la homologación (nb 09)
  tiene que deshacer; la verdad se guarda en ``ground_truth``.
- **Precios:** precio típico del canónico por un multiplicador **lognormal**
  del dueño, centrado según la escolaridad promedio de la zona (proxy
  socioeconómico real del Censo), por ruido lognormal del platillo, con
  redondeo psicológico (terminan en 0, 5 o 9).
- **Recetas:** el atributo verdadero es el del platillo SERVIDO; algunas
  recetas quitan un ingrediente secundario (y sus alérgenos se recalculan).
- **Evolución:** altas, bajas, aumentos de precio y agotados a lo largo de los
  meses. Sin esto no se pueden reconstruir los negativos "a la fecha".
  (Los agotados de aquí son exógenos; en el nb 03 se vuelven dependientes de
  la demanda.)

Usado en: notebooks 02, 03.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from puzzle_recsys.catalog.cuisine_imputation import normalize_name
from puzzle_recsys.catalog.library import Library

COURSES: tuple[str, ...] = ("MAIN", "STARTER", "DRINK", "DESSERT", "SIDE")
_DECORATIONS: tuple[str, ...] = (
    " de la casa",
    " especial",
    " (orden)",
    " tradicional",
    " estilo chef",
    " premium",
)
_UPPERCASE_RATE = 0.15
_ITEM_COLUMNS = [
    "menu_item_id",
    "restaurant_id",
    "name",
    "course",
    "protein",
    "spice_level",
    "is_shareable",
    "price_cents",
]


# ---------------------------------------------------------------- piezas puras


def psychological_round(
    prices: np.ndarray, rng: np.random.Generator, *, nine_ending_rate: float
) -> np.ndarray:
    """Redondea a múltiplos de 5 y, con cierta probabilidad, baja 1 peso (…9).

    Returns
    -------
    np.ndarray
        Precios enteros en pesos, positivos, que terminan en 0, 5 o 9.
    """
    rounded = np.maximum(5, np.round(np.asarray(prices, dtype=float) / 5) * 5)
    nine = (rounded % 10 == 0) & (rng.random(rounded.shape) < nine_ending_rate)
    return (rounded - nine).astype(np.int64)


def compose_menu(
    cuisine: str,
    dishes: pd.DataFrame,
    *,
    size: int,
    supply: Mapping[str, Any],
    rng: np.random.Generator,
) -> list[str]:
    """Elige los platillos canónicos del menú.

    Parameters
    ----------
    cuisine : str
        Cocina del restaurante (``cuisine_pred``).
    dishes : pd.DataFrame
        Tabla ``canonical_dishes``.
    size : int
        Tamaño objetivo del menú (se recorta si los pools no alcanzan).
    supply : Mapping
        Bloque ``supply`` de la configuración (``course_mix``, ``cross_cuisine_rate``).
    rng : np.random.Generator
        Generador aleatorio.

    Returns
    -------
    list[str]
        ``canonical_id`` sin repetir, con al menos un plato fuerte.
    """
    own = dishes[dishes["cuisine"] == cuisine]
    general = dishes[dishes["cuisine"] == "general"]
    others = dishes[~dishes["cuisine"].isin([cuisine, "general"])]

    n_cross = int(rng.binomial(size, supply["cross_cuisine_rate"]))
    mix = np.array([supply["course_mix"][c] for c in COURSES], dtype=float)
    quotas = rng.multinomial(size - n_cross, mix / mix.sum())

    chosen: list[str] = []
    for course, quota in zip(COURSES, quotas, strict=True):
        pool = pd.concat([own[own["course"] == course], general[general["course"] == course]])
        if quota == 0 or pool.empty:
            continue
        weights = np.where(pool["cuisine"] == cuisine, 2.0, 1.0)
        k = min(int(quota), len(pool))
        picks = rng.choice(len(pool), size=k, replace=False, p=weights / weights.sum())
        chosen.extend(pool["canonical_id"].iloc[np.sort(picks)])

    remaining = others[~others["canonical_id"].isin(chosen)]
    if n_cross and not remaining.empty:
        picks = rng.choice(len(remaining), size=min(n_cross, len(remaining)), replace=False)
        chosen.extend(remaining["canonical_id"].iloc[np.sort(picks)])

    is_main = dishes.set_index("canonical_id").loc[chosen, "course"].eq("MAIN")
    if not is_main.any():
        mains = own.loc[own["course"] == "MAIN", "canonical_id"].to_numpy()
        chosen.append(str(rng.choice(mains)))
    return list(dict.fromkeys(chosen))


def _typo(text: str, rng: np.random.Generator) -> str:
    """Error de dedo verosímil que SIEMPRE cambia el nombre normalizado.

    Intercambia dos letras contiguas **distintas** tras normalizar (intercambiar
    "rr" o "e/é" no cambia nada). Si no hay un par así, borra una letra.
    """
    swappable = [
        i
        for i in range(len(text) - 1)
        if text[i].isalpha()
        and text[i + 1].isalpha()
        and normalize_name(text[i]) != normalize_name(text[i + 1])
    ]
    if swappable:
        i = int(rng.choice(swappable))
        return text[:i] + text[i + 1] + text[i] + text[i + 2 :]
    letters = [i for i, ch in enumerate(text) if ch.isalpha()]
    i = int(rng.choice(letters))
    return text[:i] + text[i + 1 :]


def vary_name(
    name: str,
    aliases: Sequence[str],
    variation_rate: float,
    noise: Mapping[str, float],
    rng: np.random.Generator,
) -> tuple[str, str]:
    """Nombre con el que el platillo aparece en la carta.

    Returns
    -------
    tuple[str, str]
        El nombre y su tipo: ``CANONICAL``, ``ALIAS``, ``DECORATED`` o ``TYPO``.
    """
    if rng.random() >= variation_rate:
        text, kind = name, "CANONICAL"
    else:
        kinds = list(noise)
        probs = np.array([noise[k] for k in kinds], dtype=float)
        kind = str(rng.choice(kinds, p=probs / probs.sum())).upper()
        base = str(rng.choice(aliases)) if aliases else name
        if kind == "ALIAS":
            text = base
        elif kind == "DECORATE":
            kind, text = "DECORATED", base + str(rng.choice(_DECORATIONS))
        else:
            kind, text = "TYPO", _typo(name, rng)
    if rng.random() < _UPPERCASE_RATE:
        text = text.upper()
    return text, kind


# ---------------------------------------------------------------- plantillas de platillo


@dataclass(frozen=True)
class _ItemTemplate:
    """Un platillo tal como lo define su dueño (igual en todas sus sucursales)."""

    canonical_id: str
    name: str
    name_variant: str
    course: str
    protein: str
    spice_level: int
    is_shareable: bool
    served: tuple[str, ...]
    price_cents: int


class _MenuContext:
    """Índices de la biblioteca que se consultan una y otra vez."""

    def __init__(self, library: Library, supply: Mapping[str, Any]) -> None:
        self.library = library
        self.supply = supply
        frames = library.to_frames()
        self.dishes = frames["canonical_dishes"]
        self.by_id = self.dishes.set_index("canonical_id", drop=False)
        names = self.by_id["name"]
        self.aliases = (
            frames["canonical_dish_aliases"]
            .loc[lambda d: d["alias"] != d["canonical_id"].map(names)]
            .groupby("canonical_id")["alias"]
            .apply(list)
            .to_dict()
        )
        self.ingredients = {d.slug: list(d.ingredients) for d in library.dishes}

    def template(
        self, canonical_id: str, multiplier: float, rng: np.random.Generator
    ) -> _ItemTemplate:
        """Crea la definición de un platillo a partir de su canónico."""
        canonical = self.by_id.loc[canonical_id]
        supply = self.supply
        name, kind = vary_name(
            canonical["name"],
            self.aliases.get(canonical_id, []),
            supply["name_variation_rate"],
            supply["name_noise"],
            rng,
        )
        served = list(self.ingredients[canonical_id])
        if len(served) > 2 and rng.random() < supply["recipe_variation_rate"]:
            served.pop(int(rng.integers(1, len(served))))
        noise = rng.lognormal(0.0, supply["price"]["item_sigma"])
        raw = canonical["base_price"] * multiplier * noise
        pesos = psychological_round(
            np.array([raw]), rng, nine_ending_rate=supply["price"]["nine_ending_rate"]
        )[0]
        return _ItemTemplate(
            canonical_id=canonical_id,
            name=name,
            name_variant=kind,
            course=str(canonical["course"]),
            protein=str(canonical["protein"]),
            spice_level=int(canonical["spice_level"]),
            is_shareable=bool(canonical["is_shareable"]),
            served=tuple(served),
            price_cents=int(pesos) * 100,
        )

    def attribute_rows(self, item_id: str, served: Sequence[str]) -> list[dict[str, str]]:
        """Ingredientes, alérgenos y dietas del platillo SERVIDO."""
        lib = self.library
        rows = [{"menu_item_id": item_id, "kind": "ingredient", "value": i} for i in served]
        rows += [
            {"menu_item_id": item_id, "kind": "allergen", "value": a}
            for a in lib.allergens_of(served)
        ]
        rows += [
            {"menu_item_id": item_id, "kind": "diet", "value": d} for d in lib.diet_tags_of(served)
        ]
        return rows


def _item_row(item_id: str, rid: str, tpl: _ItemTemplate) -> dict[str, Any]:
    return {
        "menu_item_id": item_id,
        "restaurant_id": rid,
        "name": tpl.name,
        "course": tpl.course,
        "protein": tpl.protein,
        "spice_level": tpl.spice_level,
        "is_shareable": tpl.is_shareable,
        "price_cents": tpl.price_cents,
        "canonical_true": tpl.canonical_id,
        "name_variant": tpl.name_variant,
    }


# ---------------------------------------------------------------- dueños del menú


def menu_owners(restaurants: pd.DataFrame) -> pd.DataFrame:
    """Dueño del menú de cada restaurante: la cadena o el propio restaurante.

    Returns
    -------
    pd.DataFrame
        ``restaurant_id`` y ``menu_owner`` (``"chain:<marca>"`` o el id).
    """
    if "chain" in restaurants:
        chain = restaurants["chain"]
    else:
        chain = pd.Series(None, index=restaurants.index, dtype=object)
    owner = np.where(
        chain.notna(), "chain:" + chain.astype(str), restaurants["restaurant_id"].astype(str)
    )
    return pd.DataFrame({"restaurant_id": restaurants["restaurant_id"], "menu_owner": owner})


def _menu_size(
    bands: Sequence[str | None], supply: Mapping[str, Any], rng: np.random.Generator
) -> int:
    """Tamaño del menú según la banda de personal (la mayor, si es una cadena)."""
    table = supply["menu_size_by_staff_band"]
    ranges = [
        table.get(b, table["default"]) if isinstance(b, str) else table["default"] for b in bands
    ]
    chosen = max(ranges, key=lambda r: r["max"])
    return int(rng.integers(chosen["min"], chosen["max"] + 1))


def build_menus(
    restaurants: pd.DataFrame,
    library: Library,
    supply: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Menú inicial de cada restaurante.

    Parameters
    ----------
    restaurants : pd.DataFrame
        Con ``restaurant_id``, ``cuisine_pred`` y ``ses_z`` (escolaridad de la
        zona estandarizada). Opcionales: ``staff_band`` y ``chain`` (marca).
    library : Library
        Biblioteca de platillos canónicos.
    supply : Mapping
        Bloque ``supply`` de la configuración.
    rng : np.random.Generator
        Generador aleatorio.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``menu_items``, ``menu_item_attributes`` (del platillo SERVIDO),
        ``ground_truth_menu_items`` (canónico verdadero y tipo de nombre) y
        ``ground_truth_restaurant_pricing`` (dueño y multiplicador de precio).
    """
    ctx = _MenuContext(library, supply)
    owners = menu_owners(restaurants)
    data = restaurants.assign(menu_owner=owners["menu_owner"].to_numpy())
    if "staff_band" not in data:
        data["staff_band"] = None

    price_cfg = supply["price"]
    rows: list[dict[str, Any]] = []
    attrs: list[dict[str, str]] = []
    multipliers: dict[str, float] = {}
    for owner, branch in data.groupby("menu_owner", sort=False):
        mult = float(
            rng.lognormal(
                mean=price_cfg["ses_elasticity"] * float(branch["ses_z"].mean()),
                sigma=price_cfg["restaurant_sigma"],
            )
        )
        multipliers[str(owner)] = mult
        size = _menu_size(branch["staff_band"].tolist(), supply, rng)
        menu = compose_menu(
            str(branch["cuisine_pred"].iloc[0]), ctx.dishes, size=size, supply=supply, rng=rng
        )
        templates = [ctx.template(c, mult, rng) for c in menu]
        for rid in branch["restaurant_id"]:
            for k, tpl in enumerate(templates):
                item_id = f"{rid}-{k:03d}"
                rows.append(_item_row(item_id, rid, tpl))
                attrs.extend(ctx.attribute_rows(item_id, tpl.served))

    items = pd.DataFrame(rows)
    pricing = owners.assign(price_multiplier=owners["menu_owner"].map(multipliers))
    return {
        "menu_items": items[_ITEM_COLUMNS],
        "menu_item_attributes": pd.DataFrame(attrs),
        "ground_truth_menu_items": items[["menu_item_id", "canonical_true", "name_variant"]],
        "ground_truth_restaurant_pricing": pricing,
    }


# ---------------------------------------------------------------- evolución


def _random_moment(
    month_start: pd.Timestamp, rng: np.random.Generator, hour: int = 9
) -> pd.Timestamp:
    """Un día al azar dentro del mes (primeros 27 días) a la hora indicada."""
    return month_start + pd.Timedelta(days=int(rng.integers(0, 27)), hours=hour)


@dataclass
class _OwnerState:
    """Estado de la carta de un dueño durante la evolución."""

    branches: list[str]
    cuisine: str
    multiplier: float
    active: list[str]  # canonical_id vigentes, en orden
    templates: dict[str, _ItemTemplate]  # canonical_id → definición vigente


def _initial_states(
    items: pd.DataFrame, pricing: pd.DataFrame, cuisine: pd.Series
) -> dict[str, _OwnerState]:
    """Reconstruye la carta vigente de cada dueño a partir del menú inicial."""
    states: dict[str, _OwnerState] = {}
    for owner, group in pricing.groupby("menu_owner", sort=False):
        branches = group["restaurant_id"].tolist()
        first = items[items["restaurant_id"] == branches[0]]
        templates = {
            row.canonical_true: _ItemTemplate(
                canonical_id=row.canonical_true,
                name=row.name,
                name_variant=row.name_variant,
                course=row.course,
                protein=row.protein,
                spice_level=int(row.spice_level),
                is_shareable=bool(row.is_shareable),
                served=(),  # sus atributos ya se escribieron en build_menus
                price_cents=int(row.price_cents),
            )
            for row in first.itertuples(index=False)
        }
        states[str(owner)] = _OwnerState(
            branches=branches,
            cuisine=str(cuisine[branches[0]]),
            multiplier=float(group["price_multiplier"].iloc[0]),
            active=first["canonical_true"].tolist(),
            templates=templates,
        )
    return states


def evolve_menus(
    menus: Mapping[str, pd.DataFrame],
    restaurants: pd.DataFrame,
    library: Library,
    supply: Mapping[str, Any],
    calendar: Mapping[str, Any],
    *,
    months: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Hace evolucionar los menús mes a mes y genera su historial.

    Cada mes, por **dueño del menú** (las sucursales de una cadena cambian
    juntas): bajas (``archive_rate``), aumentos de precio
    (``price_change_rate``, solo sobre platillos vigentes) y altas
    (``new_item_rate``). Además, cada platillo se agota ciertos días
    (``daily_sold_out_rate``) y se repone a la mañana siguiente.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``menu_items`` (con ``created_at`` y ``archived_at``),
        ``menu_item_attributes``, ``menu_item_price_history``,
        ``menu_item_availability_events``, ``ground_truth_menu_items`` y
        ``ground_truth_restaurant_pricing``. Fechas en UTC.
    """
    ctx = _MenuContext(library, supply)
    evo = supply["evolution"]
    tz = calendar["timezone"]
    tz_dtype = pd.DatetimeTZDtype(tz=tz)
    start = pd.Timestamp(calendar["start_date"], tz=tz)
    end = start + pd.DateOffset(months=months)
    pricing = menus["ground_truth_restaurant_pricing"]
    cuisine = restaurants.set_index("restaurant_id")["cuisine_pred"]

    items = menus["menu_items"].merge(menus["ground_truth_menu_items"], on="menu_item_id")
    item_of: dict[tuple[str, str], str] = {
        (r, c): i
        for r, c, i in zip(
            items["restaurant_id"], items["canonical_true"], items["menu_item_id"], strict=True
        )
    }
    next_index = items.groupby("restaurant_id").size().to_dict()
    states = _initial_states(items, pricing, cuisine)

    prices = [
        {"menu_item_id": i, "price_cents": p, "valid_from": start}
        for i, p in zip(items["menu_item_id"], items["price_cents"], strict=True)
    ]
    archived: dict[str, pd.Timestamp] = {}
    created: dict[str, pd.Timestamp] = {}
    new_rows: list[dict[str, Any]] = []
    attrs = [menus["menu_item_attributes"]]
    pools = {
        c: ctx.dishes.loc[ctx.dishes["cuisine"].isin([c, "general"]), "canonical_id"].to_numpy()
        for c in {s.cuisine for s in states.values()}
    }

    for m in range(1, months):
        month_start = start + pd.DateOffset(months=m)
        for state in states.values():
            n = len(state.active)
            if n == 0:
                continue

            # 1) Bajas: el mismo día en todas las sucursales del dueño.
            n_archive = int(rng.binomial(n, evo["archive_rate"]))
            gone = {str(c) for c in rng.choice(state.active, size=n_archive, replace=False)}
            for c in gone:
                moment = _random_moment(month_start, rng, hour=23)
                for rid in state.branches:
                    archived[item_of.pop((rid, c))] = moment
            vigentes = [c for c in state.active if c not in gone]

            # 2) Aumentos de precio SOLO sobre lo vigente. (Bug corregido: antes se
            #    elegía también entre lo dado de baja y quedaban precios posteriores
            #    a la baja.)
            n_change = min(int(rng.binomial(n, evo["price_change_rate"])), len(vigentes))
            for c in rng.choice(vigentes, size=n_change, replace=False):
                c = str(c)
                tpl = state.templates[c]
                factor = rng.uniform(evo["price_increase"]["min"], evo["price_increase"]["max"])
                pesos = psychological_round(
                    np.array([tpl.price_cents / 100 * factor]),
                    rng,
                    nine_ending_rate=supply["price"]["nine_ending_rate"],
                )[0]
                new_price = int(pesos) * 100
                if new_price <= tpl.price_cents:
                    continue
                moment = _random_moment(month_start, rng, hour=7)
                for rid in state.branches:
                    prices.append(
                        {
                            "menu_item_id": item_of[(rid, c)],
                            "price_cents": new_price,
                            "valid_from": moment,
                        }
                    )
                state.templates[c] = replace(tpl, price_cents=new_price)

            # 3) Altas: platillos de su cocina o del pool general que no están vigentes.
            vigentes_set = set(vigentes)
            pool = [c for c in pools[state.cuisine] if c not in vigentes_set]
            n_new = min(int(rng.binomial(n, evo["new_item_rate"])), len(pool))
            for c in rng.choice(pool, size=n_new, replace=False):
                c = str(c)
                tpl = ctx.template(c, state.multiplier, rng)
                moment = _random_moment(month_start, rng, hour=8)
                for rid in state.branches:
                    k = next_index[rid]
                    next_index[rid] = k + 1
                    item_id = f"{rid}-{k:03d}"
                    new_rows.append(_item_row(item_id, rid, tpl))
                    attrs.append(pd.DataFrame(ctx.attribute_rows(item_id, tpl.served)))
                    prices.append(
                        {
                            "menu_item_id": item_id,
                            "price_cents": tpl.price_cents,
                            "valid_from": moment,
                        }
                    )
                    created[item_id] = moment
                    item_of[(rid, c)] = item_id
                state.templates[c] = tpl
                vigentes.append(c)
            state.active = vigentes

    if new_rows:
        items = pd.concat([items, pd.DataFrame(new_rows)], ignore_index=True)
    created_at = items["menu_item_id"].map(created).astype(tz_dtype)
    items["created_at"] = created_at.fillna(start)
    items["archived_at"] = items["menu_item_id"].map(archived).astype(tz_dtype)

    price_history = _close_intervals(pd.DataFrame(prices))
    events = _availability_events(items, end, evo["daily_sold_out_rate"], rng)
    truth = items[["menu_item_id", "canonical_true", "name_variant"]]
    out_items = items[[*_ITEM_COLUMNS, "created_at", "archived_at"]].copy()
    for col in ("created_at", "archived_at"):
        out_items[col] = out_items[col].dt.tz_convert("UTC")
    return {
        "menu_items": out_items,
        "menu_item_attributes": pd.concat(attrs, ignore_index=True),
        "menu_item_price_history": price_history,
        "menu_item_availability_events": events,
        "ground_truth_menu_items": truth,
        "ground_truth_restaurant_pricing": pricing,
    }


def _close_intervals(prices: pd.DataFrame) -> pd.DataFrame:
    """Convierte cambios de precio en intervalos ``[valid_from, valid_to)`` contiguos."""
    prices = prices.sort_values(["menu_item_id", "valid_from"]).reset_index(drop=True)
    prices["valid_to"] = prices.groupby("menu_item_id")["valid_from"].shift(-1)
    for col in ("valid_from", "valid_to"):
        prices[col] = pd.to_datetime(prices[col]).dt.tz_convert("UTC")
    return prices[["menu_item_id", "price_cents", "valid_from", "valid_to"]]


def _availability_events(
    items: pd.DataFrame, end: pd.Timestamp, sold_out_rate: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Genera CREATED, pares SOLD_OUT/RESTOCKED y ARCHIVED de cada platillo."""
    rows: list[tuple[str, pd.Timestamp, str, bool]] = []
    for item_id, created, archived in zip(
        items["menu_item_id"], items["created_at"], items["archived_at"], strict=True
    ):
        rows.append((item_id, created, "CREATED", True))
        last = archived if pd.notna(archived) else end
        first_day = created.normalize() + pd.Timedelta(days=1)
        n_days = (last.normalize() - first_day).days - 1  # el restock debe caer antes del final
        if n_days > 0:
            k = int(rng.binomial(n_days, sold_out_rate))
            for day in np.sort(rng.choice(n_days, size=k, replace=False)):
                date = first_day + pd.Timedelta(days=int(day))
                sold = date + pd.Timedelta(hours=float(rng.uniform(13, 22)))
                rows.append((item_id, sold, "SOLD_OUT", False))
                rows.append((item_id, date + pd.Timedelta(days=1, hours=8), "RESTOCKED", True))
        if pd.notna(archived):
            rows.append((item_id, archived, "ARCHIVED", False))
    events = pd.DataFrame(rows, columns=["menu_item_id", "occurred_at", "reason", "is_available"])
    events["occurred_at"] = pd.to_datetime(events["occurred_at"]).dt.tz_convert("UTC")
    return events
