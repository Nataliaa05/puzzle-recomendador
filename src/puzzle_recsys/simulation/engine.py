"""Motor Monte Carlo: visitas, mesas y pedidos de los comensales durante el año.

Algoritmo, por comensal (los comensales son independientes entre sí, así que
el orden de simulación no altera el resultado):

1. **Cuándo sale:** proceso de renovación con tiempos entre visitas Weibull,
   escalados por la frecuencia propia del comensal (lognormal: pocos muy
   frecuentes). El día de la semana se ajusta por *thinning* (aceptación-
   rechazo con el perfil semanal) y la hora sale del perfil horario empírico.
2. **A dónde va:** conjunto de consideración = sus K restaurantes más cercanos
   (``BallTree`` haversine) + los que ya frecuenta; elige con logit mixto
   (cocina, distancia, precio, lealtad, gusto por variar).
3. **Con quién:** ``pax`` de la distribución empírica; los acompañantes son
   otros usuarios registrados o personas anónimas (pool de gustos aparte).
4. **Qué pide cada quien:** por tipo de plato (hábitos Beta), entre lo
   **disponible a esa hora** (vigente y no agotado), con filtros duros de
   alergias y dieta, por logit mixto.
5. **Qué se comparte:** con probabilidad creciente en ``pax``; utilidad de grupo
   (promedio + miseria mínima) y ningún alérgeno de nadie en la mesa.

Todas las horas se manejan en hora local (CDMX no tiene horario de verano
desde 2022) y se convierten a UTC al final.

Usado en: notebooks 03.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from puzzle_recsys.catalog.library import Library
from puzzle_recsys.simulation.choice import (
    DinerVector,
    ItemFeatures,
    dish_utility,
    feasible,
    gumbel_choice,
    ideal_prices,
    restaurant_utility,
    shared_group_utility,
)
from puzzle_recsys.simulation.population import TASTE_COURSES, DinerTastes, drift_cuisine_affinity

_DAY_NS = 86_400 * 10**9
_HOUR_NS = 3_600 * 10**9
_NEVER = np.iinfo(np.int64).max
_EARTH_KM = 6_371.0


# ---------------------------------------------------------------- índice del mundo


@dataclass
class WorldIndex:
    """Todo lo que el motor consulta en cada visita, precalculado en arreglos."""

    items: ItemFeatures
    item_ids: np.ndarray
    created: np.ndarray  # ns, hora local
    archived: np.ndarray  # ns, hora local (_NEVER si sigue vigente)
    price_from: list[np.ndarray]
    price_value: list[np.ndarray]  # pesos
    soldout_from: dict[int, np.ndarray]
    soldout_to: dict[int, np.ndarray]
    rest_ids: np.ndarray
    rest_items: list[np.ndarray]
    rest_latlng: np.ndarray  # radianes
    rest_cuisine_idx: np.ndarray
    rest_price_level: np.ndarray  # ticket esperado por persona (pesos)
    course_reference: dict[str, float]
    cuisines: tuple[str, ...]


def _local_ns(series: pd.Series, tz: str) -> np.ndarray:
    """Timestamps (UTC) → nanosegundos en hora local; NaT → _NEVER."""
    local = pd.to_datetime(series).dt.tz_convert(tz).dt.tz_localize(None)
    values = local.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    values[local.isna().to_numpy()] = _NEVER
    return values


def build_world_index(
    tables: Mapping[str, pd.DataFrame],
    library: Library,
    cuisines: list[str],
    choice_cfg: Mapping[str, Any],
    tz: str,
    rng: np.random.Generator,
    course_habit_means: Mapping[str, float] | None = None,
) -> tuple[WorldIndex, pd.DataFrame]:
    """Precalcula atributos, vigencias, precios y agotados de los platillos.

    El **nivel de precio** de cada restaurante es el **ticket esperado por
    persona**: ``Σ_tipo P(pedir ese tipo) · precio mediano de ese tipo en su
    menú``. Así es comparable con el presupuesto TOTAL por visita del comensal
    (antes se comparaba con el precio del plato fuerte: escalas distintas).

    También sortea el **atractivo latente** de cada platillo canónico (cola
    larga: ``-alpha·log(rango)`` dentro de su tipo de plato) y la **calidad de
    ejecución** de cada platillo servido (Laplace). Ambos son verdad oculta.

    Returns
    -------
    tuple[WorldIndex, pd.DataFrame]
        El índice y la tabla ``ground_truth/canonical_appeal``.
    """
    items = tables["menu_items"].merge(tables["ground_truth_menu_items"], on="menu_item_id")
    dishes = tables["canonical_dishes"].set_index("canonical_id")

    appeal = pd.Series(0.0, index=dishes.index)
    for _, group in dishes.groupby("course"):
        ranks = rng.permutation(len(group)) + 1
        appeal[group.index] = -choice_cfg["popularity_alpha"] * np.log(ranks)

    cuisine_pos = {c: k for k, c in enumerate(cuisines)}
    proteins = [
        p for p in ("BEEF", "PORK", "CHICKEN", "FISH", "SEAFOOD", "PLANT", "EGG", "CHEESE", "MIXED")
    ]
    protein_pos = {p: k for k, p in enumerate(proteins)}
    ingredient_pos = {name: k for k, name in enumerate(library.ingredients)}
    allergen_pos = {a: k for k, a in enumerate(library.allergens)}

    pos = {item_id: k for k, item_id in enumerate(items["menu_item_id"])}
    attrs = tables["menu_item_attributes"]
    ing = attrs[attrs["kind"] == "ingredient"]
    ingredients = np.zeros((len(items), len(ingredient_pos)), dtype=np.int8)
    ingredients[
        ing["menu_item_id"].map(pos).to_numpy(), ing["value"].map(ingredient_pos).to_numpy()
    ] = 1
    alg = attrs[attrs["kind"] == "allergen"]
    allergens = np.zeros((len(items), len(allergen_pos)), dtype=bool)
    allergens[
        alg["menu_item_id"].map(pos).to_numpy(), alg["value"].map(allergen_pos).to_numpy()
    ] = True
    diet = attrs[attrs["kind"] == "diet"]
    vegetarian = np.zeros(len(items), dtype=bool)
    vegan = np.zeros(len(items), dtype=bool)
    vegetarian[diet.loc[diet["value"] == "vegetariano", "menu_item_id"].map(pos).to_numpy()] = True
    vegan[diet.loc[diet["value"] == "vegano", "menu_item_id"].map(pos).to_numpy()] = True

    canon_cuisine = items["canonical_true"].map(dishes["cuisine"])
    features = ItemFeatures(
        cuisine_idx=canon_cuisine.map(cuisine_pos).fillna(-1).astype(int).to_numpy(),
        protein_idx=items["protein"].map(protein_pos).fillna(-1).astype(int).to_numpy(),
        ingredients=ingredients,
        spice=items["spice_level"].to_numpy(),
        course=items["course"].to_numpy(),
        shareable=items["is_shareable"].to_numpy(dtype=bool),
        allergens=allergens,
        vegetarian=vegetarian,
        vegan=vegan,
        appeal=items["canonical_true"].map(appeal).to_numpy(),
        quality=rng.laplace(0.0, choice_cfg["item_quality_laplace_scale"], len(items)),
    )

    price = tables["menu_item_price_history"]
    price = price.assign(k=price["menu_item_id"].map(pos), t=_local_ns(price["valid_from"], tz))
    price = price.sort_values(["k", "t"])
    price_from = [np.empty(0, dtype=np.int64)] * len(items)
    price_value = [np.empty(0)] * len(items)
    for k, group in price.groupby("k", sort=False):
        price_from[int(k)] = group["t"].to_numpy()
        price_value[int(k)] = group["price_cents"].to_numpy() / 100

    ev = tables["menu_item_availability_events"]
    ev = ev[ev["reason"].isin(["SOLD_OUT", "RESTOCKED"])]
    ev = ev.assign(k=ev["menu_item_id"].map(pos), t=_local_ns(ev["occurred_at"], tz)).sort_values(
        ["k", "t"]
    )
    soldout_from: dict[int, np.ndarray] = {}
    soldout_to: dict[int, np.ndarray] = {}
    for k, group in ev.groupby("k", sort=False):
        soldout_from[int(k)] = group.loc[group["reason"] == "SOLD_OUT", "t"].to_numpy()
        soldout_to[int(k)] = group.loc[group["reason"] == "RESTOCKED", "t"].to_numpy()

    restaurants = tables["restaurants"].reset_index(drop=True)
    rest_pos = {r: k for k, r in enumerate(restaurants["restaurant_id"])}
    item_rest = items["restaurant_id"].map(rest_pos).to_numpy()
    rest_items = [np.flatnonzero(item_rest == k) for k in range(len(restaurants))]
    habits = dict(course_habit_means or {"MAIN": 1.0})
    course_median = (
        items.pivot_table(
            index="restaurant_id", columns="course", values="price_cents", aggfunc="median"
        ).fillna(0)
        / 100
    )
    level = sum(course_median[c] * p for c, p in habits.items() if c in course_median)
    base = dishes.groupby("course")["base_price"].median()

    index = WorldIndex(
        items=features,
        item_ids=items["menu_item_id"].to_numpy(),
        created=_local_ns(items["created_at"], tz),
        archived=_local_ns(items["archived_at"], tz),
        price_from=price_from,
        price_value=price_value,
        soldout_from=soldout_from,
        soldout_to=soldout_to,
        rest_ids=restaurants["restaurant_id"].to_numpy(),
        rest_items=rest_items,
        rest_latlng=np.radians(restaurants[["lat", "lng"]].to_numpy()),
        rest_cuisine_idx=restaurants["cuisine_pred"]
        .map(cuisine_pos)
        .fillna(-1)
        .astype(int)
        .to_numpy(),
        rest_price_level=restaurants["restaurant_id"].map(level).fillna(level.median()).to_numpy(),
        course_reference=(base / base["MAIN"]).to_dict(),
        cuisines=tuple(cuisines),
    )
    canonical_appeal = appeal.rename("appeal").reset_index()
    return index, canonical_appeal


def available_at(index: WorldIndex, cand: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Platillos vigentes y no agotados en el instante ``t`` y su precio vigente.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Índices disponibles y su precio (pesos) en ``t``.
    """
    live = cand[(index.created[cand] <= t) & (t < index.archived[cand])]
    keep, prices = [], []
    for k in live:
        k = int(k)
        so = index.soldout_from.get(k)
        if so is not None and len(so):
            j = np.searchsorted(so, t, side="right") - 1
            if j >= 0 and t < index.soldout_to[k][j]:
                continue
        pf = index.price_from[k]
        j = np.searchsorted(pf, t, side="right") - 1
        if j < 0:
            continue
        keep.append(k)
        prices.append(index.price_value[k][j])
    return np.asarray(keep, dtype=np.int64), np.asarray(prices, dtype=float)


# ---------------------------------------------------------------- simulación


def haversine_km(origin_rad: np.ndarray, targets_rad: np.ndarray) -> np.ndarray:
    """Distancia de gran círculo (km) de un punto a varios, todo en radianes (lat, lng)."""
    dlat = targets_rad[:, 0] - origin_rad[0]
    dlng = targets_rad[:, 1] - origin_rad[1]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(origin_rad[0]) * np.cos(targets_rad[:, 0]) * np.sin(dlng / 2) ** 2
    )
    return 2 * _EARTH_KM * np.arcsin(np.sqrt(a))


@dataclass
class Agenda:
    """Horarios ocupados de cada comensal registrado: nadie está en dos mesas a la vez.

    El motor recorre a los comensales uno por uno, pero un comensal puede aparecer
    como acompañante en la mesa de otro **antes o después** de que se simulen sus
    propias visitas. Por eso la agenda es global y se consulta en ambos casos.
    """

    window_ns: int
    _busy: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))

    def is_free(self, diner: int, t: int) -> bool:
        """Indica si ``diner`` no tiene otra mesa a menos de ``window_ns`` de ``t``."""
        times = self._busy.get(diner)
        if not times:
            return True
        k = bisect_left(times, t)
        before_ok = k == 0 or t - times[k - 1] >= self.window_ns
        after_ok = k == len(times) or times[k] - t >= self.window_ns
        return before_ok and after_ok

    def book(self, diner: int, t: int) -> None:
        """Registra que ``diner`` abrió (o se sentó en) una mesa en ``t``."""
        insort(self._busy[diner], t)


@dataclass
class _Ledger:
    """Acumula las filas de salida (listas de dicts, se convierten al final)."""

    sessions: list[dict[str, Any]]
    guests: list[dict[str, Any]]
    lines: list[dict[str, Any]]
    visit_truth: list[dict[str, Any]]


def _diner_vector(tastes: DinerTastes, i: int, affinity: np.ndarray, budget: float) -> DinerVector:
    return DinerVector(
        cuisine_affinity=affinity,
        protein_affinity=tastes.protein_affinity[i],
        ingredient_pref=tastes.ingredient_pref[i],
        spice_tolerance=int(tastes.spice_tolerance[i]),
        price_sensitivity=float(tastes.price_sensitivity[i]),
        budget=float(budget),
        diet=str(tastes.diet[i]),
        allergies=tastes.allergies[i],
    )


def visit_days(
    start: int,
    end: int,
    scale_days: float,
    shape: float,
    weekday_accept: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    """Días de visita de un comensal: renovación Weibull + *thinning* semanal.

    Los días candidatos salen de un proceso de renovación; cada candidato se
    acepta con probabilidad ``weekday_accept[día]``. Si se rechaza, se sortea
    **un nuevo intervalo** (no se pasa al día siguiente: eso arrastraba visitas
    de un día al otro y deformaba el perfil semanal). La escala se multiplica
    por la aceptación media para conservar la frecuencia esperada.

    Returns
    -------
    list[int]
        Medianoche (ns) de cada día con visita, en orden.
    """
    scale = scale_days * float(np.mean(weekday_accept))
    t = start + int(rng.uniform(0, scale)) * _DAY_NS
    days: list[int] = []
    while t < end:
        day = t - t % _DAY_NS
        weekday = int((day // _DAY_NS + 3) % 7)  # 1970-01-01 fue jueves (=3)
        if rng.random() < weekday_accept[weekday]:
            days.append(day)
        t = day + max(1, round(rng.weibull(shape) * scale)) * _DAY_NS
    return days


def menu_typical_prices(courses: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Precio mediano de cada tipo de plato en el menú disponible (alineado por platillo)."""
    typical = np.empty(len(prices))
    for course in np.unique(courses):
        mask = courses == course
        typical[mask] = np.median(prices[mask])
    return typical


def _for_visit(vec: DinerVector, budget_mult: float, lam_mult: float) -> DinerVector:
    """Ajusta presupuesto y sensibilidad al precio de un comensal para esta visita."""
    return replace(
        vec, budget=vec.budget * budget_mult, price_sensitivity=vec.price_sensitivity * lam_mult
    )


def simulate_visits(
    index: WorldIndex,
    tastes: DinerTastes,
    homes: pd.DataFrame,
    budgets: np.ndarray,
    anonymous: tuple[DinerTastes, np.ndarray],
    cfg: Mapping[str, Mapping[str, Any]],
    calendar: Mapping[str, Any],
    *,
    months: int,
    rng: np.random.Generator,
    frequency: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Simula un año de visitas de toda la población registrada.

    Parameters
    ----------
    index : WorldIndex
        Restaurantes y platillos precalculados.
    tastes : DinerTastes
        Gustos latentes de los comensales registrados (verdad oculta).
    homes : pd.DataFrame
        ``lat``/``lng`` de cada comensal (misma fila que ``tastes``).
    budgets : np.ndarray
        Presupuesto por visita de cada comensal.
    anonymous : tuple[DinerTastes, np.ndarray]
        Pool de gustos y presupuestos de acompañantes no registrados.
    cfg : Mapping
        Configuración con los bloques ``visits``, ``choice``, ``sharing``,
        ``population``, ``ratings`` y ``noise``.
    calendar : Mapping
        ``start_date`` y ``timezone``.
    months : int
        Meses simulados.
    rng : np.random.Generator
        Generador aleatorio.
    frequency : np.ndarray, optional
        Frecuencia relativa de cada comensal derivada de su gasto (1 = típica).
        Se multiplica por un ruido lognormal individual.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``dining_sessions``, ``guests``, ``order_items`` (con columnas de verdad
        ``utility``/``experienced`` que se separan después) y
        ``ground_truth_visits`` (restaurante elegido, su utilidad y si fue una
        ocasión especial).
    """
    vcfg, ccfg, scfg = cfg["visits"], cfg["choice"], cfg["sharing"]
    occ = ccfg["occasion"]
    tz = calendar["timezone"]
    start = pd.Timestamp(calendar["start_date"]).value
    end = (pd.Timestamp(calendar["start_date"]) + pd.DateOffset(months=months)).value
    n = len(tastes)
    anon_tastes, anon_budgets = anonymous

    # Conjunto de consideración geográfico (fijo: la casa no se mueve en el año).
    tree = BallTree(index.rest_latlng, metric="haversine")
    k = min(vcfg["consideration_set_size"], len(index.rest_ids))
    home_rad = np.radians(homes[["lat", "lng"]].to_numpy())
    dist, near = tree.query(home_rad, k=k)
    dist_km = dist * _EARTH_KM

    # Acompañantes registrados: vecinos del anfitrión (KNN sobre las casas).
    k_friends = min(vcfg["companion_neighbors"] + 1, n)
    _, neighbors = BallTree(home_rad, metric="haversine").query(home_rad, k=k_friends)

    # Deriva mensual de la afinidad por cocina.
    share = np.bincount(
        index.rest_cuisine_idx[index.rest_cuisine_idx >= 0], minlength=len(index.cuisines)
    )
    alpha = cfg["population"]["taste"]["cuisine_concentration"] * share / share.sum() + 0.02
    affinity = [tastes.cuisine_affinity]
    for _ in range(1, months):
        affinity.append(
            drift_cuisine_affinity(
                affinity[-1], alpha, cfg["population"]["taste"]["taste_drift_blend"], rng
            )
        )

    weekday_w = np.asarray(vcfg["weekday_weights"], dtype=float)
    weekday_accept = weekday_w / weekday_w.max()
    hours = np.array([int(h) for h in vcfg["hourly_profile"]])
    hour_p = np.array(list(vcfg["hourly_profile"].values()), dtype=float)
    hour_p /= hour_p.sum()
    pax_values = np.array([int(p) for p in vcfg["pax_probs"]])
    pax_p = np.array(list(vcfg["pax_probs"].values()), dtype=float)
    pax_p /= pax_p.sum()
    base_frequency = np.ones(n) if frequency is None else np.asarray(frequency, dtype=float)
    frequency = base_frequency * rng.lognormal(0.0, vcfg["frequency_lognormal_sigma"], n)
    weib = vcfg["inter_visit_weibull"]
    habit_pos = {c: j for j, c in enumerate(TASTE_COURSES)}
    course_ref = np.vectorize(lambda c: index.course_reference.get(c, 1.0))
    ledger = _Ledger([], [], [], [])
    noise_cfg = cfg["noise"]
    w_budget = ccfg["price_budget_weight"]
    # Para ocasiones especiales: los restaurantes más caros de la ciudad (orden descendente).
    pricey = np.argsort(-index.rest_price_level)[: occ["pricey_pool"]]
    agenda = Agenda(window_ns=int(vcfg["min_hours_between_visits"] * _HOUR_NS))

    for i in range(n):
        visits: dict[int, int] = defaultdict(int)
        days = visit_days(
            start, end, weib["scale_days"] / frequency[i], weib["shape"], weekday_accept, rng
        )
        for day in days:
            weekday = int((day // _DAY_NS + 3) % 7)
            visit_t = (
                day
                + int(rng.choice(hours, p=hour_p)) * _HOUR_NS
                + int(rng.integers(0, 60)) * 60 * 10**9
            )
            month = min(int((visit_t - start) / (30.44 * _DAY_NS)), months - 1)
            # Ya está sentado en otra mesa como acompañante: esa visita no ocurre.
            if not agenda.is_free(i, visit_t):
                continue

            # 0) ¿Ocasión especial (aniversario, cumpleaños)? Más presupuesto y menos
            #    sensibilidad al precio para TODA la mesa.
            p_occ = occ["probability"] * (occ["weekend_factor"] if weekday in (4, 5) else 1.0)
            occasion = bool(rng.random() < p_occ)
            budget_mult = (
                rng.uniform(occ["budget_multiplier"]["min"], occ["budget_multiplier"]["max"])
                if occasion
                else 1.0
            )
            lam_mult = occ["price_sensitivity_factor"] if occasion else 1.0
            # En una ocasión especial se acepta trasladarse más lejos.
            dist_mult = occ["distance_sensitivity_factor"] if occasion else 1.0
            base_vec = _diner_vector(tastes, i, affinity[month][i], budgets[i])
            # Al ELEGIR LUGAR cuenta el presupuesto más alto con la sensibilidad normal
            # (si también se bajara λ, ambos efectos se cancelaban: bug de la 1.ª versión).
            chooser = _for_visit(base_vec, budget_mult, 1.0)
            # Al PEDIR, ya en el lugar, se relaja la sensibilidad al precio.
            host = _for_visit(base_vec, budget_mult, lam_mult)

            # 1) restaurante
            cand = near[i][dist_km[i] <= vcfg["max_distance_km"]]
            if len(cand) == 0:
                cand = near[i][:5]
            frequent = sorted(visits, key=visits.get, reverse=True)[:5]
            extra = np.empty(0, dtype=np.int64)
            if occasion:  # para un aniversario la gente sí se mueve: entran lugares caros
                d_pricey = haversine_km(home_rad[i], index.rest_latlng[pricey])
                extra = pricey[d_pricey <= vcfg["max_distance_km"]][: occ["extra_candidates"]]
            cand = np.unique(np.concatenate([cand, np.asarray(frequent, dtype=np.int64), extra]))
            d_km = haversine_km(home_rad[i], index.rest_latlng[cand])
            u_rest = restaurant_utility(
                index.rest_cuisine_idx[cand],
                d_km,
                index.rest_price_level[cand],
                np.array([visits.get(int(c), 0) for c in cand]),
                chooser,
                float(tastes.distance_sensitivity[i]) * dist_mult,
                float(tastes.variety_seeking[i]),
                ccfg,
            )
            pick = gumbel_choice(u_rest, ccfg["gumbel_scale"], rng)
            r = int(cand[pick])
            avail, prices = available_at(index, index.rest_items[r], visit_t)
            if len(avail) == 0:
                continue
            visits[r] += 1
            agenda.book(i, visit_t)

            # 2) la mesa
            if rng.random() < noise_cfg["outlier_party_rate"]:
                pax = int(
                    rng.integers(
                        noise_cfg["outlier_party_size"]["min"],
                        noise_cfg["outlier_party_size"]["max"] + 1,
                    )
                )
            else:
                pax = int(rng.choice(pax_values, p=pax_p))
            session_id = f"s{len(ledger.sessions):07d}"
            members: list[tuple[str | None, DinerVector, np.ndarray, float]] = [
                (f"d{i:06d}", host, tastes.course_habit[i], float(tastes.sharer[i]))
            ]
            taste_refs = [f"d{i:06d}"]
            at_table = {i}
            for _ in range(pax - 1):
                friend = -1
                if rng.random() < vcfg["registered_companion_rate"] and k_friends > 1:
                    friend = int(neighbors[i][rng.integers(1, k_friends)])
                    # Si ya está en esta mesa o en otra a esa hora, viene alguien no registrado.
                    if friend in at_table or not agenda.is_free(friend, visit_t):
                        friend = -1
                if friend >= 0:
                    j = friend
                    at_table.add(j)
                    agenda.book(j, visit_t)
                    vec = _for_visit(
                        _diner_vector(tastes, j, affinity[month][j], budgets[j]),
                        budget_mult,
                        lam_mult,
                    )
                    members.append(
                        (f"d{j:06d}", vec, tastes.course_habit[j], float(tastes.sharer[j]))
                    )
                    taste_refs.append(f"d{j:06d}")
                else:
                    j = int(rng.integers(len(anon_tastes)))
                    vec = _for_visit(
                        _diner_vector(
                            anon_tastes, j, anon_tastes.cuisine_affinity[j], anon_budgets[j]
                        ),
                        budget_mult,
                        lam_mult,
                    )
                    members.append(
                        (None, vec, anon_tastes.course_habit[j], float(anon_tastes.sharer[j]))
                    )
                    taste_refs.append(f"a{j:05d}")
            ledger.sessions.append(
                {
                    "session_id": session_id,
                    "restaurant_id": index.rest_ids[r],
                    "opened_at": visit_t,
                    "pax": pax,
                }
            )
            ledger.visit_truth.append(
                {
                    "session_id": session_id,
                    "host_diner_id": f"d{i:06d}",
                    "restaurant_utility": float(u_rest[pick]),
                    "occasion": occasion,
                }
            )
            guest_ids = []
            for g, (diner_id, _, _, _) in enumerate(members):
                guest_id = f"{session_id}-g{g}"
                guest_ids.append(guest_id)
                ledger.guests.append(
                    {
                        "guest_id": guest_id,
                        "session_id": session_id,
                        "diner_id": diner_id,
                        "is_host": g == 0,
                        "taste_ref": taste_refs[g],  # verdad oculta: de quién son los gustos
                    }
                )

            # 3) pedidos individuales: precio ideal anclado a ESTE menú
            courses = index.items.course[avail]
            refs = course_ref(courses)
            typical = menu_typical_prices(courses, prices)
            for g, (_, vec, habit, _) in enumerate(members):
                ideal = ideal_prices(typical, vec.budget, refs, w_budget)
                u_all = dish_utility(index.items, avail, prices, ideal, vec, ccfg)
                ok = feasible(index.items, avail, vec)
                for course in ("MAIN", *TASTE_COURSES):
                    p_course = (
                        vcfg["main_course_rate"] if course == "MAIN" else habit[habit_pos[course]]
                    )
                    if rng.random() >= p_course:
                        continue
                    mask = ok & (courses == course)
                    if not mask.any():
                        continue
                    options = np.flatnonzero(mask)
                    choice = options[gumbel_choice(u_all[options], ccfg["gumbel_scale"], rng)]
                    _add_line(
                        ledger,
                        session_id,
                        guest_ids[g],
                        avail,
                        prices,
                        choice,
                        u_all[choice],
                        visit_t,
                        False,
                        cfg,
                        rng,
                    )

            # 4) para compartir: se reparte SOLO entre quienes pueden comerlo (≥ 2).
            #    Quien es alérgico o tiene una dieta incompatible no come de ese plato,
            #    pero no impide que los demás lo compartan.
            p_share = min(0.9, scfg["base_probability"] + scfg["pax_slope"] * (pax - 1))
            if pax >= 2 and rng.random() < p_share:
                options = np.flatnonzero(index.items.shareable[avail])
                if len(options):
                    utilities = np.vstack(
                        [
                            dish_utility(
                                index.items,
                                avail[options],
                                prices[options],
                                ideal_prices(
                                    typical[options], m[1].budget, refs[options], w_budget
                                ),
                                m[1],
                                ccfg,
                            )
                            for m in members
                        ]
                    )
                    eligible = np.vstack(
                        [feasible(index.items, avail[options], m[1]) for m in members]
                    )
                    u_group = shared_group_utility(
                        utilities,
                        eligible,
                        least_misery_weight=scfg["least_misery_weight"],
                        exclusion_penalty=scfg["exclusion_penalty"],
                    )
                    if np.isfinite(u_group).any():
                        chosen = gumbel_choice(u_group, ccfg["gumbel_scale"], rng)
                        eaters = np.flatnonzero(eligible[:, chosen])
                        sharer_w = np.array([members[g][3] for g in eaters]) + 1e-9
                        orderer = int(rng.choice(eaters, p=sharer_w / sharer_w.sum()))
                        _add_line(
                            ledger,
                            session_id,
                            guest_ids[orderer],
                            avail,
                            prices,
                            options[chosen],
                            float(u_group[chosen]),
                            visit_t,
                            True,
                            cfg,
                            rng,
                            shared_by=[guest_ids[g] for g in eaters],
                        )

    return _to_frames(ledger, index, tz)


def _add_line(
    ledger: _Ledger,
    session_id: str,
    guest_id: str,
    avail: np.ndarray,
    prices: np.ndarray,
    choice: int,
    utility: float,
    visit_t: int,
    shared: bool,
    cfg: Mapping[str, Mapping[str, Any]],
    rng: np.random.Generator,
    shared_by: list[str] | None = None,
) -> None:
    """Agrega una línea de pedido con su utilidad esperada y la vivida (verdad).

    ``shared_by`` (verdad oculta): invitados que comparten el platillo.
    """
    ledger.lines.append(
        {
            "order_item_id": f"{session_id}-l{len(ledger.lines):08d}",
            "session_id": session_id,
            "guest_id": guest_id,
            "menu_item_index": int(avail[choice]),
            "quantity": 1,
            "unit_price_cents": round(prices[choice] * 100),
            "is_shared": shared,
            "created_at": visit_t + int(rng.integers(5, 40)) * 60 * 10**9,
            "shared_by": shared_by or [],
            "utility": float(utility),
            "experienced": float(
                utility + rng.laplace(0.0, cfg["ratings"]["experience_noise_laplace"])
            ),
        }
    )


def _to_frames(ledger: _Ledger, index: WorldIndex, tz: str) -> dict[str, pd.DataFrame]:
    """Listas → DataFrames, con horas locales convertidas a UTC."""

    def utc(values: pd.Series) -> pd.Series:
        return pd.to_datetime(values).dt.tz_localize(tz).dt.tz_convert("UTC")

    sessions = pd.DataFrame(ledger.sessions)
    sessions["opened_at"] = utc(sessions["opened_at"])
    lines = pd.DataFrame(ledger.lines)
    lines["created_at"] = utc(lines["created_at"])
    lines.insert(3, "menu_item_id", index.item_ids[lines.pop("menu_item_index").to_numpy()])
    return {
        "dining_sessions": sessions,
        "guests": pd.DataFrame(ledger.guests),
        "order_items": lines,
        "ground_truth_visits": pd.DataFrame(ledger.visit_truth),
    }
