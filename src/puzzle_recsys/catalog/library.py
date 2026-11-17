"""Biblioteca de platillos canónicos: carga, validación y tablas del contrato.

La biblioteca (``data/external/canonical_dishes.yaml``) está escrita a mano,
así que se valida estrictamente al cargarla: ingredientes inexistentes,
alérgenos desconocidos, slugs duplicados o rangos inválidos fallan de
inmediato, no a mitad de la simulación.

Los **alérgenos de un platillo se derivan de sus ingredientes**: hay un solo
lugar de verdad, por lo que no pueden contradecirse.

Usado en: notebooks 02, 09.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from puzzle_recsys.catalog.cuisine_imputation import normalize_name

Course = Literal["STARTER", "MAIN", "SIDE", "DESSERT", "DRINK", "OTHER"]
_NON_MEAT_PROTEINS = frozenset({"huevo", "tofu"})

Protein = Literal[
    "BEEF", "PORK", "CHICKEN", "FISH", "SEAFOOD", "PLANT", "EGG", "CHEESE", "MIXED", "NONE"
]


class Ingredient(BaseModel):
    """Ingrediente del catálogo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group: str
    allergens: tuple[str, ...] = ()


class Dish(BaseModel):
    """Platillo canónico."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slug: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    cuisine: str
    course: Course
    protein: Protein
    spice: int = Field(ge=0, le=3)
    shareable: bool = False
    price: float = Field(gt=0, description="MXN, restaurante de nivel medio en CDMX")
    ingredients: tuple[str, ...] = Field(min_length=1)
    aliases: tuple[str, ...] = ()


class Library(BaseModel):
    """Biblioteca completa, validada."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    allergens: tuple[str, ...]
    ingredients: dict[str, Ingredient]
    dishes: tuple[Dish, ...]

    @model_validator(mode="after")
    def _check_consistency(self) -> Library:
        allowed = set(self.allergens)
        for slug, ing in self.ingredients.items():
            unknown = set(ing.allergens) - allowed
            if unknown:
                raise ValueError(f"ingrediente '{slug}' usa alérgeno(s) desconocido(s): {unknown}")
        seen: set[str] = set()
        for dish in self.dishes:
            if dish.slug in seen:
                raise ValueError(f"slug duplicado: '{dish.slug}'")
            seen.add(dish.slug)
            missing = [i for i in dish.ingredients if i not in self.ingredients]
            if missing:
                raise ValueError(f"'{dish.slug}' usa ingrediente(s) inexistente(s): {missing}")
        return self

    def allergens_of(self, ingredients: Sequence[str]) -> list[str]:
        """Alérgenos de una lista de ingredientes (unión, en el orden del catálogo)."""
        present = {a for i in ingredients for a in self.ingredients[i].allergens}
        return [a for a in self.allergens if a in present]

    def diet_tags_of(self, ingredients: Sequence[str]) -> list[str]:
        """Etiquetas de dieta de una lista de ingredientes.

        - ``vegetariano``: sin carne, pescado ni mariscos (ingredientes del
          grupo ``proteina`` salvo huevo y tofu).
        - ``vegano``: vegetariano y sin lácteos ni huevo.
        - ``sin_gluten``: ningún ingrediente con gluten.
        """
        meat = any(
            self.ingredients[i].group == "proteina" and i not in _NON_MEAT_PROTEINS
            for i in ingredients
        )
        allergens = set(self.allergens_of(ingredients))
        tags = []
        if not meat:
            tags.append("vegetariano")
            if not allergens & {"lacteos", "huevo"}:
                tags.append("vegano")
        if "gluten" not in allergens:
            tags.append("sin_gluten")
        return tags

    def dish_allergens(self, dish: Dish) -> list[str]:
        """Alérgenos del platillo canónico, derivados de sus ingredientes."""
        return self.allergens_of(dish.ingredients)

    def dish_diet_tags(self, dish: Dish) -> list[str]:
        """Etiquetas de dieta del platillo canónico, derivadas de sus ingredientes."""
        return self.diet_tags_of(dish.ingredients)

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """Tablas del contrato: platillos, alias e ingredientes por platillo."""
        dishes = pd.DataFrame(
            [
                {
                    "canonical_id": d.slug,
                    "name": d.name,
                    "cuisine": d.cuisine,
                    "course": d.course,
                    "protein": d.protein,
                    "spice_level": d.spice,
                    "is_shareable": d.shareable,
                    "base_price": d.price,
                    "allergens": self.dish_allergens(d),
                    "diet_tags": self.dish_diet_tags(d),
                }
                for d in self.dishes
            ]
        )
        aliases = pd.DataFrame(
            [
                {"canonical_id": d.slug, "alias": a, "alias_normalized": normalize_name(a)}
                for d in self.dishes
                for a in (d.name, *d.aliases)
            ]
        ).drop_duplicates(["canonical_id", "alias_normalized"])
        ingredients = pd.DataFrame(
            [
                {
                    "canonical_id": d.slug,
                    "ingredient": i,
                    "group": self.ingredients[i].group,
                    "is_main": pos == 0,
                }
                for d in self.dishes
                for pos, i in enumerate(d.ingredients)
            ]
        )
        return {
            "canonical_dishes": dishes,
            "canonical_dish_aliases": aliases.reset_index(drop=True),
            "canonical_dish_ingredients": ingredients,
        }


def load_library(path: Path) -> Library:
    """Carga y valida la biblioteca de platillos canónicos."""
    with path.open(encoding="utf-8") as fh:
        return Library(**yaml.safe_load(fh))
