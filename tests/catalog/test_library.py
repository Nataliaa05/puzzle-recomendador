import pytest
from pydantic import ValidationError

from puzzle_recsys.catalog.cuisine_imputation import CuisineRules
from puzzle_recsys.catalog.library import Library, load_library
from puzzle_recsys.config import load_paths

MINIMAL = {
    "version": 1,
    "allergens": ["gluten", "lacteos"],
    "ingredients": {
        "pan": {"group": "cereal", "allergens": ["gluten"]},
        "queso": {"group": "lacteo", "allergens": ["lacteos"]},
        "jitomate": {"group": "verdura"},
    },
    "dishes": [
        {
            "slug": "molletes",
            "name": "Molletes",
            "cuisine": "cafeteria",
            "course": "MAIN",
            "protein": "CHEESE",
            "spice": 0,
            "price": 75,
            "ingredients": ["pan", "queso", "jitomate"],
            "aliases": ["Molletes sencillos"],
        }
    ],
}


def _with(**changes):
    data = {**MINIMAL, "dishes": [{**MINIMAL["dishes"][0], **changes}]}
    return data


def test_allergens_are_derived_from_ingredients() -> None:
    lib = Library(**MINIMAL)
    frames = lib.to_frames()
    assert frames["canonical_dishes"].loc[0, "allergens"] == ["gluten", "lacteos"]


def test_unknown_ingredient_is_rejected() -> None:
    with pytest.raises(ValidationError, match="ingrediente"):
        Library(**_with(ingredients=["pan", "oregano_inexistente"]))


def test_unknown_allergen_in_ingredient_is_rejected() -> None:
    data = {
        **MINIMAL,
        "ingredients": {
            **MINIMAL["ingredients"],
            "nuez": {"group": "semilla", "allergens": ["frutos"]},
        },
    }
    with pytest.raises(ValidationError, match="alérgeno"):
        Library(**data)


def test_duplicate_slug_is_rejected() -> None:
    data = {**MINIMAL, "dishes": MINIMAL["dishes"] * 2}
    with pytest.raises(ValidationError, match="duplicad"):
        Library(**data)


def test_spice_and_price_ranges() -> None:
    with pytest.raises(ValidationError):
        Library(**_with(spice=5))
    with pytest.raises(ValidationError):
        Library(**_with(price=0))


# ---------------------------------------------------------------- archivo real del proyecto


@pytest.fixture(scope="module")
def project_library() -> Library:
    return load_library(load_paths().external / "canonical_dishes.yaml")


def test_project_library_is_valid(project_library: Library) -> None:
    assert len(project_library.dishes) >= 150


def test_every_rule_cuisine_has_enough_mains(project_library: Library) -> None:
    rules = CuisineRules.from_yaml(load_paths().external / "cuisine_rules.yaml")
    cuisines = {c for c, _ in rules.rules} | set(rules.scian_defaults.values())
    dishes = project_library.to_frames()["canonical_dishes"]
    mains = dishes[dishes["course"] == "MAIN"].groupby("cuisine").size()
    missing = {c: int(mains.get(c, 0)) for c in cuisines if mains.get(c, 0) < 3}
    assert not missing, f"cocinas con menos de 3 platos fuertes: {missing}"


def test_general_pool_covers_drinks_and_desserts(project_library: Library) -> None:
    dishes = project_library.to_frames()["canonical_dishes"]
    general = set(dishes.loc[dishes["cuisine"] == "general", "course"])
    assert {"DRINK", "DESSERT"} <= general


def test_aliases_are_unique_across_library(project_library: Library) -> None:
    aliases = project_library.to_frames()["canonical_dish_aliases"]
    dup = aliases[aliases.duplicated("alias_normalized", keep=False)]
    assert dup.empty, f"alias ambiguos entre platillos:\n{dup}"


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        ("carbonara", []),
        ("pesto_pasta", ["vegetariano"]),
        ("hummus", ["vegetariano", "vegano"]),
        ("tacos_coliflor", ["vegetariano", "vegano", "sin_gluten"]),
        ("arroz_rojo", ["vegetariano", "vegano", "sin_gluten"]),
        ("pollo_rostizado", ["sin_gluten"]),
    ],
)
def test_diet_tags_are_derived_from_ingredients(
    project_library: Library, slug: str, expected: list[str]
) -> None:
    dish = next(d for d in project_library.dishes if d.slug == slug)
    assert project_library.dish_diet_tags(dish) == expected
