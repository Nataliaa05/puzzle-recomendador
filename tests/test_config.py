"""Tests de la carga y validación de configuración."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from puzzle_recsys.config import (
    PROJECT_ROOT,
    SimulationScale,
    load_paths,
    load_simulation,
    load_yaml,
)


def test_load_yaml_returns_mapping() -> None:
    data = load_yaml("models")
    assert isinstance(data, dict)
    assert "split" in data


def test_load_yaml_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml("no_existe")


def test_paths_are_absolute_and_inside_project() -> None:
    paths = load_paths()
    for value in paths.model_dump().values():
        path = Path(value)
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents


def test_simulation_has_reproducible_seed_and_valid_scale() -> None:
    sim = load_simulation()
    assert isinstance(sim.seed, int)
    assert sim.scale.n_diners > 0
    assert sim.scale.months > 0


def test_scale_rejects_non_positive_values() -> None:
    with pytest.raises(ValidationError):
        SimulationScale(n_restaurants=0, n_canonical_dishes=400, n_diners=20_000, months=12)


def test_scale_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        SimulationScale(n_restaurants=1, n_canonical_dishes=1, n_diners=1, months=1, typo_field=3)
