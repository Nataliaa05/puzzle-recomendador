import numpy as np
import pytest

from puzzle_recsys.simulation.calibration import latin_hypercube, with_params


def test_latin_hypercube_covers_every_stratum_once() -> None:
    bounds = {"a": (0.0, 1.0), "b": (10.0, 20.0)}
    design = latin_hypercube(bounds, n=8, rng=np.random.default_rng(0))
    assert design.shape == (8, 2)
    for name, (lo, hi) in bounds.items():
        values = design[name].to_numpy()
        assert ((values >= lo) & (values <= hi)).all()
        strata = np.floor((values - lo) / (hi - lo) * 8).astype(int).clip(0, 7)
        assert sorted(strata) == list(range(8))  # una muestra por franja


def test_latin_hypercube_is_reproducible() -> None:
    bounds = {"a": (0.0, 1.0)}
    a = latin_hypercube(bounds, n=5, rng=np.random.default_rng(3))
    b = latin_hypercube(bounds, n=5, rng=np.random.default_rng(3))
    assert a.equals(b)


def test_with_params_overrides_nested_keys_without_mutating() -> None:
    cfg = {"choice": {"gumbel_scale": 1.0, "occasion": {"probability": 0.05}}}
    new = with_params(cfg, {"choice.gumbel_scale": 2.0, "choice.occasion.probability": 0.5})
    assert new["choice"]["gumbel_scale"] == 2.0
    assert new["choice"]["occasion"]["probability"] == 0.5
    assert cfg["choice"]["gumbel_scale"] == 1.0  # el original no cambia


def test_with_params_rejects_unknown_keys() -> None:
    with pytest.raises(KeyError):
        with_params({"choice": {}}, {"choice.gumbell_scale": 1.0})
