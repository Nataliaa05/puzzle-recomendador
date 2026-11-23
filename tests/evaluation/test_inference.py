import numpy as np
import pandas as pd

from puzzle_recsys.evaluation.power import power_curve
from puzzle_recsys.evaluation.resampling import bootstrap_ci
from puzzle_recsys.evaluation.statistical_tests import paired_bootstrap, paired_permutation


def _clustered(n_clusters: int, per: int, effect: float, seed: int):
    rng = np.random.default_rng(seed)
    cluster = np.repeat(np.arange(n_clusters), per)
    person = rng.normal(0, 1, n_clusters)[cluster]  # cada persona tiene su nivel
    diff = effect + 0.5 * person + rng.normal(0, 1, len(cluster))
    return pd.Series(diff), pd.Series(cluster)


def test_bootstrap_ci_covers_the_mean_and_widens_with_clustering() -> None:
    values, clusters = _clustered(200, 10, 0.0, 0)
    flat = bootstrap_ci(values, n_boot=500, seed=0)
    clus = bootstrap_ci(values, clusters=clusters, n_boot=500, seed=0)
    assert flat["inferior"] < values.mean() < flat["superior"]
    # Las consultas de una misma persona no son independientes: el IC por persona es más ancho.
    assert (clus["superior"] - clus["inferior"]) > (flat["superior"] - flat["inferior"])


def test_paired_tests_detect_a_real_difference_and_not_a_null_one() -> None:
    d_null, c = _clustered(300, 8, 0.0, 1)
    d_real, _ = _clustered(300, 8, 0.3, 2)
    for test in (paired_permutation, paired_bootstrap):
        assert test(d_real, c, n_iter=500, seed=0)["p"] < 0.01
        assert test(d_null, c, n_iter=500, seed=0)["p"] > 0.05


def test_permutation_p_value_is_reproducible() -> None:
    d, c = _clustered(100, 5, 0.1, 3)
    a = paired_permutation(d, c, n_iter=300, seed=7)
    b = paired_permutation(d, c, n_iter=300, seed=7)
    assert a == b


def test_power_grows_with_the_number_of_people() -> None:
    d, c = _clustered(400, 8, 0.0, 4)
    out = power_curve(d, c, effect=0.15, sizes=[50, 400], n_sims=60, n_iter=200, seed=0)
    power = out.set_index("comensales")["potencia"]
    assert power[50] < power[400]
    assert power[400] > 0.8
