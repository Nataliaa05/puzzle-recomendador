import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from puzzle_recsys.eda.multivariate import (
    benzecri_inertia,
    cluster_diners,
    diner_behavior_matrix,
    kmeans_scan,
    mca_summary,
    recovery_vs_truth,
    redundant_pairs,
)


@pytest.fixture
def tiny():
    sessions = pd.DataFrame(
        {
            "session_id": ["s1", "s2", "s3", "s4"],
            "restaurant_id": ["r1", "r2", "r1", "r1"],
            "cocina": ["mexicana", "pizza", "mexicana", "mexicana"],
            "ticket_por_persona": [100.0, 300.0, 200.0, 50.0],
        }
    )
    guests = pd.DataFrame(
        {
            "guest_id": ["g1", "g2", "g3", "g4", "g5"],
            "session_id": ["s1", "s1", "s2", "s3", "s4"],
            "diner_id": ["d1", None, "d1", "d1", "d2"],
        }
    )
    menu_items = pd.DataFrame(
        {
            "menu_item_id": ["m1", "m2", "m3"],
            "course": ["MAIN", "DRINK", "MAIN"],
            "spice_level": [2, 0, 1],
        }
    )
    order_items = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1", "s2", "s3", "s4"],
            "guest_id": ["g1", "g2", "g1", "g3", "g4", "g5"],
            "menu_item_id": ["m1", "m2", "m2", "m3", "m1", "m1"],
            "is_shared": [False, False, True, False, False, False],
        }
    )
    return dict(sessions=sessions, guests=guests, menu_items=menu_items, order_items=order_items)


def test_behavior_matrix_describes_each_diner_with_shares_and_rates(tiny) -> None:
    out = diner_behavior_matrix(**tiny, min_visits=1).loc["d1"]
    assert out["cocina__mexicana"] == pytest.approx(2 / 3)
    assert out["cocina__pizza"] == pytest.approx(1 / 3)
    # Líneas propias e individuales por visita: MAIN m1, m3, m1 → 3 en 3 visitas.
    assert out["por_visita__MAIN"] == pytest.approx(1.0)
    assert out["por_visita__DRINK"] == pytest.approx(0.0)  # la bebida compartida no cuenta
    assert out["picante_pedido"] == pytest.approx(5 / 3)
    assert out["log_ticket_pp"] == pytest.approx(np.log(200.0))
    assert out["tasa_compartir"] == pytest.approx(1 / 3)  # solo s1 tuvo algo compartido
    assert out["variedad"] == pytest.approx(2 / 3)
    assert out["log_visitas"] == pytest.approx(np.log(3))


def test_behavior_matrix_drops_diners_with_too_few_visits(tiny) -> None:
    out = diner_behavior_matrix(**tiny, min_visits=2)
    assert list(out.index) == ["d1"]


def test_redundant_pairs_lists_each_strong_pair_once() -> None:
    corr = pd.DataFrame(
        [[1.0, 0.9, 0.1], [0.9, 1.0, -0.8], [0.1, -0.8, 1.0]],
        index=list("abc"),
        columns=list("abc"),
    )
    out = redundant_pairs(corr, threshold=0.7)
    assert out[["a", "b"]].to_numpy().tolist() == [["a", "b"], ["b", "c"]]
    assert out["rho"].tolist() == [0.9, -0.8]


def test_benzecri_keeps_only_dimensions_above_average_and_sums_to_100() -> None:
    # K = 4 variables → umbral 1/K = 0.25; la tercera dimensión queda fuera.
    out = benzecri_inertia(np.array([0.5, 0.35, 0.2]), n_variables=4)
    assert out[2] == 0.0
    expected = np.array([(4 / 3) ** 2 * 0.25**2, (4 / 3) ** 2 * 0.1**2])
    assert out[:2] == pytest.approx(100 * expected / expected.sum())


def test_mca_puts_associated_categories_on_the_same_side() -> None:
    rng = np.random.default_rng(0)
    a = rng.choice(["x", "y"], 500)
    df = pd.DataFrame(
        {"a": a, "b": np.where(a == "x", "p", "q"), "ruido": rng.choice(["u", "v", "w"], 500)}
    )
    out = mca_summary(df, n_components=2, seed=0)
    coords = out["categorias"].set_index(["variable", "categoria"])["dim1"]
    assert np.sign(coords["a", "x"]) == np.sign(coords["b", "p"])
    assert np.sign(coords["a", "x"]) != np.sign(coords["a", "y"])
    inertia = out["inercia"]
    assert inertia["pct_benzecri"].sum() == pytest.approx(100)
    assert inertia["pct_benzecri"].iloc[0] > 90


@pytest.fixture
def blobs():
    rng = np.random.default_rng(1)
    centers = np.array([[0, 0, 0, 0], [6, 6, 0, 0], [0, 6, 6, 6]])
    truth = np.repeat([0, 1, 2], 150)
    x = centers[truth] + rng.normal(0, 0.7, (450, 4))
    return pd.DataFrame(x, columns=list("abcd")), truth


def test_kmeans_scan_silhouette_peaks_at_the_true_number_of_groups(blobs) -> None:
    x, _ = blobs
    out = kmeans_scan(x.to_numpy(), ks=range(2, 6), seed=0)
    assert out.loc[out["silueta"].idxmax(), "k"] == 3
    assert out["inercia"].is_monotonic_decreasing


def test_cluster_diners_recovers_well_separated_groups(blobs) -> None:
    x, truth = blobs
    res = cluster_diners(x, k=3, seed=0, variance=0.9)
    assert adjusted_rand_score(truth, res.labels) > 0.95
    assert res.labels.index.equals(x.index)
    assert 1 <= res.n_components <= 4
    assert res.centers.shape == (3, 4)  # perfil medio en las unidades originales


def test_recovery_vs_truth_separates_related_and_unrelated_traits() -> None:
    rng = np.random.default_rng(2)
    labels = pd.Series(np.repeat([0, 1, 2], 200))
    numeric = pd.DataFrame(
        {"relacionado": labels * 3 + rng.normal(0, 1, 600), "ruido": rng.normal(0, 1, 600)}
    )
    dominant = pd.Series(np.where(labels == 0, "pizza", "tacos"))
    out = recovery_vs_truth(labels, numeric, {"cocina_dominante": dominant}).set_index("rasgo")
    assert out.loc["relacionado", "metrica"] == "epsilon2"
    assert out.loc["relacionado", "valor"] > 0.6
    assert out.loc["ruido", "valor"] < 0.02
    assert out.loc["cocina_dominante", "metrica"] == "AMI"
    assert out.loc["cocina_dominante", "valor"] > 0.5
