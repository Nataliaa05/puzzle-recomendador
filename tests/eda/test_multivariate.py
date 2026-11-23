import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from puzzle_recsys.eda.multivariate import (
    benzecri_inertia,
    cluster_diners,
    diner_behavior_matrix,
    guest_trait,
    kmeans_scan,
    latent_diner_traits,
    mca_summary,
    recovery_vs_truth,
    redundant_pairs,
    sharing_by_table_trait,
    table_categories,
    trait_recovery,
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


def test_table_categories_bins_numbers_and_groups_rare_cuisines() -> None:
    mesas = pd.DataFrame(
        {
            "franja": ["comida", "cena", "desayuno", "cena"],
            "fin_de_semana": [False, True, False, True],
            "cocina": ["mexicana", "mexicana", "pizza", "sushi"],
            "pax": [1, 2, 4, 7],
            "nivel_precio": [100.0, 200.0, 300.0, 400.0],
            "personal": ["6 a 10", "6 a 10", "11 a 30", "11 a 30"],
        }
    )
    out = table_categories(mesas, top_cuisines=1)
    assert out["cocina"].tolist() == ["mexicana", "mexicana", "otras", "otras"]
    assert out["personas"].astype(str).tolist() == ["1", "2", "3-4", "5+"]
    assert out["nivel_precio"].astype(str).tolist() == ["Q1", "Q2", "Q3", "Q4"]
    assert out["fin_de_semana"].tolist() == ["entre semana", "fin de semana"] * 2


def test_trait_recovery_compares_each_observed_feature_with_its_latent_trait() -> None:
    rng = np.random.default_rng(3)
    latent = pd.DataFrame({"gusto": rng.normal(size=300), "otro": rng.normal(size=300)})
    observed = pd.DataFrame({"pedido": latent["gusto"] + rng.normal(0, 0.3, 300)})
    out = trait_recovery(observed, latent, [("pedido", "gusto"), ("pedido", "otro")])
    assert out.loc[0, "rho"] > 0.8 and out.loc[0, "efecto"] == "fuerte"
    assert abs(out.loc[1, "rho"]) < 0.15


def test_sharing_by_table_trait_reports_share_rate_per_tercile() -> None:
    sessions = pd.DataFrame({"session_id": ["s1", "s2", "s3", "s4"], "pax": [2, 2, 2, 1]})
    guests = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2", "s2", "s3", "s3", "s4"],
            "trait": [0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 0.9],
        }
    )
    order_items = pd.DataFrame({"session_id": ["s3", "s1"], "is_shared": [True, False]})
    out = sharing_by_table_trait(sessions, guests, order_items, q=3)
    assert out["mesas"].tolist() == [1, 1, 1]  # la mesa de 1 persona no cuenta
    assert out["pct_comparten"].tolist() == [0.0, 0.0, 100.0]


def test_latent_traits_follow_duplicate_accounts_to_the_true_person() -> None:
    tastes = pd.DataFrame(
        {
            "diner_id": ["d1", "d2"],
            "cuisine_affinity": [np.array([0.7, 0.3]), np.array([0.1, 0.9])],
            "sharer": [0.2, 0.8],
        }
    )
    dups = pd.DataFrame({"diner_id": ["d2-dup"], "diner_id_true": ["d2"]})
    out = latent_diner_traits(tastes, dups, pd.Index(["d1", "d2-dup"]), ["mexicana", "pizza"])
    assert out.loc["d2-dup", "sharer"] == 0.8
    assert out["cocina_dominante"].tolist() == ["mexicana", "pizza"]


def test_guest_trait_reads_registered_and_anonymous_tastes() -> None:
    guests = pd.DataFrame({"guest_id": ["g1", "g2"], "session_id": ["s1", "s1"]})
    refs = pd.DataFrame({"guest_id": ["g1", "g2"], "taste_ref": ["d1", "a1"]})
    registered = pd.DataFrame({"diner_id": ["d1"], "sharer": [0.3]})
    anonymous = pd.DataFrame({"diner_id": ["a1"], "sharer": [0.6]})
    out = guest_trait(guests, refs, [registered, anonymous], "sharer")
    assert out.set_index("guest_id")["trait"].to_dict() == {"g1": 0.3, "g2": 0.6}
    assert set(out.columns) == {"guest_id", "session_id", "trait"}
