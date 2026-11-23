import numpy as np
import pandas as pd

from puzzle_recsys.ranking.baselines import content_score, popularity_score
from puzzle_recsys.ranking.feature_sets import FOR_YOU, model_frame
from puzzle_recsys.ranking.logistic import make_logistic


def test_popularity_uses_the_restaurant_only_when_it_has_enough_history() -> None:
    df = pd.DataFrame(
        {
            "pedidos_previos_restaurante": [100, 100, 3, 3],
            "share_restaurante": [0.5, 0.1, 0.9, 0.1],
            "share_global_canonico": [0.1, 0.9, 0.2, 0.6],
        }
    )
    out = popularity_score(df, min_orders=10)
    assert out.tolist() == [0.5, 0.1, 0.2, 0.6]


def test_content_score_prefers_what_matches_the_profile_and_ignores_missing() -> None:
    df = pd.DataFrame(
        {
            "veces_pidio_canonico": [5, 0, 0],
            "share_proteina": [0.8, 0.1, np.nan],
            "afinidad_cocina_platillo": [0.6, 0.2, np.nan],
            "dif_picante": [0.0, 2.0, np.nan],
            "ajuste_precio": [0.0, 1.5, np.nan],
        }
    )
    s = content_score(df)
    assert s.iloc[0] > s.iloc[2] > s.iloc[1]  # sin datos queda neutral, en medio
    assert s.notna().all()


def _toy(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FOR_YOU.numeric})
    for c in FOR_YOU.categorical:
        df[c] = rng.choice(["a", "b"], n)
    df["canonical_pred"] = rng.choice(["x", "y", "z"], n)
    for c in FOR_YOU.raw_counts:
        df[c] = rng.integers(0, 10, n)
    df["y"] = (df["habito_tipo"] + rng.normal(0, 0.5, n) > 0.8).astype(int)
    return df


def test_model_frame_logs_counts_and_keeps_missing_as_nan() -> None:
    df = _toy(5, 0)
    df.loc[0, "habito_tipo"] = np.nan
    out = model_frame(df)
    assert np.isnan(out.loc[0, "habito_tipo"])
    assert out["log_veces_pidio_canonico"].equals(np.log1p(df["veces_pidio_canonico"]))
    assert set(FOR_YOU.columns) <= set(out.columns)


def test_logistic_pipeline_learns_signal_and_survives_missing_and_new_categories() -> None:
    train, test = model_frame(_toy(3_000, 1)), model_frame(_toy(500, 2))
    test.loc[:20, "habito_tipo"] = np.nan
    test.loc[:5, "course"] = "nunca_visto"
    model = make_logistic(FOR_YOU, c=1.0).fit(train[list(FOR_YOU.columns)], train["y"])
    p = model.predict_proba(test[list(FOR_YOU.columns)])[:, 1]
    assert np.isfinite(p).all()
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(test["y"], p) > 0.8
