import numpy as np
import pandas as pd

from puzzle_recsys.evaluation.metrics import rank_metrics
from puzzle_recsys.ranking.feature_sets import FeatureSet
from puzzle_recsys.ranking.lgbm_ranker import fit_ranker, predict, to_lgbm

FS = FeatureSet(
    numeric=("x", "ruido"), raw_counts=(), categorical=("tipo",), target_encoded=("canon",)
)


def _toy(n_queries: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_queries):
        k = rng.integers(3, 7)
        x = rng.normal(size=k)
        y = np.zeros(k, dtype=int)
        y[np.argmax(x + rng.normal(0, 0.3, k))] = 1
        rows.append(
            pd.DataFrame(
                {
                    "query_id": q,
                    "x": x,
                    "ruido": rng.normal(size=k),
                    "y": y,
                    "tipo": rng.choice(["a", "b"], k),
                    "canon": rng.choice(["c1", "c2"], k),
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    return df.sample(frac=1, random_state=seed)  # filas desordenadas a propósito


def test_to_lgbm_groups_rows_by_query_and_fixes_categories() -> None:
    df = _toy(50, 0)
    data = to_lgbm(df, FS)
    assert data.group.sum() == len(df)
    assert data.X["query_id"].is_monotonic_increasing if "query_id" in data.X else True
    assert (np.diff(data.query_id) >= 0).all()  # filas contiguas por consulta
    new = to_lgbm(df.assign(tipo="nunca_visto"), FS, categories=data.categories)
    assert new.X["tipo"].isna().all()  # categoría desconocida -> faltante, no un código nuevo


def test_lambdarank_learns_the_signal() -> None:
    train, valid = to_lgbm(_toy(400, 1), FS), _toy(200, 2)
    v = to_lgbm(valid, FS, categories=train.categories)
    params = {"num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 5, "verbose": -1}
    booster = fit_ranker(train, v, params, num_boost_round=200, early_stopping_rounds=20, seed=0)
    scored = valid.assign(score=predict(booster, valid, FS, train.categories))
    random = valid.assign(score=np.random.default_rng(0).random(len(valid)))
    assert (
        rank_metrics(scored, ks=(1,))["ndcg@1"].mean()
        > rank_metrics(random, ks=(1,))["ndcg@1"].mean() + 0.3
    )
    assert booster.best_iteration > 0
