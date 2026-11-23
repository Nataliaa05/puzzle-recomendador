import numpy as np
import pandas as pd
import pytest

from puzzle_recsys.features.encoding import OneHotTop, TargetEncoder, target_encode_oof
from puzzle_recsys.features.missing import MedianImputer, imputation_error, missing_report


def test_missing_report_lists_columns_with_missing_values() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [1, 2, 3, 4]})
    out = missing_report(df).set_index("variable")
    assert out.loc["a", "pct_faltante"] == 50.0
    assert "b" not in out.index


def test_median_imputer_learns_on_train_and_adds_indicators() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]})
    other = pd.DataFrame({"x": [np.nan, 10.0]})
    imp = MedianImputer(["x"]).fit(train)
    out = imp.transform(other)
    assert out["x"].tolist() == [2.0, 10.0]  # la mediana del ENTRENAMIENTO, no la de "other"
    assert out["x__faltante"].tolist() == [1, 0]


def test_imputation_error_knn_beats_median_when_predictors_carry_signal() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=2_000)
    df = pd.DataFrame(
        {"objetivo": z + rng.normal(0, 0.3, 2_000), "pista": z, "ruido": rng.normal(size=2_000)}
    )
    out = imputation_error(
        df,
        target="objetivo",
        predictors=["pista", "ruido"],
        k=10,
        mask_share=0.2,
        rng=np.random.default_rng(1),
    ).set_index("metodo")
    assert out.loc["KNN", "MAE"] < 0.6 * out.loc["mediana", "MAE"]


def test_one_hot_top_fixes_columns_on_train_and_groups_rare_values() -> None:
    train = pd.DataFrame({"c": ["a", "a", "a", "b", "b", "z"]})
    enc = OneHotTop(["c"], top=2).fit(train)
    out = enc.transform(pd.DataFrame({"c": ["b", "nuevo"]}))
    assert list(out.columns) == ["c=a", "c=b", "c=otras"]
    assert out.to_numpy().tolist() == [[0, 1, 0], [0, 0, 1]]
    same = enc.transform(pd.DataFrame({"c": ["a"]}))
    assert list(same.columns) == list(out.columns)  # "otras" existe aunque venga en cero


def test_target_encoding_out_of_fold_never_uses_the_row_own_label() -> None:
    df = pd.DataFrame({"cat": ["x"] * 4, "y": [1, 1, 0, 0], "fold": [0, 1, 0, 1]})
    out = target_encode_oof(df, col="cat", target="y", folds=df["fold"], smoothing=0.0)
    # Fila del pliegue 0 se codifica con el pliegue 1 (y = 1, 0): 0.5. Nunca con su propia y.
    assert out.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])
    enc = TargetEncoder("cat", smoothing=2.0).fit(df, target="y")
    new = enc.transform(pd.DataFrame({"cat": ["x", "nunca_visto"]}))
    assert new.tolist() == pytest.approx([0.5, 0.5])  # desconocido -> media global
