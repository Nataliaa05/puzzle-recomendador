import numpy as np
import pandas as pd
import pytest

from puzzle_recsys.catalog.cuisine_imputation import (
    CuisineRules,
    classify_by_rules,
    impute_cuisine_knn,
    normalize_name,
)

RULES = CuisineRules.from_dict(
    {
        "name_rules": [
            {"cuisine": "japonesa", "pattern": "sushi|japon"},
            {"cuisine": "italiana", "pattern": "trattoria|italian"},
            {"cuisine": "tacos_tortas", "pattern": "taco|taquer"},
        ],
        "scian_defaults": {"722514": "tacos_tortas", "722512": "mariscos"},
    }
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RESTAURANTE ROBERTA TRATTORIA", "restaurante roberta trattoria"),
        ("Café  Niño, S.A.", "cafe nino s a"),
        (None, ""),
    ],
)
def test_normalize_name(raw: str | None, expected: str) -> None:
    assert normalize_name(raw) == expected


def test_name_rule_wins_over_scian_default() -> None:
    out = classify_by_rules(pd.Series(["SUSHI ROLL"]), pd.Series(["722514"]), RULES)
    assert out.loc[0, "cuisine_rule"] == "japonesa"
    assert out.loc[0, "rule_source"] == "NAME"


def test_scian_default_applies_when_name_is_silent() -> None:
    out = classify_by_rules(pd.Series(["EL GÜERO"]), pd.Series(["722512"]), RULES)
    assert out.loc[0, "cuisine_rule"] == "mariscos"
    assert out.loc[0, "rule_source"] == "SCIAN"


def test_unlabeled_when_nothing_matches() -> None:
    out = classify_by_rules(pd.Series(["TENEDOR DEL CIELO"]), pd.Series(["722511"]), RULES)
    assert pd.isna(out.loc[0, "cuisine_rule"])


def test_masked_name_removes_matched_keyword() -> None:
    out = classify_by_rules(pd.Series(["LA TRATTORIA DE ANA"]), pd.Series(["722511"]), RULES)
    assert "trattoria" not in out.loc[0, "masked_name"]
    assert "ana" in out.loc[0, "masked_name"]


def _knn_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "masked_name": ["casa nonna"] * n + ["el pastorcito"] * n + ["casa nonna"],
            "scian_code": ["722511"] * n + ["722514"] * n + ["722511"],
            "lat": np.r_[rng.normal(19.43, 0.001, n), rng.normal(19.30, 0.001, n), 19.43],
            "lng": np.r_[rng.normal(-99.19, 0.001, n), rng.normal(-99.05, 0.001, n), -99.19],
            "cuisine_rule": ["italiana"] * n + ["tacos_tortas"] * n + [None],
        }
    )


def test_knn_imputes_from_similar_labeled_neighbors() -> None:
    out = impute_cuisine_knn(_knn_frame(), n_neighbors=5, min_confidence=0.4, default="mexicana")
    last = out.iloc[-1]
    assert last["cuisine_pred"] == "italiana"
    assert last["cuisine_source"] == "KNN"
    assert last["cuisine_confidence"] >= 0.4


def test_knn_keeps_rule_labels_untouched() -> None:
    out = impute_cuisine_knn(_knn_frame(), n_neighbors=5, min_confidence=0.4, default="mexicana")
    assert (out.iloc[:-1]["cuisine_source"] == "RULE").all()
    assert (out.iloc[:-1]["cuisine_confidence"] == 1.0).all()


def test_non_transferable_labels_are_not_learned_by_knn() -> None:
    df = _knn_frame()
    df.loc[:59, "cuisine_rule"] = "familiar_cadena"  # "marca" sin palabra clave transferible
    out = impute_cuisine_knn(
        df,
        n_neighbors=5,
        min_confidence=0.4,
        default="mexicana",
        exclude_from_training={"familiar_cadena"},
    )
    assert out.iloc[-1]["cuisine_pred"] != "familiar_cadena"


def test_rules_read_transferable_flag() -> None:
    rules = CuisineRules.from_dict(
        {
            "name_rules": [
                {"cuisine": "familiar_cadena", "pattern": "vips", "transferable": False},
                {"cuisine": "japonesa", "pattern": "sushi"},
            ]
        }
    )
    assert rules.non_transferable == frozenset({"familiar_cadena"})


def test_low_confidence_falls_back_to_default() -> None:
    out = impute_cuisine_knn(_knn_frame(), n_neighbors=5, min_confidence=1.01, default="mexicana")
    assert out.iloc[-1]["cuisine_pred"] == "mexicana"
    assert out.iloc[-1]["cuisine_source"] == "DEFAULT"


def test_project_rules_file_has_no_control_characters() -> None:
    """En YAML con comillas dobles, "\b" es un retroceso, no un límite de palabra."""
    from puzzle_recsys.config import load_paths

    rules = CuisineRules.from_yaml(load_paths().external / "cuisine_rules.yaml")
    for cuisine, pattern in rules.rules:
        assert "\b" not in pattern.pattern, f"regla '{cuisine}' tiene un retroceso"
    assert rules.rules[0][1].search("") is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("RESTAURANTE IHOP SANTA FE", "familiar_cadena"),
        ("SHUSHI MIGUEL", "japonesa"),
        ("GUZINA OAXACA", "mexicana_regional"),
        ("RESTAURANTE EL SAZON DE LA ABUELA", "mexicana"),
        ("ROBERTA TRATTORIA", "italiana"),
    ],
)
def test_project_rules_on_real_examples(name: str, expected: str) -> None:
    from puzzle_recsys.config import load_paths

    rules = CuisineRules.from_yaml(load_paths().external / "cuisine_rules.yaml")
    out = classify_by_rules(pd.Series([name]), pd.Series(["722511"]), rules)
    assert out.loc[0, "cuisine_rule"] == expected
