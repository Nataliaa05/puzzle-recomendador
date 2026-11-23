import numpy as np
import pandas as pd
import pytest

from puzzle_recsys.evaluation.metrics import catalog_coverage, rank_metrics, summarize


def _q(scores, labels, qid="q"):
    return pd.DataFrame({"query_id": qid, "score": scores, "y": labels})


def test_perfect_ranking_scores_one_everywhere() -> None:
    out = rank_metrics(_q([0.9, 0.5, 0.1], [1, 0, 0]), ks=(1, 3)).iloc[0]
    assert out["ndcg@1"] == 1.0 and out["hit@3"] == 1.0 and out["map@3"] == 1.0
    assert out["recall@1"] == 1.0


def test_positive_in_second_place() -> None:
    out = rank_metrics(_q([0.9, 0.5, 0.1], [0, 1, 0]), ks=(1, 3)).iloc[0]
    assert out["hit@1"] == 0.0 and out["hit@3"] == 1.0
    assert out["ndcg@3"] == pytest.approx(1 / np.log2(3))
    assert out["map@3"] == pytest.approx(0.5)


def test_two_positives_partial_recall_and_ap() -> None:
    # Orden: pos, neg, pos, neg.
    out = rank_metrics(_q([4, 3, 2, 1], [1, 0, 1, 0]), ks=(2, 3)).iloc[0]
    assert out["recall@2"] == pytest.approx(0.5)
    assert out["map@3"] == pytest.approx((1 / 1 + 2 / 3) / 2)
    idcg = 1 + 1 / np.log2(3)
    assert out["ndcg@3"] == pytest.approx((1 + 1 / np.log2(4)) / idcg)


def test_ties_are_broken_reproducibly_not_in_favor_of_the_positive() -> None:
    """Con empates (p. ej. popularidad 0) el orden no puede depender de dónde quedó el positivo."""
    many = pd.concat([_q([1.0] * 4, [1, 0, 0, 0], qid=i) for i in range(2_000)])
    out = rank_metrics(many, ks=(1,), seed=0)
    assert out["hit@1"].mean() == pytest.approx(0.25, abs=0.03)
    again = rank_metrics(many, ks=(1,), seed=0)
    assert out["hit@1"].equals(again["hit@1"])


def test_summarize_by_segment_reports_mean_and_count() -> None:
    per_q = pd.DataFrame({"query_id": [1, 2, 3], "ndcg@3": [1.0, 0.0, 0.5], "seg": ["a", "a", "b"]})
    out = summarize(per_q, by="seg").set_index("seg")
    assert out.loc["a", "ndcg@3"] == 0.5 and out.loc["a", "consultas"] == 2
    total = summarize(per_q)
    assert total["ndcg@3"].iloc[0] == pytest.approx(0.5)


def test_catalog_coverage_counts_distinct_items_shown_in_top_k() -> None:
    df = pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2],
            "menu_item_id": ["a", "b", "a", "c"],
            "score": [2, 1, 2, 1],
            "y": [1, 0, 0, 1],
        }
    )
    assert catalog_coverage(df, k=1) == pytest.approx(1 / 3)  # solo "a" llega al primer lugar
