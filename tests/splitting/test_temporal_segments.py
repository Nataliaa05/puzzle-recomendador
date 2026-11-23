import numpy as np
import pandas as pd

from puzzle_recsys.splitting.segments import drop_groups_without_positive, history_band
from puzzle_recsys.splitting.temporal import assign_split, month_index, rolling_origin_folds

SPLIT = {"train": [1, 9], "valid": [10, 10], "test": [11, 12]}


def test_month_index_counts_months_from_the_start_in_local_time() -> None:
    times = pd.Series(
        pd.to_datetime(["2025-01-06 12:00", "2025-02-06 05:30", "2025-02-06 06:30"], utc=True)
    )
    # 05:30 UTC del 6 de febrero = 23:30 del 5 de febrero en CDMX: aún es el mes 1.
    out = month_index(times, start="2025-01-06", tz="America/Mexico_City")
    assert out.tolist() == [1, 1, 2]


def test_assign_split_follows_the_configured_month_ranges() -> None:
    out = assign_split(pd.Series([1, 9, 10, 11, 12]), SPLIT)
    assert out.tolist() == ["train", "train", "valid", "test", "test"]


def test_rolling_origin_folds_never_validate_on_the_past() -> None:
    months = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
    folds = rolling_origin_folds(months, train_end=9, n_folds=3)
    assert [int(months[v].iloc[0]) for _, v in folds] == [7, 8, 9]
    for fit, val in folds:
        assert months[fit].max() < months[val].min()


def test_history_band_labels_previous_visits() -> None:
    out = history_band(pd.Series([0, 1, 5, 6, 20, 21, 300]))
    assert out.astype(str).tolist() == ["0", "1-5", "1-5", "6-20", "6-20", "21+", "21+"]


def test_drop_groups_without_positive_keeps_only_groups_with_a_label() -> None:
    df = pd.DataFrame({"q": ["a", "a", "b", "b", "c"], "y": [0, 1, 0, 0, 1]})
    out = drop_groups_without_positive(df, group="q", label="y")
    assert out["q"].tolist() == ["a", "a", "c"]
    assert np.array_equal(out.index, [0, 1, 4])
