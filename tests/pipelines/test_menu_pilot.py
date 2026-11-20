import numpy as np
import pandas as pd

from puzzle_recsys.pipelines.menu_pilot import select_pilot


def test_select_pilot_keeps_only_valid_sites_and_is_reproducible() -> None:
    restaurants = pd.DataFrame(
        {"restaurant_id": [str(i) for i in range(6)], "name": list("abcdef")}
    )
    www = pd.Series(
        ["www.a.mx", "www.salvaje_mx", None, "b.com", "www.facebook.com/x", "c.com.mx"],
        index=[str(i) for i in range(6)],
    )
    a = select_pilot(restaurants, www, 10, np.random.default_rng(0))
    b = select_pilot(restaurants, www, 10, np.random.default_rng(0))
    assert a["restaurant_id"].tolist() == ["0", "3", "5"]
    assert a["website"].tolist() == ["https://www.a.mx", "https://b.com", "https://c.com.mx"]
    pd.testing.assert_frame_equal(a, b)
