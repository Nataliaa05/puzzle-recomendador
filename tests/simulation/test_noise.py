import numpy as np
import pandas as pd

from puzzle_recsys.simulation.noise import duplicate_accounts, unassign_waiter_orders


def test_waiter_orders_lose_guest_but_truth_keeps_it() -> None:
    lines = pd.DataFrame({"order_item_id": [f"l{i}" for i in range(5000)], "guest_id": "g1"})
    out, truth = unassign_waiter_orders(lines, 0.1, np.random.default_rng(0))
    assert abs(out["guest_id"].isna().mean() - 0.1) < 0.02
    assert (out.loc[out["guest_id"].isna(), "source"] == "WAITER").all()
    assert (truth["guest_id_true"] == "g1").all()


def test_duplicates_split_history_and_keep_mapping() -> None:
    diners = pd.DataFrame({"diner_id": [f"d{i}" for i in range(200)], "created_at": 0})
    guests = pd.DataFrame(
        {"guest_id": range(4000), "diner_id": [f"d{i % 200}" for i in range(4000)]}
    )
    new_diners, new_guests, truth = duplicate_accounts(
        diners, guests, 0.1, np.random.default_rng(1)
    )
    assert len(new_diners) == 200 + len(truth)
    assert 10 <= len(truth) <= 30
    for dup, original in zip(truth["diner_id"], truth["diner_id_true"], strict=True):
        moved = (new_guests["diner_id"] == dup).sum()
        assert 0 < moved < 20
        assert (new_guests["diner_id"] == original).sum() + moved == 20
    assert new_guests["diner_id"].isin(new_diners["diner_id"]).all()
