import numpy as np
import pandas as pd

from puzzle_recsys.evaluation.oracle import oracle_scores

W = {"affinity_weight": 1.0, "protein_weight": 0.6, "ingredient_weight": 0.5, "spice_penalty": 1.2}


def test_oracle_prefers_what_the_person_truly_likes_plus_the_period_bonus() -> None:
    cands = pd.DataFrame(
        {"query_id": [0, 0], "diner_id": ["d1", "d1"], "menu_item_id": ["m1", "m2"]}
    )
    tastes = pd.DataFrame(
        {
            "cuisine_affinity": [np.array([0.9, 0.1])],
            "protein_affinity": [np.array([1.0])],
            "ingredient_pref": [np.array([0])],
            "spice_tolerance": [3],
        },
        index=["d1"],
    )
    items = pd.DataFrame(
        {
            "cuisine_idx": [0, 1],
            "protein_idx": [-1, -1],
            "ingredients": [[], []],
            "spice": [0, 0],
            "appeal": [0.0, 0.0],
            "quality": [0.0, 0.0],
        },
        index=["m1", "m2"],
    )
    s = oracle_scores(cands, tastes, items, W)
    assert s[0] > s[1]
    flipped = oracle_scores(cands, tastes, items, W, period_bonus=np.array([0.0, 5.0]))
    assert flipped[1] > flipped[0]
