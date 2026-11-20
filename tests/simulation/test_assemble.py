import pandas as pd
import pytest

from puzzle_recsys.simulation.assemble import assert_no_truth_leak, split_truth


def _tables():
    lines = pd.DataFrame(
        {"order_item_id": ["l1"], "guest_id": ["g1"], "utility": [1.0], "experienced": [0.5]}
    )
    guests = pd.DataFrame({"guest_id": ["g1"], "diner_id": ["d1"], "taste_ref": ["d000001"]})
    return lines, guests


def test_split_truth_removes_every_truth_column() -> None:
    lines, guests = _tables()
    c_lines, c_guests, t_lines, t_guests = split_truth(lines, guests)
    assert_no_truth_leak({"order_items": c_lines, "guests": c_guests})
    assert {"utility", "experienced"} <= set(t_lines.columns)
    assert "taste_ref" in t_guests.columns


def test_leak_is_detected() -> None:
    lines, _ = _tables()
    with pytest.raises(ValueError, match="utility"):
        assert_no_truth_leak({"order_items": lines})


def test_canonical_true_cannot_reach_the_contract() -> None:
    items = pd.DataFrame({"menu_item_id": ["m1"], "canonical_true": ["carbonara"]})
    with pytest.raises(ValueError, match="canonical_true"):
        assert_no_truth_leak({"menu_items": items})


def test_tastes_roundtrip_through_ground_truth() -> None:
    import numpy as np

    from puzzle_recsys.config import load_simulation
    from puzzle_recsys.simulation.assemble import tastes_frame, tastes_from_frame
    from puzzle_recsys.simulation.population import sample_tastes

    sim = load_simulation()
    names = {
        "cuisines": ["mexicana", "italiana", "japonesa"],
        "proteins": ["BEEF", "PLANT"],
        "ingredients": ["queso", "cebolla", "camaron"],
        "allergens": list(sim.population["allergy_prevalence"]),
    }
    cfg = {**sim.population, "allergy_prevalence": {a: 0.3 for a in names["allergens"]}}
    tastes = sample_tastes(
        200, cuisine_share=np.array([0.5, 0.3, 0.2]), cfg=cfg, rng=np.random.default_rng(0), **names
    )
    frame = tastes_frame(tastes, [f"d{i:06d}" for i in range(200)])
    back = tastes_from_frame(frame, **names)
    for field in (
        "cuisine_affinity",
        "protein_affinity",
        "ingredient_pref",
        "spice_tolerance",
        "price_sensitivity",
        "distance_sensitivity",
        "variety_seeking",
        "sharer",
        "course_habit",
        "diet",
        "allergies",
    ):
        np.testing.assert_array_equal(getattr(back, field), getattr(tastes, field), err_msg=field)
