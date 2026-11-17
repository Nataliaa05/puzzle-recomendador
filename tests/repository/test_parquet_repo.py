from pathlib import Path

import pandas as pd
import pytest

from puzzle_recsys.repository.parquet_repo import ParquetRepository, run_id_for, start_run


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    repo = ParquetRepository(tmp_path)
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    repo.write_table("menu_items", df)
    pd.testing.assert_frame_equal(repo.read_table("menu_items"), df)


def test_read_missing_table_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="menu_items"):
        ParquetRepository(tmp_path).read_table("menu_items")


def test_ground_truth_lives_in_its_own_folder(tmp_path: Path) -> None:
    repo = ParquetRepository(tmp_path)
    repo.ground_truth.write_table("diner_tastes", pd.DataFrame({"x": [1]}))
    assert (tmp_path / "ground_truth" / "diner_tastes.parquet").exists()
    assert not (tmp_path / "diner_tastes.parquet").exists()


def test_start_run_snapshots_config(tmp_path: Path) -> None:
    config_file = tmp_path / "simulation.yaml"
    config_file.write_text("seed: 1\n", encoding="utf-8")
    repo = start_run(tmp_path / "synthetic", "base-1", config_files=[config_file])
    assert (repo.root / "config" / "simulation.yaml").read_text(encoding="utf-8") == "seed: 1\n"


def test_run_id_is_deterministic() -> None:
    assert run_id_for("base", 20261116) == "base-20261116"
