import sys

from puzzle_recsys.runtime import keep_awake


def test_keep_awake_only_acts_on_windows() -> None:
    assert keep_awake() is (sys.platform == "win32")
