import numpy as np
import pandas as pd
import pytest

from ratings.luck import FUMBLE_SWING_PTS, close_games, turnovers

TEAMS = ["A", "B", "C"]


def test_close_games():
    s = pd.DataFrame({"home_team": ["A", "A", "B", "C"], "away_team": ["B", "C", "C", "A"],
                      "result": [3.0, -20.0, 0.0, 8.0]})
    c = close_games(s, TEAMS)
    assert (c.loc["A", ["close_w", "close_l", "close_t"]] == [1, 1, 0]).all()   # 20-pt loss not close
    assert c.loc["B", "close_luck"] == pytest.approx(-0.5 + 0.0)               # L by 3, T
    assert c.close_luck.sum() == pytest.approx(0)


def test_turnovers_luck_sums_to_zero_and_skips_out_of_bounds():
    pbp = pd.DataFrame([
        # A fumbles, A recovers; A fumbles, B recovers; B fumbles out of bounds
        dict(posteam="A", defteam="B", fumble=1, fumbled_1_team="A", fumble_recovery_1_team="A", interception=0),
        dict(posteam="A", defteam="B", fumble=1, fumbled_1_team="A", fumble_recovery_1_team="B", interception=0),
        dict(posteam="B", defteam="A", fumble=1, fumbled_1_team="B", fumble_recovery_1_team=None, interception=0),
        dict(posteam="B", defteam="A", fumble=0, fumbled_1_team=None, fumble_recovery_1_team=None, interception=1),
    ])
    t = turnovers(pbp, TEAMS)
    assert t.loc["A", "fumbles"] == 2 and t.loc["A", "recovered"] == 1
    assert t.loc["B", "recovered"] == 1
    assert t.fum_luck.sum() == pytest.approx(0)
    assert t.loc["A", "fum_pts"] == pytest.approx(0 * FUMBLE_SWING_PTS)
    assert t.loc["A", "to_margin"] == 0     # lost 1 fumble, took 1 INT
    assert t.loc["B", "to_margin"] == 0
    assert t.loc["C", "fumbles"] == 0
