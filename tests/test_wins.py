import numpy as np
import pandas as pd
import pytest

from ratings.ladder import NormalModel
from ratings.wins import (Season, TeamLadder, WinProbs, bucket_probs, expected_wins_market,
                          poisson_binomial)


def test_poisson_binomial_matches_binomial():
    d = poisson_binomial([0.5] * 4)
    np.testing.assert_allclose(d, np.array([1, 4, 6, 4, 1]) / 16)


def test_wins_dist_adds_current_wins_and_sums_to_one():
    sched = pd.DataFrame({
        "game_id": ["g1", "g2", "g3"], "week": [1, 2, 3],
        "home_team": ["A", "A", "B"], "away_team": ["B", "B", "A"],
        "neutral": [False, False, True], "result": [7.0, np.nan, np.nan],
    })
    season = Season.build(sched, ["A", "B"])
    assert list(season.wins) == [1, 0]
    probs = WinProbs(NormalModel(), 13.0)
    dist = season.wins_dist(np.array([0.0, 0.0]), 0.0, probs)
    np.testing.assert_allclose(dist.sum(axis=1), 1)
    np.testing.assert_allclose(dist[0, 1:4], [0.25, 0.5, 0.25])   # 1 win + Bin(2, .5)
    np.testing.assert_allclose(dist[1, 0:3], [0.25, 0.5, 0.25])


def test_bucket_probs_and_market_expected_wins():
    dist = np.zeros(18)
    dist[[8, 9, 10]] = [0.25, 0.5, 0.25]
    np.testing.assert_allclose(bucket_probs(dist, np.array([9, 10])), [0.25, 0.5, 0.25])
    lad = TeamLadder("A", np.array([9]), np.array([0.4, 0.6]),
                     pd.Series([0.9, 0.6, 0.1], index=[8, 9, 10]), 0.01)
    assert expected_wins_market(lad, 7) == pytest.approx(8.6)


def test_fit_joint_fixed_hfa_lines_only_matches_wls():
    from ratings.linear import fit_margins
    from ratings.wins import fit_joint

    teams = ["A", "B", "C"]
    obs = pd.DataFrame({"home_team": ["A", "B", "C", "A"], "away_team": ["B", "C", "A", "C"],
                        "neutral": [False] * 4, "y": [5.0, 1.0, -2.0, 6.0], "weight": 1.0})
    sched = pd.DataFrame({"game_id": ["g"], "week": [1], "home_team": ["A"], "away_team": ["B"],
                          "neutral": [False], "result": [np.nan]})
    season = Season.build(sched, teams)
    probs = WinProbs(NormalModel(), 13.0)
    wls = fit_margins(obs, "y", teams)
    j = fit_joint(obs, season, {}, probs, lam=0.0, fixed_hfa=wls.hfa)
    assert j.hfa == wls.hfa
    np.testing.assert_allclose(j.ratings[teams], wls.ratings[teams], atol=1e-3)


def test_winprobs_resolves_tiny_steps_for_gradient_fits():
    from ratings.ladder import KeyNumberModel, MAX_M
    probs = WinProbs(KeyNumberModel(np.ones(MAX_M + 2)), 12.5)
    x = np.array([3.0])
    assert probs(x + 1e-8)[0][0] - probs(x)[0][0] > 0      # float32 would give exactly 0


def test_fit_joint_win_totals_alone_move_ratings():
    from ratings.wins import TeamLadder, fit_joint

    teams = ["A", "B"]
    sched = pd.DataFrame({"game_id": [f"g{i}" for i in range(4)], "week": [1, 2, 3, 4],
                          "home_team": ["A", "B", "A", "B"], "away_team": ["B", "A", "B", "A"],
                          "neutral": [False] * 4, "result": [np.nan] * 4})
    season = Season.build(sched, teams)
    probs = WinProbs(NormalModel(), 13.0)
    # market: A very likely to win at least 3 of 4
    ladders = {"A": TeamLadder("A", np.array([3]), np.array([0.2, 0.8]), pd.Series([0.8], index=[3]), 0.01)}
    empty = pd.DataFrame(columns=["home_team", "away_team", "neutral", "y", "weight"])
    f = fit_joint(empty, season, ladders, probs, lam=100.0, start=(np.zeros(2), 0.0), fixed_hfa=0.0)
    assert f.ratings["A"] > 2.0          # moved well away from the flat start
