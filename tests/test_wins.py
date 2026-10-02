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
