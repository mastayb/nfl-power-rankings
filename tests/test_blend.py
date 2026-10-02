import numpy as np
import pandas as pd
import pytest

from ratings.blend import Blend, blend_ratings, completed_week, load_blend


def sched():
    return pd.DataFrame({
        "week": [1, 1, 2, 2, 3],
        "home_team": ["A", "C", "A", "B", "A"], "away_team": ["B", "D", "C", "D", "D"],
        "neutral": [False] * 5, "result": [10.0, -3.0, 7.0, 0.0, np.nan],
    })


def test_completed_week():
    assert completed_week(sched()) == 2
    s = sched().assign(result=np.nan)
    assert completed_week(s) == 0


def test_load_blend_clamps_to_calibrated_range():
    assert load_blend(0).cutoff_week == 2
    assert load_blend(3).cutoff_week == 3
    assert load_blend(40).cutoff_week == 10


def test_blend_ratings_math():
    market = pd.Series({"A": 3.0, "B": -1.0, "C": 0.0, "D": -2.0})
    market = market - market.mean()
    b = Blend(cutoff_week=2, a=0.3, b_market=0.8, b_resume=0.0, half_life=3.0, cap=21.0)
    out, hfa = blend_ratings(market, 1.5, sched(), b)
    np.testing.assert_allclose(out.blend, 0.8 * market, atol=1e-12)
    assert out.blend.sum() == pytest.approx(0)
    b2 = Blend(2, 0.3, 0.8, 0.2, 3.0, 21.0)
    out2, _ = blend_ratings(market, 1.5, sched(), b2)
    np.testing.assert_allclose(out2.blend, 0.8 * out2.market + 0.2 * out2.resume
                               - (0.8 * out2.market + 0.2 * out2.resume).mean(), atol=1e-12)
    assert out2.loc["A", "resume"] > 0          # A beat B and C
