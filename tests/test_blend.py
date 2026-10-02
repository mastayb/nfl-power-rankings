import numpy as np
import pandas as pd
import pytest

from ratings.blend import Blend, blend_ratings, completed_week, load_blend


def sched():
    s = pd.DataFrame({
        "week": [1, 1, 2, 2, 3],
        "home_team": ["A", "C", "A", "B", "A"], "away_team": ["B", "D", "C", "D", "D"],
        "neutral": [False] * 5, "result": [10.0, -3.0, 7.0, 0.0, np.nan],
    })
    s["margin_neutral"] = s.result - np.array([4.0, 0.0, 0.0, 0.0, np.nan])
    return s


def blend(**kw):
    base = dict(cutoff_week=2, curve_week=2, a=0.3, b_market=0.8, b_margin=0.0, b_neutral=0.0,
                half_life=3.0, cap=21.0)
    return Blend(**(base | kw))


MARKET = pd.Series({"A": 3.0, "B": -1.0, "C": 0.0, "D": -2.0}) - 0.0


def test_completed_week():
    assert completed_week(sched()) == 2
    assert completed_week(sched().assign(result=np.nan)) == 0


def test_load_blend_clamps_curve_week_but_keeps_cutoff():
    b = load_blend(0)
    assert b.curve_week == 2 and b.cutoff_week == 0
    assert load_blend(5).curve_week == 5
    b = load_blend(17)
    assert b.curve_week == 12 and b.cutoff_week == 17
    assert b.resume_weight == pytest.approx(b.b_margin + b.b_neutral)


def test_market_only_weights():
    m = MARKET - MARKET.mean()
    out, _ = blend_ratings(m, 1.5, sched(), blend())
    np.testing.assert_allclose(out.blend, 0.8 * m, atol=1e-12)


def test_blend_combines_three_parts_and_centres():
    m = MARKET - MARKET.mean()
    out, hfa = blend_ratings(m, 1.5, sched(), blend(b_margin=0.1, b_neutral=0.2))
    raw = 0.8 * out.market + 0.1 * out.resume + 0.2 * out.resume_neutral
    np.testing.assert_allclose(out.blend, raw - raw.mean(), atol=1e-12)
    assert out.blend.sum() == pytest.approx(0, abs=1e-12)
    # A's 4 lucky points are removed from the luck-neutral résumé
    assert out.loc["A", "resume_neutral"] < out.loc["A", "resume"]
