import numpy as np
import pandas as pd
import pytest

from ratings.linear import fit_margins, predict_margin

TEAMS = ["A", "B", "C", "D"]
TRUE_R = pd.Series([6.0, 1.0, -2.0, -5.0], index=TEAMS)
HFA = 2.0


def round_robin(neutral_every=0):
    rows = []
    for h in TEAMS:
        for a in TEAMS:
            if h != a:
                rows.append({"home_team": h, "away_team": a})
    g = pd.DataFrame(rows)
    g["neutral"] = False
    if neutral_every:
        g.loc[::neutral_every, "neutral"] = True
    return g


def test_recovers_exact_ratings_and_hfa():
    g = round_robin(neutral_every=3)
    g["y"] = (TRUE_R[g.home_team].to_numpy() - TRUE_R[g.away_team].to_numpy()
              + HFA * ~g.neutral.to_numpy())
    fit = fit_margins(g, "y", TEAMS)
    assert fit.ratings.sum() == pytest.approx(0, abs=1e-9)
    np.testing.assert_allclose(fit.ratings[TEAMS], TRUE_R[TEAMS], atol=1e-9)
    assert fit.hfa == pytest.approx(HFA)
    np.testing.assert_allclose(predict_margin(fit, g), g.y, atol=1e-9)


def test_neutral_games_get_no_hfa():
    g = round_robin()
    g["neutral"] = True
    g["y"] = TRUE_R[g.home_team].to_numpy() - TRUE_R[g.away_team].to_numpy()
    g.loc[0, "neutral"] = False
    g.loc[0, "y"] += HFA
    fit = fit_margins(g, "y", TEAMS)
    assert fit.hfa == pytest.approx(HFA)


def test_disconnected_graph_raises():
    g = pd.DataFrame({"home_team": ["A", "C"], "away_team": ["B", "D"],
                      "neutral": [False, False], "y": [3.0, -1.0]})
    with pytest.raises(ValueError, match="disconnected"):
        fit_margins(g, "y", TEAMS)
