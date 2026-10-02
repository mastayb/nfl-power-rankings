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


def test_decay_weight():
    from ratings.linear import decay_weight
    np.testing.assert_allclose(decay_weight([0, 21, 42, -3], 3.0), [1, 0.5, 0.25, 1])
    np.testing.assert_allclose(decay_weight([0, 100], None), [1, 1])


def test_line_observations_sources_and_conversion():
    from ratings.ladder import NormalModel
    from ratings.linear import line_observations

    games = pd.DataFrame({"game_id": ["g_open"], "home_team": ["A"], "away_team": ["B"],
                          "gameday": [pd.Timestamp("2026-10-04").date()]})
    stage_a = pd.DataFrame({"game_id": ["g_open"], "neutral": [False], "mean": [4.2]})
    sched = pd.DataFrame({
        "game_id": ["g_old", "g_open", "g_future"],
        "home_team": ["C", "A", "D"], "away_team": ["D", "B", "C"],
        "gameday": [pd.Timestamp("2026-09-11").date(), pd.Timestamp("2026-10-04").date(),
                    pd.Timestamp("2026-10-11").date()],
        "neutral": [True, False, False],
        "result": [10.0, np.nan, np.nan], "spread_line": [-3.0, 4.0, 1.0],
    })
    obs = line_observations(stage_a, games, sched, NormalModel(), 13.0,
                            asof="2026-10-02T14:49:30Z", half_life_weeks=3.0,
                            results_weight=0.1).set_index("source")
    assert obs.loc["kalshi", "y"] == 4.2 and obs.loc["kalshi", "weight"] == 1.0
    # normal model: median == mean, so the closing line passes straight through
    assert obs.loc["close", "y"] == pytest.approx(-3.0)
    assert obs.loc["close", "weight"] == pytest.approx(0.5)      # 21 days, 3-wk half-life
    assert obs.loc["close", "neutral"]
    assert obs.loc["result", "y"] == 10.0
    assert obs.loc["result", "weight"] == pytest.approx(0.05)
    assert "g_future" not in set(obs.game_id)                    # unplayed, not on Kalshi
