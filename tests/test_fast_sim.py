import numpy as np
import pandas as pd
import pytest

from ratings.fast_sim import FastSim
from ratings.fit import FuturesSet, JointObjective
from ratings.ladder import NormalModel
from ratings.season_sim import League, Schedule
from ratings.wins import WinProbs

TEAMS = [f"{d}{i}" for d in "ABCDEFGH" for i in range(3)]
LEAGUE = League(TEAMS, conf=np.repeat([0] * 4 + [1] * 4, 3), div=np.repeat(np.arange(8), 3))
PROBS = WinProbs(NormalModel(), 13.0)


def schedule(played_frac):
    g = [(h, a) for h in TEAMS for a in TEAMS if h < a]
    df = pd.DataFrame(g, columns=["home_team", "away_team"]).assign(neutral=False)
    n = int(len(df) * played_frac)
    df["result"] = [3.0] * n + [np.nan] * (len(df) - n)
    return Schedule.from_frame(df, LEAGUE)


def test_outputs_are_probabilities_and_crn_deterministic():
    sim = FastSim(LEAGUE, schedule(0.5), PROBS, n_sims=2000, seed=3)
    r = np.linspace(-5, 5, len(TEAMS))
    o1, o2 = sim.run(r, 1.5, 2.0), sim.run(r, 1.5, 2.0)
    np.testing.assert_array_equal(o1.sb, o2.sb)                      # common random numbers
    assert o1.sb.sum() == pytest.approx(1) and o1.conf.sum() == pytest.approx(2)
    assert o1.seed1.sum() == pytest.approx(2) and o1.division.sum() == pytest.approx(8)
    np.testing.assert_allclose(o1.wins_dist.sum(1), 1)


def test_completed_season_has_no_uncertainty_in_standings():
    sim = FastSim(LEAGUE, schedule(1.0), PROBS, n_sims=500)
    o = sim.run(np.zeros(len(TEAMS)), 1.5, 0.0)
    assert set(np.unique(o.division)) <= {0.0, 1.0}
    assert set(np.unique(o.wins_dist)) <= {0.0, 1.0}


def test_stronger_team_wins_more_and_tau_flattens():
    sim = FastSim(LEAGUE, schedule(0.0), PROBS, n_sims=4000)
    r = np.zeros(len(TEAMS))
    r[0] = 8.0
    sharp, flat = sim.run(r, 1.5, 0.0), sim.run(r, 1.5, 6.0)
    assert sharp.sb[0] == sharp.sb.max()
    assert flat.sb[0] < sharp.sb[0]


def test_objective_unpack_and_parts():
    obs = pd.DataFrame({"home_team": ["A0", "B0"], "away_team": ["A1", "B1"], "neutral": False,
                        "y": [3.0, -1.0], "weight": 1.0})
    sim = FastSim(LEAGUE, schedule(0.5), PROBS, n_sims=500)
    sets = [FuturesSet("division", "divA", np.array([0, 1, 2]), np.array([0.5, 0.3, 0.2]))]
    obj = JointObjective(obs, TEAMS, sim, {}, sets, 1.5, lam_w=0.0, lam_f=10.0)
    x = np.append(np.linspace(-2, 2, len(TEAMS) - 1), np.log(3.0))
    r, tau = obj.unpack(x)
    assert r.sum() == pytest.approx(0) and tau == pytest.approx(3.0)
    p = obj.parts(x)
    assert p["loss"] == pytest.approx(p["line"] + 10.0 * p["futures"])
    assert set(p["futures_kl"]) == {"division"}
