import numpy as np
import pandas as pd
import pytest

from ratings.ladder import NormalModel
from ratings.season_sim import League, Schedule, _Sim, _playoffs, simulate
from ratings.wins import WinProbs

# 2 conferences x 4 divisions x 3 teams: A-D in conf 0, E-H in conf 1
TEAMS = [f"{d}{i}" for d in "ABCDEFGH" for i in range(3)]
LEAGUE = League(TEAMS, conf=np.repeat([0] * 4 + [1] * 4, 3), div=np.repeat(np.arange(8), 3))
IDX = {t: i for i, t in enumerate(TEAMS)}


def sched(games):
    """games: list of (home, away, home margin or None)."""
    df = pd.DataFrame(games, columns=["home_team", "away_team", "result"])
    df["neutral"] = False
    return Schedule.from_frame(df, LEAGUE)


def outcomes(s):
    r = s.result
    return np.where(r > 0, 1.0, np.where(r < 0, 0.0, 0.5))


def test_division_tie_broken_head_to_head():
    # A0 and A1 both 2-1; A0 beat A1 -> A0 wins the division
    g = [("A0", "A1", 3), ("A1", "A2", 7), ("A2", "A0", 3), ("A0", "B0", 7), ("A1", "B1", 7),
         ("A2", "B2", -7)]
    s = sched(g)
    sim = _Sim(LEAGUE, s, outcomes(s), np.random.default_rng(0))
    _, divs = sim.seed_conference(0)
    assert TEAMS[divs[0][0]] == "A0"


def test_division_record_breaks_split_head_to_head():
    # A0, A1 split 1-1 head to head; A0 also beat A2 in division, A1 lost to A2
    g = [("A0", "A1", 3), ("A1", "A0", 3), ("A0", "A2", 3), ("A2", "A1", 3),
         ("A0", "B0", -3), ("A1", "B1", 3)]
    s = sched(g)
    sim = _Sim(LEAGUE, s, outcomes(s), np.random.default_rng(0))
    assert sim.pct[IDX["A0"]] == sim.pct[IDX["A1"]]
    _, divs = sim.seed_conference(0)
    assert TEAMS[divs[0][0]] == "A0"


def test_wildcard_keeps_only_top_team_per_division():
    # A1 and A2 tied with B1 for the wild card: A1 > A2 within the division, so the
    # A-division representative is A1 and A2 cannot jump it.
    sim_rng = np.random.default_rng(0)
    g = [("A0", "A1", 10), ("A0", "A2", 10), ("A1", "A2", 3), ("A2", "B1", 3), ("B1", "A1", 3),
         ("B0", "B1", 3), ("B0", "B2", 3), ("B1", "B2", 3), ("A2", "A1", -3)]
    s = sched(g)
    sim = _Sim(LEAGUE, s, outcomes(s), sim_rng)
    seeds, divs = sim.seed_conference(0)
    wild = [TEAMS[t] for t in seeds[2:]]
    assert wild.index("A1") < wild.index("A2")


def test_playoff_bracket_favourite_wins_and_shapes():
    rng = np.random.default_rng(0)
    n_sims = 2000
    seeds = np.tile(np.array([[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]]), (n_sims, 1, 1))
    r = np.zeros((n_sims, 14))
    r[:, 0] = 60.0          # AFC #1 seed is overwhelming
    champs, sb = _playoffs(seeds, r, 1.5, WinProbs(NormalModel(), 13.0), rng)
    assert (champs[:, 0] == 0).mean() > 0.99
    assert (sb == 0).mean() > 0.95
    assert set(np.unique(champs[:, 1])) <= set(range(7, 14))


def test_simulate_completed_season_is_deterministic_and_probs_sum():
    g = [(h, a, 3.0) for h in TEAMS for a in TEAMS if h < a]
    df = pd.DataFrame(g, columns=["home_team", "away_team", "result"]).assign(neutral=False)
    s = Schedule.from_frame(df, LEAGUE)
    res = simulate(LEAGUE, s, np.zeros(len(TEAMS)), 1.5, WinProbs(NormalModel(), 13.0),
                   n_sims=50, seed=0)
    p = res.probs
    assert set(p.division.unique()) <= {0.0, 1.0}           # no uncertainty left
    assert p.sb.sum() == pytest.approx(1.0)
    assert p.conf.sum() == pytest.approx(2.0)
    assert p.division.sum() == pytest.approx(8.0)           # one winner per division
