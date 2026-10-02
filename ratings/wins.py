"""Win totals: exact wins distributions from ratings, and the joint lines + wins fit.

Final wins = current wins + Poisson-binomial over remaining games, with each
game's win probability from the Stage A margin model at mean margin
mu_g(r) = r_home - r_away + H. A tie is not a win ("at least n games").

The KXNFLWINS ladder gives market P(W >= n) for every n still undecided.
Rungs quoted at <= 2c or >= 98c are dropped (1c tick distortion), and the
informative rungs cut the wins axis into buckets; the market and model are
compared as distributions over those buckets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ratings.ladder import EDGE, GRID

N_GAMES = 17


class WinProbs:
    """Fast P(home win), P(away win) as a function of expected home margin.

    Tabulated once over the model's location grid, then interpolated, so the
    joint fit can evaluate hundreds of games per step.
    """

    def __init__(self, model, sigma: float):
        locs = np.linspace(-45, 45, 1801)
        if hasattr(model, "pmf"):
            p = model.pmf(locs, sigma)
            self._mean = p @ GRID
            self._home = p[:, GRID > 0].sum(axis=1)
            self._away = p[:, GRID < 0].sum(axis=1)
        else:  # continuous model: no ties
            self._mean = locs
            self._home = np.array([model.G(np.zeros(1), m, sigma)[0] for m in locs])
            self._away = 1 - self._home

    def __call__(self, mean) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(mean, float)
        return (np.interp(mean, self._mean, self._home),
                np.interp(mean, self._mean, self._away))


# --- season state ------------------------------------------------------------------

def current_wins(schedule: pd.DataFrame, teams: list[str]) -> pd.Series:
    played = schedule[schedule.result.notna()]
    home_w = played[played.result > 0].groupby("home_team").size()
    away_w = played[played.result < 0].groupby("away_team").size()
    return home_w.reindex(teams, fill_value=0) + away_w.reindex(teams, fill_value=0)


def remaining_games(schedule: pd.DataFrame) -> pd.DataFrame:
    rem = schedule[schedule.result.isna()]
    return rem[["game_id", "week", "home_team", "away_team", "neutral"]].reset_index(drop=True)


@dataclass
class Season:
    """Precomputed index arrays so wins distributions are fast to evaluate."""
    teams: list[str]
    wins: np.ndarray        # current wins per team
    home_idx: np.ndarray    # per remaining game
    away_idx: np.ndarray
    not_neutral: np.ndarray
    slots: np.ndarray       # (team, slot) -> (game index, is_home) or -1
    slot_home: np.ndarray

    @classmethod
    def build(cls, schedule: pd.DataFrame, teams: list[str]) -> "Season":
        rem = remaining_games(schedule)
        idx = {t: i for i, t in enumerate(teams)}
        h = rem.home_team.map(idx).to_numpy()
        a = rem.away_team.map(idx).to_numpy()
        per_team = [[] for _ in teams]
        for g, (hi, ai) in enumerate(zip(h, a)):
            per_team[hi].append((g, True))
            per_team[ai].append((g, False))
        width = max(len(x) for x in per_team)
        slots = -np.ones((len(teams), width), int)
        slot_home = np.zeros((len(teams), width), bool)
        for t, lst in enumerate(per_team):
            for s, (g, is_home) in enumerate(lst):
                slots[t, s], slot_home[t, s] = g, is_home
        return cls(teams, current_wins(schedule, teams).to_numpy(), h, a,
                   ~rem.neutral.to_numpy(bool), slots, slot_home)

    def game_means(self, ratings: np.ndarray, hfa: float) -> np.ndarray:
        return ratings[self.home_idx] - ratings[self.away_idx] + hfa * self.not_neutral

    def wins_dist(self, ratings: np.ndarray, hfa: float, probs: WinProbs) -> np.ndarray:
        """(team, n) -> P(final wins == n), n = 0..17."""
        p_home, p_away = probs(self.game_means(ratings, hfa))
        p = np.where(self.slot_home, p_home[self.slots], p_away[self.slots])
        p = np.where(self.slots >= 0, p, 0.0)
        dist = np.zeros((len(self.teams), N_GAMES + 1))
        dist[np.arange(len(self.teams)), self.wins] = 1.0
        for s in range(p.shape[1]):
            ps = p[:, s:s + 1]
            dist = dist * (1 - ps) + np.roll(dist, 1, axis=1) * ps
        return dist


def poisson_binomial(ps) -> np.ndarray:
    """P(sum of independent Bernoulli(ps) == k), k = 0..len(ps)."""
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return dist


# --- market ladders ------------------------------------------------------------------

@dataclass
class TeamLadder:
    team: str
    cuts: np.ndarray         # informative thresholds n (P(W >= n) quoted in (2c, 98c))
    q_buckets: np.ndarray    # market prob of each bucket between cuts
    mids: pd.Series          # all open rungs, threshold -> monotonised mid
    width: float             # median bid/ask width of the informative rungs


def market_ladders(tidy: pd.DataFrame) -> dict[str, TeamLadder]:
    """Per team: monotonised P(W >= n) and bucket probabilities from KXNFLWINS."""
    out = {}
    for team, g in tidy[tidy.kind == "win_total"].groupby("team"):
        g = g.sort_values("threshold")
        mids = pd.Series(np.minimum.accumulate(g.mid.to_numpy()), index=g.threshold.astype(int))
        keep = (mids > EDGE) & (mids < 1 - EDGE)
        cuts = mids.index[keep].to_numpy()
        s = mids[keep].to_numpy()
        q = -np.diff(np.concatenate([[1.0], s, [0.0]]))
        out[team] = TeamLadder(team, cuts, q, mids, float(g.width[keep.to_numpy()].median()))
    return out


def bucket_probs(dist: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Model probs for buckets [0, c1), [c1, c2), ..., [ck, 17] of one team's dist."""
    sf = np.concatenate([[1.0], [dist[c:].sum() for c in cuts], [0.0]])
    return -np.diff(sf)


def expected_wins_market(ladder: TeamLadder, wins_now: int) -> float:
    """E[W] = sum_n P(W >= n): 1 for rungs already reached, market mid otherwise."""
    return wins_now + float(ladder.mids.sum())


# --- joint fit -------------------------------------------------------------------------

def kl(q: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1)
    m = q > 0
    return float(np.sum(q[m] * np.log(q[m] / p[m])))


@dataclass
class JointFit:
    ratings: pd.Series
    hfa: float
    line_rmse: float
    wins_kl: float           # mean KL(q || p) per team
    lam: float
    success: bool


def fit_joint(obs: pd.DataFrame, season: Season, ladders: dict[str, TeamLadder],
              probs: WinProbs, lam: float, tau: float = 2.0,
              start: tuple[np.ndarray, float] | None = None) -> JointFit:
    """Minimise  sum_g w_g (y_g - mu_g)^2 / (2 tau^2) + lam * sum_t KL(q_t || p_t(r)).

    The line term is a Gaussian log-likelihood with closing-line noise tau, so
    `lam` reads as the effective number of observed seasons each team's
    win-total market is worth.
    """
    teams = season.teams
    k = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    h = obs.home_team.map(idx).to_numpy()
    a = obs.away_team.map(idx).to_numpy()
    nn = ~obs.neutral.to_numpy(bool)
    y = obs.y.to_numpy(float)
    w = obs.weight.to_numpy(float)
    lad = [(idx[t], L) for t, L in ladders.items()]

    def unpack(x):
        r = np.append(x[:k - 1], -x[:k - 1].sum())
        return r, x[k - 1]

    def loss(x):
        r, hfa = unpack(x)
        resid = y - (r[h] - r[a] + hfa * nn)
        f = np.sum(w * resid**2) / (2 * tau**2)
        if lam:
            dist = season.wins_dist(r, hfa, probs)
            f += lam * sum(kl(L.q_buckets, bucket_probs(dist[i], L.cuts)) for i, L in lad)
        return f

    if start is None:
        x0 = np.zeros(k)
        x0[-1] = 1.5
    else:
        x0 = np.append(start[0][:k - 1], start[1])
    res = minimize(loss, x0, method="L-BFGS-B")
    r, hfa = unpack(res.x)
    resid = y - (r[h] - r[a] + hfa * nn)
    dist = season.wins_dist(r, hfa, probs)
    mean_kl = np.mean([kl(L.q_buckets, bucket_probs(dist[i], L.cuts)) for i, L in lad])
    return JointFit(pd.Series(r, index=teams), float(hfa),
                    float(np.sqrt(np.average(resid**2, weights=w))), float(mean_kl), lam,
                    bool(res.success))
