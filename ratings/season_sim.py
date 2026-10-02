"""Monte Carlo season: remaining games, NFL tiebreakers, seeding, playoff bracket.

Regular season: each remaining game is a home win / away win / tie draw with
the Stage A margin model's probabilities (WinProbs), so ties occur at their
real ~0.3% rate. Optional per-sim team shocks (tau) model rating uncertainty
and drift: one N(0, tau) draw per team per sim, applied to all its games.

Tiebreakers (NFL procedure, simplified where noted):
  division:  head-to-head, division record, common games, conference record,
             strength of victory, strength of schedule, coin flip.
  wild card / seeding (different divisions): keep only the top club of each
             division, head-to-head sweep, conference record, common games
             (min 4), strength of victory, strength of schedule, coin flip.
  When a step separates a 3+ team group without settling it, the remaining
  clubs restart at step 1; once a club is placed, the rest restart too.
  Not modelled: net points / touchdowns steps (essentially never reached).

Playoffs: 7 seeds per conference; 2v7, 3v6, 4v5; #1 hosts the lowest
surviving seed; higher seed hosts; Super Bowl on a neutral field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ratings.wins import WinProbs

CONFS = ["AFC", "NFC"]
EPS = 1e-9


@dataclass
class League:
    teams: list[str]
    conf: np.ndarray    # 0 = AFC, 1 = NFC
    div: np.ndarray     # 0..7, conf * 4 + division

    @classmethod
    def from_nflverse(cls, teams: list[str]) -> "League":
        import nflreadpy as nfl

        t = nfl.load_teams().to_pandas().set_index("team_abbr").loc[teams]
        conf = t.team_conf.map({c: i for i, c in enumerate(CONFS)}).to_numpy()
        divs = sorted(t.team_division.unique())
        return cls(teams, conf, t.team_division.map({d: i for i, d in enumerate(divs)}).to_numpy())


@dataclass
class Schedule:
    home: np.ndarray
    away: np.ndarray
    neutral: np.ndarray
    played: np.ndarray       # has a result
    active: np.ndarray       # played or still to play (False = cancelled)
    result: np.ndarray       # home margin, nan if unplayed

    @classmethod
    def from_frame(cls, sched: pd.DataFrame, league: League, complete: bool = False) -> "Schedule":
        """`complete`: the season is over, so unplayed games were cancelled."""
        idx = {t: i for i, t in enumerate(league.teams)}
        reg = sched[sched.game_type.eq("REG")] if "game_type" in sched else sched
        played = reg.result.notna().to_numpy()
        return cls(reg.home_team.map(idx).to_numpy(), reg.away_team.map(idx).to_numpy(),
                   reg.neutral.to_numpy(bool), played,
                   played if complete else np.ones(len(reg), bool),
                   reg.result.to_numpy(float))


class _Sim:
    """One simulated season's standings + tiebreak helpers."""

    def __init__(self, league: League, sched: Schedule, out: np.ndarray, rng):
        self.lg, self.rng = league, rng
        act = sched.active
        self.h, self.a, self.out = sched.home[act], sched.away[act], out[act]  # out: home score 1/.5/0
        n = len(league.teams)
        self.opp = [[] for _ in range(n)]     # (opponent, score for this team)
        for hh, aa, o in zip(self.h, self.a, self.out):
            self.opp[hh].append((aa, o))
            self.opp[aa].append((hh, 1.0 - o))
        self.pts = np.array([sum(s for _, s in lst) for lst in self.opp])
        self.gp = np.array([len(lst) for lst in self.opp], float)
        self.pct = self.pts / np.maximum(self.gp, 1)
        self.oppsets = [set(o for o, _ in lst) for lst in self.opp]

    def _pct(self, t, keep) -> float | None:
        games = [s for o, s in self.opp[t] if keep(o)]
        return sum(games) / len(games) if games else None

    # --- tiebreak steps: each returns {team: value} or None if not applicable ---
    def h2h(self, g):
        vals = {t: self._pct(t, lambda o, g=g, t=t: o in g and o != t) for t in g}
        return None if any(v is None for v in vals.values()) else vals

    def h2h_sweep(self, g):
        if len(g) == 2:
            return self.h2h(g)
        swept_all, lost_all = [], []
        for t in g:
            res = {o: [] for o in g if o != t}
            for o, s in self.opp[t]:
                if o in res:
                    res[o].append(s)
            if all(res[o] and all(s == 1.0 for s in res[o]) for o in res):
                swept_all.append(t)
            if all(res[o] and all(s == 0.0 for s in res[o]) for o in res):
                lost_all.append(t)
        if swept_all:
            return {t: float(t in swept_all) for t in g}
        if lost_all:
            return {t: float(t not in lost_all) for t in g}
        return None

    def div_rec(self, g):
        vals = {t: self._pct(t, lambda o, t=t: self.lg.div[o] == self.lg.div[t]) for t in g}
        return None if any(v is None for v in vals.values()) else vals

    def conf_rec(self, g):
        vals = {t: self._pct(t, lambda o, t=t: self.lg.conf[o] == self.lg.conf[t]) for t in g}
        return None if any(v is None for v in vals.values()) else vals

    def common(self, g, min_games=0):
        common = set.intersection(*(self.oppsets[t] for t in g)) - set(g)
        vals, n = {}, []
        for t in g:
            games = [s for o, s in self.opp[t] if o in common]
            n.append(len(games))
            vals[t] = sum(games) / len(games) if games else None
        if any(v is None for v in vals.values()) or min(n) < min_games:
            return None
        return vals

    def common4(self, g):
        return self.common(g, min_games=4)

    def sov(self, g):
        out = {}
        for t in g:
            beat = [o for o, s in self.opp[t] if s == 1.0]
            out[t] = self.pts[beat].sum() / self.gp[beat].sum() if beat else 0.0
        return out

    def sos(self, g):
        out = {}
        for t in g:
            opps = [o for o, _ in self.opp[t]]
            out[t] = self.pts[opps].sum() / self.gp[opps].sum() if opps else 0.0
        return out

    # --- ordering ---
    def _pick(self, group, steps, prefilter=None):
        g = list(group)
        while len(g) > 1:
            if prefilter:
                g = prefilter(g)
                if len(g) == 1:
                    break
            for step in steps:
                vals = step(g)
                if vals is None:
                    continue
                vals = {t: (v if v == v else -np.inf) for t, v in vals.items()}  # NaN ranks last
                best = max(vals.values())
                keep = [t for t in g if vals[t] >= best - EPS]
                if len(keep) < len(g):
                    g = keep
                    break          # separated some: restart at step 1
            else:
                return g[self.rng.integers(len(g))]
        return g[0]

    def order(self, teams, steps, prefilter_factory=None) -> list[int]:
        """Full order: by win pct, ties broken by repeatedly picking the best."""
        remaining = sorted(teams, key=lambda t: -self.pct[t])
        out = []
        while remaining:
            top = self.pct[remaining[0]]
            tied = [t for t in remaining if self.pct[t] >= top - EPS]
            pf = prefilter_factory() if prefilter_factory else None
            best = tied[0] if len(tied) == 1 else self._pick(tied, steps, pf)
            out.append(best)
            remaining.remove(best)
        return out

    def seed_conference(self, c: int) -> tuple[list[int], list[list[int]]]:
        lg = self.lg
        div_steps = [self.h2h, self.div_rec, self.common, self.conf_rec, self.sov, self.sos]
        wc_steps = [self.h2h_sweep, self.conf_rec, self.common4, self.sov, self.sos]
        div_orders = {d: self.order(list(np.flatnonzero(lg.div == d)), div_steps)
                      for d in np.unique(lg.div[lg.conf == c])}
        rank_in_div = {t: i for o in div_orders.values() for i, t in enumerate(o)}

        def top_per_div():
            def pf(g):
                best = {}
                for t in g:
                    d = lg.div[t]
                    if d not in best or rank_in_div[t] < rank_in_div[best[d]]:
                        best[d] = t
                return [t for t in g if best[lg.div[t]] == t]
            return pf

        winners = [o[0] for o in div_orders.values()]
        seeds = self.order(winners, wc_steps)
        others = [t for o in div_orders.values() for t in o[1:]]
        wild = self.order(others, wc_steps, top_per_div)[:3]
        return seeds + wild, list(div_orders.values())


@dataclass
class SimResult:
    teams: list[str]
    probs: pd.DataFrame          # per team: playoffs, division, seed_1..7, conf, sb, mean_wins
    n_sims: int


def simulate(league: League, sched: Schedule, ratings: np.ndarray, hfa: float,
             probs: WinProbs, n_sims: int = 10_000, tau: float = 0.0,
             seed: int = 0) -> SimResult:
    rng = np.random.default_rng(seed)
    n = len(league.teams)
    shocks = rng.normal(0.0, tau, (n_sims, n)) if tau > 0 else np.zeros((n_sims, n))
    r = ratings[None, :] + shocks

    rem = sched.active & ~sched.played
    out = np.empty((n_sims, len(sched.home)), np.float32)
    res = sched.result[sched.played]
    out[:, sched.played] = np.where(res > 0, 1.0, np.where(res < 0, 0.0, 0.5))
    if rem.any():
        mu = (r[:, sched.home[rem]] - r[:, sched.away[rem]]
              + hfa * ~sched.neutral[rem])
        p_home, p_away = probs(mu)
        u = rng.random(mu.shape)
        out[:, rem] = np.where(u < p_home, 1.0, np.where(u > 1 - p_away, 0.0, 0.5))
    out[:, ~sched.active] = np.nan

    seeds = np.empty((n_sims, 2, 7), int)
    div_win = np.zeros((n_sims, n), bool)
    wins = np.zeros((n_sims, n))
    for s in range(n_sims):
        sim = _Sim(league, sched, out[s], rng)
        wins[s] = sim.pts
        for c in range(2):
            sd, divs = sim.seed_conference(c)
            seeds[s, c] = sd
            div_win[s, [o[0] for o in divs]] = True

    conf_champ, sb = _playoffs(seeds, r, hfa, probs, rng)
    seed_hits = np.zeros((n_sims, n, 7), bool)
    for c in range(2):
        for k in range(7):
            seed_hits[np.arange(n_sims), seeds[:, c, k], k] = True
    df = pd.DataFrame({
        "mean_wins": wins.mean(0),
        "playoffs": seed_hits.any(2).mean(0),
        "division": div_win.mean(0),
        **{f"seed_{k + 1}": seed_hits[:, :, k].mean(0) for k in range(7)},
        "conf": np.bincount(conf_champ.ravel(), minlength=n) / n_sims,
        "sb": np.bincount(sb, minlength=n) / n_sims,
    }, index=league.teams)
    return SimResult(league.teams, df, n_sims)


def _game(home, away, r, hfa, neutral, probs, rng):
    """Vectorised playoff game (no ties): returns winner team index per sim."""
    s = np.arange(len(home))
    mu = r[s, home] - r[s, away] + (0.0 if neutral else hfa)
    ph, pa = probs(mu)
    p = ph / (ph + pa)
    return np.where(rng.random(len(home)) < p, home, away)


def _playoffs(seeds, r, hfa, probs, rng):
    n_sims = seeds.shape[0]
    s = np.arange(n_sims)
    champs = np.empty((n_sims, 2), int)
    for c in range(2):
        sd = seeds[:, c]                                    # (S, 7) team by seed
        wc = []
        for hi, lo in [(1, 6), (2, 5), (3, 4)]:             # 0-based: 2v7, 3v6, 4v5
            w = _game(sd[:, hi], sd[:, lo], r, hfa, False, probs, rng)
            wc.append(np.where(w == sd[:, hi], hi, lo))     # winner's seed index
        wc = np.sort(np.column_stack(wc), axis=1)           # surviving seeds, ascending
        low = wc[:, 2]                                      # lowest seed plays #1
        w1 = _game(sd[:, 0], sd[s, low], r, hfa, False, probs, rng)
        a, b = wc[:, 0], wc[:, 1]                           # other game: a (higher seed) hosts
        w2 = _game(sd[s, a], sd[s, b], r, hfa, False, probs, rng)
        seed_w1 = np.where(w1 == sd[:, 0], 0, low)
        seed_w2 = np.where(w2 == sd[s, a], a, b)
        home = np.minimum(seed_w1, seed_w2)
        away = np.maximum(seed_w1, seed_w2)
        champs[:, c] = _game(sd[s, home], sd[s, away], r, hfa, False, probs, rng)
    sb = _game(champs[:, 0], champs[:, 1], r, hfa, True, probs, rng)
    return champs, sb
