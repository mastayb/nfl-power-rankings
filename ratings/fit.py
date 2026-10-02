"""Joint fit: game lines + win totals + futures, through the simulated season.

    L(r, tau) = sum_g w_g (y_g - mu_g(r))^2 / (2 line_sd^2)
              + lam_w * sum_teams KL(q_wins || p_wins(r, tau))
              + lam_f * sum_sets  KL(q_set  || p_set(r, tau))

p_wins and p_set come from FastSim (common random numbers, so L is a
deterministic, piecewise-constant function of the parameters) and are
optimised with CMA-ES. Home-field advantage stays at the lines-only estimate.
Futures sets: 8 divisions, 2 x #1 seed, 2 conferences, Super Bowl, each
power-de-vigged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ratings.devig import power_devig
from ratings.fast_sim import FastSim
from ratings.wins import TeamLadder, bucket_probs, kl

FUTURES = {"division": "division", "one_seed": "seed1", "conference": "conf", "super_bowl": "sb"}


@dataclass
class FuturesSet:
    kind: str
    event: str
    idx: np.ndarray      # team indices
    q: np.ndarray        # de-vigged market probabilities


def futures_sets(tidy: pd.DataFrame, teams: list[str]) -> list[FuturesSet]:
    pos = {t: i for i, t in enumerate(teams)}
    out = []
    for kind in FUTURES:
        for ev, g in tidy[(tidy.kind == kind) & tidy.team.notna()].groupby("event"):
            out.append(FuturesSet(kind, ev, g.team.map(pos).to_numpy(), power_devig(g.mid.to_numpy())))
    return out


@dataclass
class JointResult:
    ratings: pd.Series
    tau: float
    hfa: float
    loss: float
    parts: dict          # line, wins, futures terms and per-kind futures KL
    evals: int


class JointObjective:
    def __init__(self, obs: pd.DataFrame, teams: list[str], sim: FastSim,
                 ladders: dict[str, TeamLadder], sets: list[FuturesSet], hfa: float,
                 lam_w: float, lam_f: float, line_sd: float = 2.0):
        pos = {t: i for i, t in enumerate(teams)}
        self.teams, self.sim, self.hfa = teams, sim, hfa
        self.h = obs.home_team.map(pos).to_numpy()
        self.a = obs.away_team.map(pos).to_numpy()
        self.nn = ~obs.neutral.to_numpy(bool)
        self.y, self.w = obs.y.to_numpy(float), obs.weight.to_numpy(float)
        self.lad = [(pos[t], L) for t, L in ladders.items()]
        self.sets, self.lam_w, self.lam_f, self.line_sd = sets, lam_w, lam_f, line_sd
        self.floor = 0.5 / sim.S        # probability floor: half a simulated season

    def unpack(self, x):
        k = len(self.teams)
        r = np.append(x[:k - 1], -x[:k - 1].sum())
        return r, float(np.exp(x[k - 1]))

    def parts(self, x) -> dict:
        r, tau = self.unpack(x)
        resid = self.y - (r[self.h] - r[self.a] + self.hfa * self.nn)
        line = float(np.sum(self.w * resid**2) / (2 * self.line_sd**2))
        o = self.sim.run(r, self.hfa, tau)
        wins = sum(kl(L.q_buckets, np.maximum(bucket_probs(o.wins_dist[i], L.cuts), self.floor))
                   for i, L in self.lad)
        fk = {}
        for s in self.sets:
            p = np.maximum(getattr(o, FUTURES[s.kind])[s.idx], self.floor)
            fk.setdefault(s.kind, []).append(kl(s.q, p / p.sum()))
        fut = sum(sum(v) for v in fk.values())
        return {"line": line, "wins": wins, "futures": fut,
                "loss": line + self.lam_w * wins + self.lam_f * fut,
                "futures_kl": {k: float(np.mean(v)) for k, v in fk.items()},
                "line_rmse": float(np.sqrt(np.average(resid**2, weights=self.w)))}

    def __call__(self, x) -> float:
        return self.parts(x)["loss"]

    def fit(self, r0: np.ndarray, tau0: float = 4.0, sigma0: float = 0.5,
            max_evals: int = 4000, seed: int = 1) -> JointResult:
        import cma

        k = len(self.teams)
        x0 = np.append(r0[:k - 1] - r0.mean(), np.log(tau0))
        es = cma.CMAEvolutionStrategy(x0, sigma0, {"seed": seed, "maxfevals": max_evals,
                                                    "tolfun": 1e-3, "verbose": -9})
        es.optimize(self)
        x = es.result.xbest
        r, tau = self.unpack(x)
        return JointResult(pd.Series(r, index=self.teams), tau, self.hfa, float(es.result.fbest),
                           self.parts(x), int(es.result.evaluations))
