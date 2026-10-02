"""Vectorised season simulation with common random numbers, for the joint fit.

Same game model as ratings/season_sim.py, but every random draw (game
uniforms, team shocks, tiebreak draws, playoff uniforms) is fixed at
construction, so outputs are a deterministic function of (ratings, hfa, tau)
and an optimiser can compare candidates without simulation noise.

Simplified tiebreaks (for speed): win pct, then division record (division
ranking) or conference record (seeding, wild cards), then a fixed random
draw. No head-to-head, common games or one-team-per-division wild-card rule;
scripts/joint_fit.py measures the gap against the exact simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ratings.season_sim import League, Schedule
from ratings.wins import N_GAMES, WinProbs


@dataclass
class FastOut:
    division: np.ndarray     # (T,) P(win division)
    seed1: np.ndarray        # (T,) P(#1 seed)
    conf: np.ndarray         # (T,) P(conference champion)
    sb: np.ndarray           # (T,) P(Super Bowl)
    wins_dist: np.ndarray    # (T, 18) P(final wins == n)


class FastSim:
    def __init__(self, league: League, sched: Schedule, probs: WinProbs, n_sims: int = 10_000,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.lg, self.probs, self.S = league, probs, n_sims
        T = len(league.teams)
        act, played = sched.active, sched.played
        rem = act & ~played
        self.h, self.a = sched.home[rem], sched.away[rem]
        self.nn = ~sched.neutral[rem]
        G = rem.sum()

        def onehot(idx, mask=None):
            M = np.zeros((len(idx), T), np.float32)
            M[np.arange(len(idx)), idx] = 1
            if mask is not None:
                M *= mask[:, None]
            return M

        same_div = league.div[self.h] == league.div[self.a]
        same_conf = league.conf[self.h] == league.conf[self.a]
        self.Hm, self.Am = onehot(self.h), onehot(self.a)
        self.Hd, self.Ad = onehot(self.h, same_div), onehot(self.a, same_div)
        self.Hc, self.Ac = onehot(self.h, same_conf), onehot(self.a, same_conf)

        # played games: fixed contributions
        ph, pa, res = sched.home[played], sched.away[played], sched.result[played]
        hs = np.where(res > 0, 1.0, np.where(res < 0, 0.0, 0.5))

        def tally(mask):
            pts, gp = np.zeros(T), np.zeros(T)
            np.add.at(pts, ph[mask], hs[mask])
            np.add.at(pts, pa[mask], 1 - hs[mask])
            np.add.at(gp, ph[mask], 1)
            np.add.at(gp, pa[mask], 1)
            return pts, gp

        allm = np.ones(len(ph), bool)
        self.pts0, gp_played = tally(allm)
        self.dpts0, dgp_played = tally(league.div[ph] == league.div[pa])
        self.cpts0, cgp_played = tally(league.conf[ph] == league.conf[pa])
        self.wins0 = np.zeros(T)
        np.add.at(self.wins0, ph[hs == 1], 1)
        np.add.at(self.wins0, pa[hs == 0], 1)
        self.gp = gp_played + self.Hm.sum(0) + self.Am.sum(0)
        self.dgp = dgp_played + self.Hd.sum(0) + self.Ad.sum(0)
        self.cgp = cgp_played + self.Hc.sum(0) + self.Ac.sum(0)

        # fixed draws
        self.U = rng.random((n_sims, G)).astype(np.float32)
        self.Z = rng.standard_normal((n_sims, T)).astype(np.float32)
        self.R = rng.random((n_sims, T)) * 1e-9
        self.Upo = rng.random((n_sims, 2, 6)).astype(np.float32)   # 6 conf games each
        self.Usb = rng.random(n_sims).astype(np.float32)

        self.div_members = [np.flatnonzero(league.div == d) for d in np.unique(league.div)]
        self.conf_members = [np.flatnonzero(league.conf == c) for c in (0, 1)]

    def _p_home(self, rs, home, away, hfa_on, hfa):
        s = np.arange(self.S)
        mu = rs[s, home] - rs[s, away] + hfa * hfa_on
        ph, pa = self.probs(mu)
        return ph / (ph + pa)

    def run(self, ratings: np.ndarray, hfa: float, tau: float) -> FastOut:
        S, T = self.S, len(ratings)
        rs = ratings[None, :] + tau * self.Z
        mu = rs[:, self.h] - rs[:, self.a] + hfa * self.nn
        ph, pa = self.probs(mu)
        hw = (self.U < ph).astype(np.float32)
        aw = (self.U > 1 - pa).astype(np.float32)
        hs = hw + 0.5 * (1 - hw - aw)                         # home score 1 / .5 / 0
        pts = self.pts0 + hs @ self.Hm + (1 - hs) @ self.Am
        dpts = self.dpts0 + hs @ self.Hd + (1 - hs) @ self.Ad
        cpts = self.cpts0 + hs @ self.Hc + (1 - hs) @ self.Ac
        wins = self.wins0 + hw @ self.Hm + aw @ self.Am
        pct = pts / self.gp
        div_key = pct + 1e-3 * dpts / np.maximum(self.dgp, 1) + self.R
        seed_key = pct + 1e-3 * cpts / np.maximum(self.cgp, 1) + self.R

        div_win = np.zeros((S, T), bool)
        s_idx = np.arange(S)
        for m in self.div_members:
            div_win[s_idx, m[np.argmax(div_key[:, m], axis=1)]] = True

        seeds = np.empty((S, 2, 7), int)
        for c, m in enumerate(self.conf_members):
            k = np.where(div_win[:, m], seed_key[:, m] + 10.0, seed_key[:, m])  # winners first
            order = np.argsort(-k, axis=1)[:, :7]
            seeds[:, c] = m[order]

        champs = np.empty((S, 2), int)
        for c in range(2):
            sd = seeds[:, c]
            u = self.Upo[:, c]

            def game(hi_seed, lo_seed, j):
                hp = self._p_home(rs, sd[s_idx, hi_seed], sd[s_idx, lo_seed], True, hfa)
                return np.where(u[:, j] < hp, hi_seed, lo_seed)

            w = np.column_stack([game(np.full(S, hi), np.full(S, lo), j)
                                 for j, (hi, lo) in enumerate([(1, 6), (2, 5), (3, 4)])])
            w = np.sort(w, axis=1)
            d1 = game(np.zeros(S, int), w[:, 2], 3)
            d2 = game(w[:, 0], w[:, 1], 4)
            final = game(np.minimum(d1, d2), np.maximum(d1, d2), 5)
            champs[:, c] = sd[s_idx, final]
        hp = self._p_home(rs, champs[:, 0], champs[:, 1], False, hfa)
        sb = np.where(self.Usb < hp, champs[:, 0], champs[:, 1])

        wd = np.zeros((T, N_GAMES + 1))
        wi = np.clip(wins.astype(int), 0, N_GAMES)
        for t in range(T):
            wd[t] = np.bincount(wi[:, t], minlength=N_GAMES + 1)[:N_GAMES + 1] / S
        return FastOut(division=div_win.mean(0),
                       seed1=np.bincount(seeds[:, :, 0].ravel(), minlength=T) / S,
                       conf=np.bincount(champs.ravel(), minlength=T) / S,
                       sb=np.bincount(sb, minlength=T) / S,
                       wins_dist=wd)
