"""Scoreboard: how well does each market model reproduce prices it was not fitted on?

Per daily snapshot (everything as known at the snapshot time):

  T1 near-term lines   models fitted WITHOUT the snapshot's Kalshi game lines
                       predict each listed game's Stage A expected margin (RMSE, pts)
  T2 futures           each model's ratings -> FastSim at a common tau -> division,
                       #1 seed, conference, Super Bowl probabilities vs power-de-vigged
                       Kalshi (mean KL per set; no model here is fitted on futures)
  T3 forward closes    each model's fit at the snapshot predicts the CLOSING Kalshi
                       expected margin of games kicking off in the next 10 days
                       (RMSE, pts); listed games also get the "carry today's price
                       forward" baseline

Models are functions (inputs, use_current_lines) -> (ratings, hfa). Add a model by
adding an entry to MODELS; the scoreboard script picks it up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ratings.fast_sim import FastSim
from ratings.fit import FUTURES, futures_sets
from ratings.linear import fit_margins, line_observations
from ratings.schedule import NFLVERSE_TEAMS
from ratings.wins import Season, WinProbs, fit_joint, kl, market_ladders

TEAMS = sorted(NFLVERSE_TEAMS)
LAM_W = 300.0
DEFAULT_HFA = 1.45


@dataclass
class Inputs:
    stem: str
    t_unix: int
    obs: pd.DataFrame          # past closing lines ("close") + current Kalshi lines ("kalshi")
    ladders: dict
    season: Season
    probs: WinProbs
    sigma: float
    sets: list
    blind: pd.DataFrame | None  # daily futures-blind fit (rating, tau, hfa), if saved
    sched_t: pd.DataFrame


def build_inputs(snap, sched_t, model, t_unix) -> Inputs:
    stem = snap.stem
    tidy = pd.read_parquet(snap)
    stage_a = pd.read_parquet(snap.with_name(stem + "_stage_a.parquet"))
    games = pd.read_parquet(snap.with_name(stem + "_games.parquet"))
    pooled = stage_a.loc[stage_a.sigma_source == "pooled", "sigma"]
    sigma = float(pooled.iloc[0]) if len(pooled) else float(stage_a.sigma.median())
    obs = line_observations(stage_a, games, sched_t, model, sigma, pd.Timestamp(stem))
    bpath = snap.with_name(stem + "_blind.parquet")
    return Inputs(stem, t_unix, obs, market_ladders(tidy), Season.build(sched_t, TEAMS),
                  WinProbs(model, sigma), sigma, futures_sets(tidy, TEAMS),
                  pd.read_parquet(bpath).set_index("team") if bpath.exists() else None, sched_t)


# --- models ---------------------------------------------------------------------------

def _lines_fit(o):
    try:
        f = fit_margins(o, "y", TEAMS, weights=o.weight.to_numpy())
        return f.ratings.reindex(TEAMS), f.hfa
    except ValueError:                    # disconnected (too few lines)
        return pd.Series(0.0, index=TEAMS), DEFAULT_HFA


def _obs(inp: Inputs, current: bool, closes: bool = True) -> pd.DataFrame:
    keep = (inp.obs.source == "close") if closes else pd.Series(False, index=inp.obs.index)
    if current:
        keep |= inp.obs.source == "kalshi"
    return inp.obs[keep]


def m_lines(inp, current):
    return _lines_fit(_obs(inp, current))


def m_wins_only(inp, current):
    _, h = _lines_fit(_obs(inp, current))          # lines only supply the HFA
    f = fit_joint(inp.obs.iloc[0:0], inp.season, inp.ladders, inp.probs, LAM_W,
                  start=(np.zeros(len(TEAMS)), h), fixed_hfa=h)
    return f.ratings.reindex(TEAMS), h


def m_lines_wins(inp, current):
    o = _obs(inp, current)
    r0, h = _lines_fit(o)
    f = fit_joint(o, inp.season, inp.ladders, inp.probs, LAM_W,
                  start=(r0.to_numpy(), h), fixed_hfa=h)
    return f.ratings.reindex(TEAMS), h


def m_blind(inp, current):
    if not current or inp.blind is None:
        return None                       # fitted with current lines; T2/T3 only
    return inp.blind.rating.reindex(TEAMS), float(inp.blind.hfa.iloc[0])


MODELS = {
    "lines": m_lines,
    "win totals only": m_wins_only,
    "lines + win totals (current)": m_lines_wins,
    "blind: lines + wins, tau (CMA)": m_blind,
}


# --- tasks ----------------------------------------------------------------------------

def predict(ratings: pd.Series, hfa: float, games: pd.DataFrame) -> np.ndarray:
    return (ratings[games.home_team].to_numpy() - ratings[games.away_team].to_numpy()
            + hfa * ~games.neutral.to_numpy(bool))


def task_near_lines(inp: Inputs, fits_wo: dict) -> list[dict]:
    k = inp.obs[inp.obs.source == "kalshi"]
    rows = []
    for name, fit in fits_wo.items():
        if fit is None or k.empty:
            continue
        e = k.y.to_numpy() - predict(*fit, k)
        rows.append({"task": "T1 near-term lines", "model": name, "n": len(e),
                     "score": float(np.sqrt(np.mean(e**2))), "metric": "RMSE pts"})
    return rows


def task_futures(inp: Inputs, fits: dict, sim: FastSim, tau: float) -> list[dict]:
    rows = []
    for name, fit in fits.items():
        if fit is None:
            continue
        r, h = fit
        o = sim.run(r.to_numpy(), h, tau)
        per_kind = {}
        for s in inp.sets:
            p = np.maximum(getattr(o, FUTURES[s.kind])[s.idx], 0.5 / sim.S)
            per_kind.setdefault(s.kind, []).append(kl(s.q, p / p.sum()))
        allk = [v for vs in per_kind.values() for v in vs]
        rows.append({"task": "T2 futures", "model": name, "n": len(allk),
                     "score": float(np.mean(allk)), "metric": "mean KL"})
        for kind, vs in per_kind.items():
            rows.append({"task": f"T2 futures: {kind}", "model": name, "n": len(vs),
                         "score": float(np.mean(vs)), "metric": "mean KL"})
    return rows


def task_forward(inp: Inputs, fits: dict, targets: pd.DataFrame, horizon_days: int = 10) -> list[dict]:
    """targets: game_id, home_team, away_team, neutral, kickoff_unix, close_mean."""
    t = targets[(targets.kickoff_unix > inp.t_unix)
                & (targets.kickoff_unix <= inp.t_unix + horizon_days * 86400)]
    if t.empty:
        return []
    listed = inp.obs[inp.obs.source == "kalshi"].set_index("game_id").y
    rows = []
    for name, fit in fits.items():
        if fit is None:
            continue
        e = t.close_mean.to_numpy() - predict(*fit, t)
        rows.append({"task": "T3 forward closes", "model": name, "n": len(e),
                     "score": float(np.sqrt(np.mean(e**2))), "metric": "RMSE pts"})
        lt = t[t.game_id.isin(listed.index)]
        if len(lt):
            el = lt.close_mean.to_numpy() - predict(*fit, lt)
            rows.append({"task": "T3 forward closes (listed games)", "model": name, "n": len(el),
                         "score": float(np.sqrt(np.mean(el**2))), "metric": "RMSE pts"})
    lt = t[t.game_id.isin(listed.index)]
    if len(lt):
        eb = lt.close_mean.to_numpy() - listed.reindex(lt.game_id).to_numpy()
        rows.append({"task": "T3 forward closes (listed games)", "model": "baseline: today's price",
                     "n": len(eb), "score": float(np.sqrt(np.mean(eb**2))), "metric": "RMSE pts"})
    return rows
