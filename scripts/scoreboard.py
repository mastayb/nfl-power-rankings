"""Score every market model on every daily snapshot (see ratings/scoreboard.py).

Run: python -u scripts/scoreboard.py

Closing expected margins for the forward task come from Kalshi hourly candles
at kickoff, run through the same Stage A as the live pipeline, and are cached
in data/scoreboard/closing_margins.parquet.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_clv import discover, fetch_all, tidy_at  # noqa: E402

from ratings.candles import ts  # noqa: E402
from ratings.fast_sim import FastSim  # noqa: E402
from ratings.game_paper import kickoff_utc, stage_a_frame  # noqa: E402
from ratings.kalshi import KalshiClient  # noqa: E402
from ratings.ladder import load_key_weights  # noqa: E402
from ratings.schedule import load_schedule, schedule_as_of  # noqa: E402
from ratings.scoreboard import (MODELS, build_inputs, task_forward, task_futures,  # noqa: E402
                                task_near_lines)
from ratings.season_sim import League, Schedule  # noqa: E402

SEASON = 2026
SNAP_DIR = ROOT / "data" / "snapshots"
OUT = ROOT / "data" / "scoreboard"
N_SIMS = 10_000
DEFAULT_TAU = 4.8

pd.set_option("display.width", 200)


def closing_margins(sched, model) -> pd.DataFrame:
    """Stage A expected home margin from Kalshi quotes at kickoff, for played games."""
    cache = OUT / "closing_margins.parquet"
    have = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["game_id"])
    with KalshiClient() as client:
        gm, joined, _ = discover(client, sched)
        todo = joined[~joined.game_id.isin(have.game_id)]
        if todo.empty:
            return have
        quotes = fetch_all(client, [gm[gm.key.isin(todo.key)]])
    rows = []
    for g in todo.itertuples():
        k = ts(g.kickoff)
        tg = tidy_at(gm[gm.key == g.key], quotes, k)
        if tg.empty or "game_winner" not in set(tg.kind):
            continue
        sa, _ = stage_a_frame(tg, todo[todo.game_id == g.game_id], model)
        if len(sa):
            rows.append({"game_id": g.game_id, "home_team": g.home_team, "away_team": g.away_team,
                         "neutral": bool(g.neutral), "kickoff_unix": k, "close_mean": float(sa["mean"].iloc[0])})
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def main() -> None:
    model = load_key_weights()
    sched = load_schedule(SEASON)
    kick = kickoff_utc(sched)
    print("closing margins...", flush=True)
    targets = closing_margins(sched, model)
    print(f"  {len(targets)} games with a closing price", flush=True)
    league = League.from_nflverse(sorted(sched.home_team.unique()))

    snaps = sorted(p for p in SNAP_DIR.glob("*.parquet") if "_" not in p.stem)
    rows = []
    for snap in snaps:
        t0 = time.time()
        t_unix = int(pd.Timestamp(snap.stem).timestamp())
        sched_t = schedule_as_of(sched, kick, t_unix)
        inp = build_inputs(snap, sched_t, model, t_unix)
        fits = {n: f(inp, True) for n, f in MODELS.items()}
        fits_wo = {n: f(inp, False) for n, f in MODELS.items()}
        tau = float(inp.blind.tau.iloc[0]) if inp.blind is not None else DEFAULT_TAU
        sim = FastSim(league, Schedule.from_frame(sched_t, league), inp.probs, n_sims=N_SIMS, seed=0)
        for r in (task_near_lines(inp, fits_wo) + task_futures(inp, fits, sim, tau)
                  + task_forward(inp, fits, targets)):
            rows.append({"snapshot": snap.stem, **r})
        print(f"  {snap.stem}: {time.time() - t0:.0f}s", flush=True)

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT / "results.parquet", index=False)
    report(res)


def report(res: pd.DataFrame) -> None:
    def pooled(g):
        if g.metric.iloc[0] == "RMSE pts":                   # pool squared errors by n
            return np.sqrt(np.sum(g.score**2 * g.n) / g.n.sum())
        return np.average(g.score, weights=g.n)

    tab = res.groupby(["task", "model"]).apply(pooled, include_groups=False).unstack("model")
    n = res.groupby("task").snapshot.nunique()
    tab.insert(0, "snapshots", n)
    print("\n=== Scoreboard (lower is better; RMSE pooled over snapshots, KL averaged) ===")
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
