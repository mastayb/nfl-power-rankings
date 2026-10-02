"""Daily job: snapshot Kalshi, refit the futures-blind model, update the paper-trade logs.

Run: python -u scripts/daily.py      (cron, 9:30 ET; scripts/game_watch.py runs hourly)

  1. scripts/snapshot.py      new snapshot + nflverse join
  2. scripts/stage_a.py       per-game implied margins
  3. blind fit                lines + win totals, tau fitted, futures excluded
                              (<snapshot>_blind.parquet)
  4. season simulation        10,000 seasons from the blind ratings (<snapshot>_blind_sim.parquet)
  5. futures paper trades     open / mark / settle in data/paper/{trades,marks}.csv
  6. game paper trades        leave-one-out game model; data/paper/game_{trades,marks}.csv; CLV
  7. git commit               data/snapshots + data/paper only (no push)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from joint_fit import blind_fit  # noqa: E402

from ratings.kalshi import KalshiClient  # noqa: E402
from ratings.ladder import load_key_weights  # noqa: E402
from ratings.paper import (JobLock, commit_paths, price_markets, settled_results,  # noqa: E402
                           update)
from ratings.schedule import NFLVERSE_TEAMS, load_schedule  # noqa: E402
from ratings.season_sim import League, Schedule, simulate  # noqa: E402
from ratings.wins import WinProbs  # noqa: E402

TEAMS = sorted(NFLVERSE_TEAMS)
SNAP_DIR = ROOT / "data" / "snapshots"
N_SIMS = 10_000
COMMIT_PATHS = ["data/snapshots", "data/paper"]


def step(msg: str) -> float:
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    return time.time()


def run(script: str) -> None:
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script)], check=True, cwd=ROOT)


def latest_snapshot() -> Path:
    return sorted(p for p in SNAP_DIR.glob("*.parquet") if "_" not in p.stem)[-1]


def run_game_pass(snap: Path) -> None:
    from ratings.game_paper import game_pass

    with KalshiClient() as client:
        g = game_pass(snap, client, load_schedule(int(snap.stem[:4])), load_key_weights())
    print(f"{g.get('games', 0)} games priced (leave-one-out model vs market RMSE "
          f"{g.get('loo_rmse', float('nan')):.2f} pts); opened {g.get('opened', [])}, "
          f"closed {g.get('closed', [])}, CLV recorded {g.get('clv_recorded', [])}")
    if "top" in g and len(g["top"]):
        print(g["top"][["ticker", "mu_model", "mu_market", "model_p", "yes_bid", "yes_ask",
                        "ev"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    with JobLock():
        _main()


def _main() -> None:
    t0 = step("1/7 snapshot")
    run("snapshot.py")
    step("2/7 stage A")
    run("stage_a.py")
    snap = latest_snapshot()
    stem = snap.stem

    step("3/7 futures-blind fit (lines + win totals, tau fitted)")
    blind = blind_fit(snap).set_index("team")
    tau, hfa, sigma = (float(blind[c].iloc[0]) for c in ("tau", "hfa", "sigma"))
    print(f"tau {tau:.2f}, HFA {hfa:.2f}, line RMSE {blind.line_rmse.iloc[0]:.2f}, "
          f"{int(blind.evals.iloc[0])} evals")

    step("4/7 season simulation")
    league = League.from_nflverse(TEAMS)
    sched = Schedule.from_frame(load_schedule(int(stem[:4])), league)
    sim = simulate(league, sched, blind.rating.reindex(TEAMS).to_numpy(), hfa,
                   WinProbs(load_key_weights(), sigma), n_sims=N_SIMS, tau=tau, seed=1).probs
    sim.reset_index(names="team").to_parquet(snap.with_name(stem + "_blind_sim.parquet"), index=False)

    step("5/7 futures paper trades")
    tidy = pd.read_parquet(snap)
    raw = json.load(open(snap.with_suffix(".json")))
    raw_markets = {m["ticker"]: m for blob in raw["series"].values() for m in blob["markets"]}
    priced = price_markets(tidy, raw_markets, sim)
    trades_path = ROOT / "data" / "paper" / "trades.csv"
    open_now = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame(columns=["ticker", "status"])
    missing = sorted(set(open_now[open_now.status == "open"].ticker) - set(priced.ticker))
    with KalshiClient() as client:
        settled = settled_results(client, missing)
    summary = update(priced, settled=settled, now=raw["fetched_at"][:19] + "Z")
    print(f"opened {summary['opened']}, closed {summary['closed']}, open {summary['open']}, "
          f"marked {summary['marked']}")
    best = priced.assign(ev=np.maximum(priced.ev_yes, priced.ev_no),
                         side=np.where(priced.ev_yes >= priced.ev_no, "yes", "no"))
    print("\nTop futures edges today (EV per contract after fees; size = contracts at that price):")
    top = best.sort_values("ev", ascending=False).head(10)
    print(top[["ticker", "side", "model_p", "yes_bid", "yes_ask", "ev", "size_yes", "size_no"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    step("6/7 game markets (leave-one-out model, one trade per game, CLV after kickoff)")
    run_game_pass(snap)

    step("7/7 git")
    print(commit_paths(ROOT, COMMIT_PATHS, f"daily: snapshot {stem} "
                                           f"({summary['open']} open futures paper trades)"))
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
