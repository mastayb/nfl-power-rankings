"""Daily job: snapshot Kalshi, refit the futures-blind model, update the paper-trade log.

Run: python -u scripts/daily.py          (cron: see README notes in IDEAS.md / commit log)

  1. scripts/snapshot.py      new snapshot + nflverse join
  2. scripts/stage_a.py       per-game implied margins
  3. blind fit                lines + win totals, tau fitted, futures excluded
                              (<snapshot>_blind.parquet)
  4. season simulation        10,000 seasons from the blind ratings (<snapshot>_blind_sim.parquet)
  5. paper trades             open / mark / settle in data/paper/{trades,marks}.csv
  6. git commit               data/snapshots + data/paper only (no push)
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
from ratings.paper import price_markets, update  # noqa: E402
from ratings.schedule import NFLVERSE_TEAMS, load_schedule  # noqa: E402
from ratings.season_sim import League, Schedule, simulate  # noqa: E402
from ratings.wins import WinProbs  # noqa: E402

TEAMS = sorted(NFLVERSE_TEAMS)
SNAP_DIR = ROOT / "data" / "snapshots"
N_SIMS = 10_000


def step(msg: str) -> float:
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    return time.time()


def run(script: str) -> None:
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / script)], check=True, cwd=ROOT)


def latest_snapshot() -> Path:
    return sorted(p for p in SNAP_DIR.glob("*.parquet") if "_" not in p.stem)[-1]


def settled_results(tickers: list[str]) -> dict[str, str]:
    """Look up markets that dropped out of the open set; return settled results."""
    out = {}
    if not tickers:
        return out
    with KalshiClient() as client:
        for t in tickers:
            m = client.get(f"/markets/{t}")["market"]
            if m.get("result") in ("yes", "no"):
                out[t] = m["result"]
    return out


COMMIT_PATHS = ["data/snapshots", "data/paper"]


def commit_outputs(stem: str, n_open: int) -> None:
    """Commit today's outputs (parquet + paper log; raw JSON is git-ignored).

    Only COMMIT_PATHS are committed, so unrelated work in progress elsewhere in
    the repo is never swept in. Never pushes. A git failure is reported but
    does not fail the job.
    """
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

    git("add", "--", *COMMIT_PATHS)
    if git("diff", "--cached", "--quiet", "--", *COMMIT_PATHS).returncode == 0:
        print("git: nothing new to commit")
        return
    msg = f"daily: snapshot {stem} ({n_open} open paper trades)"
    r = git("commit", "-m", msg, "--", *COMMIT_PATHS)
    print(f"git: {'committed ' + msg if r.returncode == 0 else 'commit failed: ' + r.stderr.strip()}")


def main() -> None:
    t0 = step("1/5 snapshot")
    run("snapshot.py")
    step("2/5 stage A")
    run("stage_a.py")
    snap = latest_snapshot()
    stem = snap.stem

    step("3/5 futures-blind fit (lines + win totals, tau fitted)")
    blind = blind_fit(snap).set_index("team")
    tau, hfa, sigma = (float(blind[c].iloc[0]) for c in ("tau", "hfa", "sigma"))
    print(f"tau {tau:.2f}, HFA {hfa:.2f}, line RMSE {blind.line_rmse.iloc[0]:.2f}, "
          f"{int(blind.evals.iloc[0])} evals")

    step("4/5 season simulation")
    league = League.from_nflverse(TEAMS)
    sched = Schedule.from_frame(load_schedule(int(stem[:4])), league)
    sim = simulate(league, sched, blind.rating.reindex(TEAMS).to_numpy(), hfa,
                   WinProbs(load_key_weights(), sigma), n_sims=N_SIMS, tau=tau, seed=1).probs
    sim.reset_index(names="team").to_parquet(snap.with_name(stem + "_blind_sim.parquet"), index=False)

    step("5/5 paper trades")
    tidy = pd.read_parquet(snap)
    raw = json.load(open(snap.with_suffix(".json")))
    raw_markets = {m["ticker"]: m for blob in raw["series"].values() for m in blob["markets"]}
    priced = price_markets(tidy, raw_markets, sim)
    trades_path = ROOT / "data" / "paper" / "trades.csv"
    open_now = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame(columns=["ticker", "status"])
    missing = sorted(set(open_now[open_now.status == "open"].ticker) - set(priced.ticker))
    summary = update(priced, settled=settled_results(missing), now=raw["fetched_at"][:19] + "Z")
    print(f"opened {summary['opened']}, closed {summary['closed']}, open {summary['open']}, "
          f"marked {summary['marked']}")

    best = priced.assign(ev=np.maximum(priced.ev_yes, priced.ev_no),
                         side=np.where(priced.ev_yes >= priced.ev_no, "yes", "no"))
    print("\nTop edges today (EV per contract after fees; size = contracts at that price):")
    top = best.sort_values("ev", ascending=False).head(10)
    print(top[["ticker", "side", "model_p", "yes_bid", "yes_ask", "ev", "size_yes", "size_no"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    commit_outputs(stem, summary["open"])
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
