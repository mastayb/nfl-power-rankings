"""Phase 6: simulate the rest of the season and compare with Kalshi futures.

Run: python scripts/season_sim.py [data/snapshots/<ts>.parquet]

Uses the market ratings (lines + win totals) from <snapshot>_stage_b.parquet.
For each rating-uncertainty level tau (per-team, per-sim N(0, tau) shock),
simulates N_SIMS seasons and compares division, #1 seed, conference, Super
Bowl and seed #2-#7 probabilities with power-de-vigged Kalshi prices.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.devig import power_devig  # noqa: E402
from ratings.ladder import load_key_weights  # noqa: E402
from ratings.schedule import NFLVERSE_TEAMS, load_schedule  # noqa: E402
from ratings.season_sim import League, Schedule, simulate  # noqa: E402
from ratings.wins import WinProbs  # noqa: E402

TEAMS = sorted(NFLVERSE_TEAMS)
SNAP_DIR = ROOT / "data" / "snapshots"
N_SIMS = 10_000
TAUS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # best fit to Kalshi futures, 2026-10-02: 4
SET_KINDS = {"division": "division", "one_seed": "seed_1", "conference": "conf",
             "super_bowl": "sb", "seed": None}

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)


def latest_snapshot() -> Path:
    return sorted(p for p in SNAP_DIR.glob("*.parquet") if "_" not in p.stem)[-1]


def market_sets(tidy: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, team): de-vigged market prob and the model column it maps to."""
    rows = []
    for kind, col in SET_KINDS.items():
        for ev, g in tidy[tidy.kind == kind].groupby("event"):
            g = g[g.team.notna()]
            q = power_devig(g.mid.to_numpy())
            mcol = col or f"seed_{ev[-1]}"   # KXNFLSEED-27AFC2 -> seed_2
            for t, mid, qq, w in zip(g.team, g.mid, q, g.width):
                rows.append({"kind": kind, "event": ev, "team": t, "model_col": mcol,
                             "mid": mid, "q": qq, "width": w})
    return pd.DataFrame(rows)


def compare(sim_probs: pd.DataFrame, mk: pd.DataFrame) -> pd.DataFrame:
    mk = mk.copy()
    mk["p"] = [sim_probs.loc[t, c] for t, c in zip(mk.team, mk.model_col)]
    return mk


def kl_by_set(mk: pd.DataFrame) -> pd.Series:
    def kl(g):
        p = np.clip(g.p.to_numpy(), 1e-4, 1)
        p = p / p.sum()
        q = g.q.to_numpy()
        return float(np.sum(q * np.log(q / p)))
    return mk.groupby(["kind", "event"]).apply(kl, include_groups=False)


def main(snap_path: Path) -> None:
    stem = snap_path.stem
    b = pd.read_parquet(snap_path.with_name(stem + "_stage_b.parquet")).set_index("team")
    tidy = pd.read_parquet(snap_path)
    season = int(pd.read_parquet(snap_path.with_name(stem + "_games.parquet")).game_id.str[:4].iloc[0])
    sigma, hfa = float(b.sigma.iloc[0]), float(b.hfa.iloc[0])
    ratings = b.market.reindex(TEAMS).to_numpy()
    taus = TAUS
    joint_path = snap_path.with_name(stem + "_joint.parquet")
    if joint_path.exists():
        jf = pd.read_parquet(joint_path).set_index("team")
        ratings, taus = jf.joint.reindex(TEAMS).to_numpy(), [float(jf.tau.iloc[0])]
        print(f"using joint fit {joint_path.name}: tau {taus[0]:.2f}")
    league = League.from_nflverse(TEAMS)
    sched = Schedule.from_frame(load_schedule(season), league)
    probs = WinProbs(load_key_weights(), sigma)
    mk = market_sets(tidy)
    print(f"snapshot {snap_path.relative_to(ROOT)}: market ratings, HFA {hfa:.2f}, sigma {sigma:.2f}; "
          f"{(~sched.played).sum()} games left; {mk.event.nunique()} futures sets")

    results, kls = {}, {}
    for tau in taus:
        t0 = time.time()
        res = simulate(league, sched, ratings, hfa, probs, n_sims=N_SIMS, tau=tau, seed=1)
        results[tau] = res.probs
        cmp = compare(res.probs, mk)
        kls[tau] = kl_by_set(cmp)
        print(f"  tau {tau:.1f}: {N_SIMS} sims in {time.time() - t0:.0f}s")

    print("\nKL(market || model) by futures type, mean over sets (lower = closer to market):")
    tab = pd.DataFrame({f"tau={t:g}": k.groupby(level="kind").mean() for t, k in kls.items()})
    tab.loc["ALL (excl. seeds 2-7)"] = pd.DataFrame(
        {f"tau={t:g}": [k.drop("seed", level="kind").mean()] for t, k in kls.items()}).iloc[0]
    print(tab.round(4).to_string())
    best = min(taus, key=lambda t: kls[t].drop("seed", level="kind").mean())
    print(f"-> closest to market: tau = {best:g}")

    cmp = compare(results[best], mk)
    cmp["diff"] = cmp.p - cmp.q
    for kind, title in [("division", "Division winner"), ("one_seed", "#1 seed"),
                        ("conference", "Conference champion"), ("super_bowl", "Super Bowl")]:
        g = cmp[cmp.kind == kind].sort_values(["event", "q"], ascending=[True, False])
        if kind in ("conference", "super_bowl", "one_seed"):
            g = g[(g.q >= 0.02) | (g.p >= 0.02)]
        print(f"\n=== {title} (tau {best:g}): market mid, de-vigged market, model ===")
        print(g[["event", "team", "mid", "q", "p", "diff"]].to_string(
            index=False, formatters={c: "{:.3f}".format for c in ["mid", "q", "p"]}
            | {"diff": "{:+.3f}".format}))

    po = tidy[tidy.kind == "playoff"].set_index("team").mid
    pp = results[best].playoffs.reindex(po.index)
    print(f"\nPlayoffs (not exclusive): market mids sum {po.sum():.2f}; mean |model - market| "
          f"{(pp - po).abs().mean():.3f}; largest gaps:")
    gap = (pp - po).sort_values(key=abs, ascending=False).head(6)
    print(pd.DataFrame({"market": po[gap.index], "model": pp[gap.index], "diff": gap}).round(3).to_string())

    out = results[best].assign(tau=best)
    dest = snap_path.with_name(stem + "_sim.parquet")
    out.reset_index(names="team").to_parquet(dest, index=False)
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_snapshot())
