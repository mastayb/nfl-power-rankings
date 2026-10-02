"""Phase 7: joint fit of ratings + tau to lines, win totals and futures.

Run: python scripts/joint_fit.py [data/snapshots/<ts>.parquet]

1. Choose lam_f (futures weight) by validation: fit on past closing lines +
   win totals + futures, hold out the current Kalshi game lines, score how
   well the ratings predict them. Smallest lam_f within 1% of the best wins.
2. Refit on everything; write <snapshot>_joint.parquet (ratings, tau, HFA)
   and config/joint_fit.yaml. Stage B and the season sim pick these up.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.fast_sim import FastSim  # noqa: E402
from ratings.fit import JointObjective, futures_sets  # noqa: E402
from ratings.ladder import load_key_weights  # noqa: E402
from ratings.linear import fit_margins, line_observations  # noqa: E402
from ratings.schedule import NFLVERSE_TEAMS, load_schedule  # noqa: E402
from ratings.season_sim import League, Schedule  # noqa: E402
from ratings.wins import WinProbs, market_ladders  # noqa: E402

TEAMS = sorted(NFLVERSE_TEAMS)
SNAP_DIR = ROOT / "data" / "snapshots"
LAM_F = [0.0, 30.0, 100.0, 300.0, 1000.0]
TOLERANCE = 0.01
N_SIMS = 10_000
CONFIG = ROOT / "config" / "joint_fit.yaml"


def latest_snapshot() -> Path:
    return sorted(p for p in SNAP_DIR.glob("*.parquet") if "_" not in p.stem)[-1]


def fmt_parts(p: dict) -> str:
    fk = p["futures_kl"]
    return (f"line RMSE {p['line_rmse']:.2f}  wins KL {p['wins'] / 32:.4f}  futures KL: "
            + " ".join(f"{k} {v:.4f}" for k, v in fk.items()))


def main(snap_path: Path) -> None:
    stem = snap_path.stem
    tidy = pd.read_parquet(snap_path)
    stage_a = pd.read_parquet(snap_path.with_name(stem + "_stage_a.parquet"))
    games = pd.read_parquet(snap_path.with_name(stem + "_games.parquet"))
    season = int(games.game_id.str[:4].iloc[0])
    sched_df = load_schedule(season)
    model = load_key_weights()
    sigma = float(stage_a.loc[stage_a.sigma_source == "pooled", "sigma"].iloc[0])
    obs = line_observations(stage_a, games, sched_df, model, sigma, pd.Timestamp(stem))
    lam_w = yaml.safe_load(open(ROOT / "config" / "win_totals.yaml"))["lambda"]

    league = League.from_nflverse(TEAMS)
    sim = FastSim(league, Schedule.from_frame(sched_df, league), WinProbs(model, sigma),
                  n_sims=N_SIMS, seed=0)
    ladders = market_ladders(tidy)
    sets = futures_sets(tidy, TEAMS)
    print(f"snapshot {snap_path.relative_to(ROOT)}: {len(obs)} lines, {len(ladders)} win "
          f"ladders, {len(sets)} futures sets; lam_w {lam_w:g}; {N_SIMS} sims per evaluation")

    # 1. validation
    train, test = obs[obs.source != "kalshi"], obs[obs.source == "kalshi"]
    lines_tr = fit_margins(train, "y", TEAMS, weights=train.weight.to_numpy())
    print(f"\n1. Choose lam_f: fit without the {len(test)} current Kalshi lines, predict them "
          f"(HFA {lines_tr.hfa:.2f} from training lines)")
    val, fits = {}, {}
    for lam_f in LAM_F:
        t0 = time.time()
        obj = JointObjective(train, TEAMS, sim, ladders, sets, lines_tr.hfa, lam_w, lam_f)
        res = obj.fit(lines_tr.ratings.reindex(TEAMS).to_numpy())
        r = res.ratings
        pred = (r[test.home_team].to_numpy() - r[test.away_team].to_numpy()
                + res.hfa * ~test.neutral.to_numpy(bool))
        val[lam_f] = float(np.sqrt(np.mean((test.y.to_numpy() - pred) ** 2)))
        fits[lam_f] = res
        print(f"  lam_f {lam_f:6g}: held-out RMSE {val[lam_f]:.3f}  tau {res.tau:.2f}  "
              f"[{fmt_parts(res.parts)}]  {res.evals} evals, {time.time() - t0:.0f}s")
    floor = min(val.values())
    lam_f = min(l for l, e in val.items() if e <= floor * (1 + TOLERANCE))
    print(f"-> chosen lam_f = {lam_f:g} (smallest within {TOLERANCE:.0%} of best {floor:.3f})")

    # 2. final fit on everything
    lines = fit_margins(obs, "y", TEAMS, weights=obs.weight.to_numpy())
    t0 = time.time()
    obj = JointObjective(obs, TEAMS, sim, ladders, sets, lines.hfa, lam_w, lam_f)
    res = obj.fit(lines.ratings.reindex(TEAMS).to_numpy(), tau0=fits[lam_f].tau)
    print(f"\n2. Final joint fit (lam_w {lam_w:g}, lam_f {lam_f:g}): tau {res.tau:.2f}, HFA "
          f"{res.hfa:.2f}, {res.evals} evals, {time.time() - t0:.0f}s\n   {fmt_parts(res.parts)}")

    sb = pd.read_parquet(snap_path.with_name(stem + "_stage_b.parquet")).set_index("team")
    prev = sb.market_lines_wins if "market_lines_wins" in sb else sb.market
    tab = pd.DataFrame({"joint": res.ratings, "lines_wins": prev.reindex(TEAMS),
                        "lines_only": lines.ratings.reindex(TEAMS)})
    tab["d_vs_lines_wins"] = tab.joint - tab.lines_wins
    tab["rank"] = tab.joint.rank(ascending=False).astype(int)
    tab["rank_lines_wins"] = tab.lines_wins.rank(ascending=False).astype(int)
    print(tab.sort_values("joint", ascending=False).to_string(float_format=lambda x: f"{x:+.2f}"))
    print(f"max |change vs lines + win totals| {tab.d_vs_lines_wins.abs().max():.2f}; "
          f"rank corr {tab.joint.rank().corr(tab.lines_wins.rank()):.3f}")

    dest = snap_path.with_name(stem + "_joint.parquet")
    tab.reset_index(names="team").assign(tau=res.tau, hfa=res.hfa, lam_w=lam_w,
                                         lam_f=lam_f).to_parquet(dest, index=False)
    with open(CONFIG, "w") as f:
        yaml.safe_dump({"lam_f": float(lam_f), "lam_w": float(lam_w), "n_sims": N_SIMS,
                        "chosen_from": f"{snap_path.name}: held-out RMSE on {len(test)} Kalshi lines",
                        "heldout_rmse": {f"{k:g}": round(v, 4) for k, v in val.items()},
                        "tau_by_lam_f": {f"{k:g}": round(v.tau, 3) for k, v in fits.items()}},
                       f, sort_keys=False)
    print(f"\nwrote {dest.relative_to(ROOT)} and {CONFIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_snapshot())
