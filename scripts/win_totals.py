"""Phase 5: fit market ratings to game lines + KXNFLWINS win totals jointly.

Run: python scripts/win_totals.py [data/snapshots/<ts>.parquet]

1. Choose lambda (win-total weight) by validation: fit on past closing lines
   + win totals, holding out the current Kalshi game lines, and score how well
   the ratings predict those held-out lines.
2. Refit on everything with the chosen lambda; compare each team's expected
   wins (model vs market) and how much ratings moved vs lines only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.ladder import load_key_weights  # noqa: E402
from ratings.linear import fit_margins, line_observations  # noqa: E402
from ratings.schedule import NFLVERSE_TEAMS, load_schedule  # noqa: E402
from ratings.wins import (N_GAMES, Season, WinProbs, expected_wins_market,  # noqa: E402
                          fit_joint, market_ladders)

N_OUT = N_GAMES + 1

TEAMS = sorted(NFLVERSE_TEAMS)
SNAP_DIR = ROOT / "data" / "snapshots"
LAMBDAS = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
TOLERANCE = 0.01
CONFIG = ROOT / "config" / "win_totals.yaml"

pd.set_option("display.width", 200)


def latest_snapshot() -> Path:
    return sorted(p for p in SNAP_DIR.glob("*.parquet") if "_" not in p.stem)[-1]


def ladders_ew(ladders, season_state) -> np.ndarray:
    return np.array([expected_wins_market(ladders[t], season_state.wins[i])
                     for i, t in enumerate(TEAMS)])


def validate(train, test, season_state, ladders, probs, start, e_market, fixed_hfa) -> dict:
    """Held-out RMSE on `test` lines for each lambda; prints a row per lambda."""
    out = {}
    for lam in LAMBDAS:
        f = fit_joint(train, season_state, ladders, probs, lam, start=start, fixed_hfa=fixed_hfa)
        r = f.ratings
        pred = (r[test.home_team].to_numpy() - r[test.away_team].to_numpy()
                + f.hfa * ~test.neutral.to_numpy(bool))
        rmse = float(np.sqrt(np.mean((test.y.to_numpy() - pred) ** 2)))
        ew = season_state.wins_dist(r.to_numpy(), f.hfa, probs) @ np.arange(N_OUT)
        de = float(np.mean(np.abs(ew - e_market)))
        out[lam] = rmse
        print(f"  {lam:7g} {rmse:14.3f} {f.hfa:6.2f} {f.wins_kl:13.4f} {de:13.2f}"
              + ("" if f.success else "  (optimizer warning)"))
    return out


def main(snap_path: Path) -> None:
    stem = snap_path.stem
    tidy = pd.read_parquet(snap_path)
    stage_a = pd.read_parquet(snap_path.with_name(stem + "_stage_a.parquet"))
    games = pd.read_parquet(snap_path.with_name(stem + "_games.parquet"))
    season = int(games.game_id.str[:4].iloc[0])
    sched = load_schedule(season)
    model = load_key_weights()
    sigma = float(stage_a.loc[stage_a.sigma_source == "pooled", "sigma"].iloc[0])
    obs = line_observations(stage_a, games, sched, model, sigma, pd.Timestamp(stem))
    season_state = Season.build(sched, TEAMS)
    probs = WinProbs(model, sigma)
    ladders = market_ladders(tidy)
    print(f"snapshot {snap_path.relative_to(ROOT)}: {len(obs)} line observations, "
          f"{len(ladders)} win-total ladders, {len(season_state.home_idx)} remaining games")

    def lines_start(o):
        f = fit_margins(o, "y", TEAMS, weights=o.weight.to_numpy())
        return f.ratings.to_numpy(), f.hfa

    # 1. validation: hold out current Kalshi lines
    train = obs[obs.source != "kalshi"]
    test = obs[obs.source == "kalshi"]
    start = lines_start(train)
    val = {}
    print(f"\n1. Choose lambda: fit on {len(train)} past closing lines + win totals, "
          f"predict {len(test)} held-out Kalshi game lines")
    hfa_train = start[1]  # lines-only HFA from the training lines (no peeking at held-out)
    for label, fix in [("HFA free", None), (f"HFA fixed at lines-only {hfa_train:.2f}", hfa_train)]:
        print(f"  [{label}]")
        print(f"  {'lambda':>7} {'held-out RMSE':>14} {'HFA':>6} {'win-total KL':>13} "
              f"{'mean |dE[W]|':>13}")
        val[fix] = validate(train, test, season_state, ladders, probs, start, ladders_ew(
            ladders, season_state), fix)
    # The held-out error plateaus once win totals dominate, so take the smallest
    # lambda within TOLERANCE of the best (a one-standard-error-style rule)
    # instead of the arg-min, which drifts to the edge of the grid on noise.
    best_fix = min(val, key=lambda f: min(val[f].values()))
    floor = min(val[best_fix].values())
    best = min(lam for lam, e in val[best_fix].items() if e <= floor * (1 + TOLERANCE))
    fix_hfa = best_fix is not None
    print(f"-> chosen: lambda = {best:g} (smallest within {TOLERANCE:.0%} of best held-out RMSE "
          f"{floor:.3f}), HFA {'fixed at lines-only' if fix_hfa else 'free'}")
    with open(CONFIG, "w") as fh:
        yaml.safe_dump({
            "lambda": float(best), "fix_hfa_to_lines": fix_hfa,
            "chosen_from": f"{snap_path.name}: held-out RMSE on {len(test)} Kalshi lines",
            "heldout_rmse": {f"{lam:g}": round(e, 4) for lam, e in val[best_fix].items()},
        }, fh, sort_keys=False)
    print(f"wrote {CONFIG.relative_to(ROOT)}")

    # 2. final fit on everything
    start = lines_start(obs)
    f0 = fit_joint(obs, season_state, ladders, probs, 0.0, start=start)
    f1 = fit_joint(obs, season_state, ladders, probs, best, start=start,
                   fixed_hfa=start[1] if fix_hfa else None)
    d0 = season_state.wins_dist(f0.ratings.to_numpy(), f0.hfa, probs) @ np.arange(18)
    d1 = season_state.wins_dist(f1.ratings.to_numpy(), f1.hfa, probs) @ np.arange(18)
    tab = pd.DataFrame({
        "wins_now": season_state.wins,
        "E_market": [expected_wins_market(ladders[t], season_state.wins[i]) for i, t in enumerate(TEAMS)],
        "E_lines": d0, "E_joint": d1,
        "r_lines": f0.ratings.to_numpy(), "r_joint": f1.ratings.to_numpy(),
    }, index=TEAMS)
    tab["miss_lines"] = tab.E_lines - tab.E_market
    tab["miss_joint"] = tab.E_joint - tab.E_market
    tab["dr"] = tab.r_joint - tab.r_lines
    tab["rank_lines"] = tab.r_lines.rank(ascending=False).astype(int)
    tab["rank_joint"] = tab.r_joint.rank(ascending=False).astype(int)
    tab = tab.sort_values("r_joint", ascending=False)

    print(f"\n2. Final fit, lambda = {best:g}: HFA {f0.hfa:.2f} -> {f1.hfa:.2f}, line RMSE "
          f"{f0.line_rmse:.2f} -> {f1.line_rmse:.2f}, win-total KL {f0.wins_kl:.4f} -> "
          f"{f1.wins_kl:.4f}")
    print(tab.to_string(float_format=lambda x: f"{x:+.2f}"))
    print(f"\nmean |expected wins - market|: lines only {tab.miss_lines.abs().mean():.2f}, "
          f"joint {tab.miss_joint.abs().mean():.2f};  max |rating change| {tab.dr.abs().max():.2f}; "
          f"rank corr {tab.r_lines.rank().corr(tab.r_joint.rank()):.3f}")
    print("\nLargest remaining misses (model expected wins - market), joint fit:")
    print(tab.reindex(tab.miss_joint.abs().sort_values(ascending=False).index)
          [["wins_now", "E_market", "E_joint", "miss_joint"]].head(6)
          .to_string(float_format=lambda x: f"{x:+.2f}"))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_snapshot())
