"""Stage B, lines only: team ratings from current Kalshi lines + past closing lines.

Run: python scripts/stage_b.py [data/snapshots/<ts>.parquet]

Model: expected home margin = r_home - r_away + H (H = 0 at neutral sites),
sum r = 0, weighted least squares. Inputs are Stage A's expected margins for
games open on Kalshi plus decayed closing lines of games already played.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.blend import blend_ratings, completed_week, load_blend  # noqa: E402
from ratings.ladder import load_key_weights, win_prob  # noqa: E402
from ratings.luck import load_pbp, luck_table  # noqa: E402
from ratings.linear import fit_margins, line_observations, predict_margin  # noqa: E402
from ratings.schedule import NFLVERSE_TEAMS, load_schedule  # noqa: E402

TEAMS = sorted(NFLVERSE_TEAMS)
SNAP_DIR = ROOT / "data" / "snapshots"
HALF_LIFE = 3.0    # weeks
LINE_NOISE = 2.0   # pts, closing-line noise vs true margin (residual SD of lines-only fit)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def latest_snapshot() -> Path:
    return sorted(p for p in SNAP_DIR.glob("*.parquet") if "_" not in p.stem)[-1]


def records(schedule: pd.DataFrame) -> pd.Series:
    """W-L (-T) per team from played games."""
    p = schedule[schedule.result.notna()]
    out = {}
    for t in TEAMS:
        home, away = p[p.home_team == t].result, -p[p.away_team == t].result
        m = pd.concat([home, away])
        w, l, ties = (m > 0).sum(), (m < 0).sum(), (m == 0).sum()
        out[t] = f"{w}-{l}" + (f"-{ties}" if ties else "")
    return pd.Series(out)


def stage_a_sigma(stage_a: pd.DataFrame) -> float:
    # ML-only games carry the pooled sigma from Stage A.
    pooled = stage_a.loc[stage_a.sigma_source == "pooled", "sigma"]
    return float(pooled.iloc[0]) if len(pooled) else float(stage_a.sigma.median())


def main(snap_path: Path) -> None:
    stem = snap_path.stem
    stage_a = pd.read_parquet(snap_path.with_name(stem + "_stage_a.parquet"))
    games = pd.read_parquet(snap_path.with_name(stem + "_games.parquet"))
    asof = pd.Timestamp(stem)
    season = int(games.game_id.str[:4].iloc[0])
    sched = load_schedule(season)
    model = load_key_weights()
    sigma = stage_a_sigma(stage_a)

    def obs_for(half_life=HALF_LIFE, results_weight=0.0):
        return line_observations(stage_a, games, sched, model, sigma, asof,
                                 half_life_weeks=half_life, results_weight=results_weight)

    print(f"snapshot {snap_path.relative_to(ROOT)}  asof {asof:%Y-%m-%d %H:%M}Z  "
          f"sigma {sigma:.2f} (Stage A pooled)")

    # 1. Kalshi lines alone
    obs = obs_for()
    try:
        fit_margins(obs[obs.source == "kalshi"], "y", TEAMS)
        print("Kalshi-only fit: identified")
    except ValueError as e:
        print(f"Kalshi-only fit: {e} -> adding past closing lines")

    # 2. Main fit: Kalshi + decayed closing lines
    fit = fit_margins(obs, "y", TEAMS, weights=obs.weight.to_numpy())
    obs["pred"] = predict_margin(fit, obs)
    obs["resid"] = obs.y - obs.pred

    tab = pd.DataFrame({"r": fit.ratings, "se": fit.se})
    tab["win_vs_avg"] = [win_prob(model, r, sigma) for r in tab.r]
    tab = tab.sort_values("r", ascending=False)
    tab.insert(0, "rank", range(1, len(tab) + 1))

    print(f"\n=== Ratings: Kalshi + closing lines (half-life {HALF_LIFE:g} wk) ===")
    print(f"inputs: {(obs.source == 'kalshi').sum()} Kalshi games, "
          f"{(obs.source == 'close').sum()} closing lines "
          f"(weights {obs.loc[obs.source == 'close', 'weight'].min():.2f}-"
          f"{obs.loc[obs.source == 'close', 'weight'].max():.2f})")
    print(f"HFA {fit.hfa:.2f} (se {fit.hfa_se:.2f})   sum r = {fit.ratings.sum():.1e}   "
          f"weighted resid SD {fit.resid_sd:.2f}")
    print(tab.to_string(formatters={"r": "{:+.2f}".format, "se": "{:.2f}".format,
                                    "win_vs_avg": "{:.3f}".format}))

    print("\nResidual (input - fitted) by source:")
    for src, g in obs.groupby("source"):
        print(f"  {src:7} n={len(g):3}  RMSE {np.sqrt(np.mean(g.resid**2)):.2f}  "
              f"weighted RMSE {np.sqrt(np.average(g.resid**2, weights=g.weight)):.2f}")
    k = obs[obs.source == "kalshi"].reindex(columns=["game_id", "y", "pred", "resid"])
    k = k.reindex(k.resid.abs().sort_values(ascending=False).index)
    print("\nLargest Kalshi residuals (market expected margin vs ratings):")
    print(k.head(6).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    # 3. Measured résumé blend
    k = completed_week(sched)
    blend = load_blend(k)
    br, b_hfa = blend_ratings(fit.ratings, fit.hfa, sched, blend)
    br["record"] = records(sched).reindex(br.index)
    br["rank_market"] = br.market.rank(ascending=False).astype(int)
    br["rank_resume"] = br.resume.rank(ascending=False).astype(int)
    br["rank"] = br.blend.rank(ascending=False).astype(int)
    br["move"] = br.rank_market - br["rank"]
    br["win_vs_avg"] = [win_prob(model, r, sigma) for r in br.blend]
    br = br.sort_values("rank")
    print(f"\n=== Blended rating: {blend.b_market:.3f} x market + {blend.b_resume:.3f} x résumé "
          f"(cutoff week {blend.cutoff_week}) ===")
    print(f"résumé = schedule-adjusted results through week {blend.cutoff_week}, margins capped "
          f"+-{blend.cap:g}, {blend.half_life:g}-week half-life.  Blended HFA {b_hfa:.2f}")
    luck = luck_table(sched, load_pbp(season), TEAMS)
    br = br.join(luck)
    br["close"] = (br.close_w.astype(str) + "-" + br.close_l.astype(str)
                   + np.where(br.close_t > 0, "-" + br.close_t.astype(str), ""))
    br["fum"] = br.recovered.astype(str) + "/" + br.fumbles.astype(str)
    print(br[["rank", "move", "record", "blend", "win_vs_avg", "market", "rank_market",
              "resume", "rank_resume", "close", "close_luck", "fum", "fum_luck", "fum_pts",
              "to_margin"]].to_string(formatters={
        "blend": "{:+.2f}".format, "market": "{:+.2f}".format, "resume": "{:+.2f}".format,
        "win_vs_avg": "{:.3f}".format, "move": "{:+d}".format, "close_luck": "{:+.1f}".format,
        "fum_luck": "{:+.1f}".format, "fum_pts": "{:+.1f}".format, "to_margin": "{:+d}".format}))
    print("close = one-score (<=8) record; close_luck = wins above .500 in those games.\n"
          "fum = fumbles recovered / recoverable fumbles in the team's games; fum_luck = recoveries\n"
          "above 50%; fum_pts = that x 4.16 pts. Luck columns use every played game (incl. week "
          f"{blend.cutoff_week + 1} so far).")

    # 4. Sensitivity
    print("\n=== Sensitivity (vs main fit) ===")
    print(f"{'variant':34} {'HFA':>5} {'max|dr|':>8} {'rank corr':>9}  top 5")
    variants = [(f"half-life {h} wk", dict(half_life=h)) for h in (1.5, 6.0)]
    variants += [("no decay", dict(half_life=None)),
                 (f"+ results, weight (2/sigma)^2", dict(results_weight=(LINE_NOISE / sigma) ** 2)),
                 ("+ results, weight 10x that", dict(results_weight=10 * (LINE_NOISE / sigma) ** 2))]
    for name, kw in variants:
        o = obs_for(**kw)
        f = fit_margins(o, "y", TEAMS, weights=o.weight.to_numpy())
        d = (f.ratings - fit.ratings).abs().max()
        rc = f.ratings.rank().corr(fit.ratings.rank())
        top = ",".join(f.ratings.sort_values(ascending=False).index[:5])
        print(f"{name:34} {f.hfa:5.2f} {d:8.2f} {rc:9.3f}  {top}")

    dest = snap_path.with_name(stem + "_stage_b.parquet")
    out = tab.reset_index(names="team").assign(hfa=fit.hfa, sigma=sigma, half_life=HALF_LIFE)
    out = out.merge(br[["resume", "blend", "rank", "record", "close", "close_luck", "fumbles",
                        "recovered", "fum_luck", "fum_pts", "to_margin"]].rename(
        columns={"rank": "blend_rank"}).reset_index(names="team"), on="team")
    out["blend_hfa"], out["blend_cutoff_week"] = b_hfa, blend.cutoff_week
    out.to_parquet(dest, index=False)
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_snapshot())
