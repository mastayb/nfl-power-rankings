"""Historical tests, 1999-2025 regular seasons (nflverse closing lines + results).

Run: python scripts/historical_test.py

A. Market vs résumé at a cutoff week: which predicts the rest of the season,
   and does résumé-minus-market predict results against the closing spread?
B. "Broken teams": does recent performance vs the spread carry forward?
C. "Playing down": do teams that underperform the spread as favourites keep
   doing it?

Intervals are 90% season-bootstrap intervals (whole seasons resampled).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.backtest import (BREAKEVEN, cover_rate, cutoff_panel, load_history,  # noqa: E402
                              ols, season_bootstrap, team_games)

SEASONS = range(1999, 2026)
CUTOFFS = [3, 4, 6, 8]
N_BOOT = 500


def ci(draws) -> str:
    lo, hi = np.percentile(draws, [5, 95])
    return f"[{lo:+.3f}, {hi:+.3f}]"


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def coef(df, xcols, ycol, which=-1):
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in xcols])
    return ols(X, df[ycol].to_numpy(float))[which]


def bet_line(label: str, ats_for_bet: pd.Series) -> str:
    rate, n = cover_rate(ats_for_bet)
    return f"  {label:46} cover {rate:6.1%}  n={n:5}  (breakeven {BREAKEVEN:.1%})"


def section(t):
    print(f"\n{'=' * 4} {t} {'=' * max(0, 74 - len(t))}")


def test_market_vs_resume(hist: pd.DataFrame) -> None:
    section("A. Market vs résumé at cutoff week")
    print("Predicting final margins of all later games (RMSE, points). 'closing line' is each\n"
          "game's own closing spread (the market's best, latest estimate) as a benchmark.\n")
    print(f"{'cutoff':>6} {'games':>6} {'closing':>8} {'market@k':>9} {'résumé@k':>9}"
          f"   {'résumé weight in blend':>24}   {'ATS pts per gap pt':>20}")
    panels = {}
    for k in CUTOFFS:
        p = cutoff_panel(hist, k)
        panels[k] = p
        b_res = coef(p, ["market_pred", "resume_pred"], "result")
        b_res_ci = season_bootstrap(p, lambda d: coef(d, ["market_pred", "resume_pred"], "result"), N_BOOT)
        b_gap = coef(p, ["gap_diff"], "ats")
        b_gap_ci = season_bootstrap(p, lambda d: coef(d, ["gap_diff"], "ats"), N_BOOT)
        print(f"{k:6} {len(p):6} {rmse(p.spread_line, p.result):8.2f} {rmse(p.market_pred, p.result):9.2f}"
              f" {rmse(p.resume_pred, p.result):9.2f}   {b_res:+.3f} {ci(b_res_ci):>18}"
              f"   {b_gap:+.3f} {ci(b_gap_ci)}")

    print("\nRésumé weight: coefficient on the résumé prediction when regressing results on\n"
          "market + résumé predictions (0 = market already has it all).\n"
          "ATS pts per gap pt: regress (result - closing spread) on the home-minus-away gap,\n"
          "gap = résumé rating - market rating. >0 = market under-reacts to résumé.")

    p = panels[4]
    print("\nBetting the side whose résumé most exceeds its market rating (cutoff week 4):")
    for q in (0.5, 0.8, 0.9):
        thr = p.gap_diff.abs().quantile(q)
        sel = p[p.gap_diff.abs() >= thr]
        print(bet_line(f"top {100 * (1 - q):.0f}% |gap diff| (>= {thr:.1f} pts)",
                       np.sign(sel.gap_diff) * sel.ats))
    print("By weeks after the cutoff (all games, sign of gap):")
    for lo, hi in [(1, 1), (2, 4), (5, 99)]:
        sel = p[p.weeks_ahead.between(lo, hi)]
        print(bet_line(f"weeks ahead {lo}-{hi if hi < 99 else 'end'}", np.sign(sel.gap_diff) * sel.ats))


def test_ats_momentum(hist: pd.DataFrame, tg: pd.DataFrame) -> None:
    section("B. 'Broken teams': does recent ATS performance carry forward?")
    tg = tg.copy()
    tg["prev_ats"] = tg.groupby(["season", "team"]).ats.shift(1)
    lag1 = tg.dropna(subset=["prev_ats"])
    print(f"Team-level corr(ATS this game, ATS last game): {lag1.ats.corr(lag1.prev_ats):+.3f} "
          f"(n={len(lag1)})")

    for n in (3, 5):
        tg[f"form{n}"] = (tg.groupby(["season", "team"]).ats
                          .transform(lambda s: s.shift(1).rolling(n, min_periods=n).mean()))
    games = hist.merge(tg[["game_id", "team", "form3", "form5"]].rename(
        columns={"team": "home_team", "form3": "f3h", "form5": "f5h"}), on=["game_id", "home_team"])
    games = games.merge(tg[["game_id", "team", "form3", "form5"]].rename(
        columns={"team": "away_team", "form3": "f3a", "form5": "f5a"}), on=["game_id", "away_team"])
    for n in (3, 5):
        g = games.dropna(subset=[f"f{n}h", f"f{n}a"]).copy()
        g["form_diff"] = g[f"f{n}h"] - g[f"f{n}a"]
        b = coef(g, ["form_diff"], "ats")
        bci = season_bootstrap(g, lambda d: coef(d, ["form_diff"], "ats"), N_BOOT)
        print(f"ATS ~ (home - away) mean ATS over last {n}: {b:+.3f} pts per pt {ci(bci)}  n={len(g)}")

    t = tg.dropna(subset=["form3"])
    print("\nNext-game ATS for teams by how badly they missed the spread over their last 3:")
    for label, sel in [("missed by 10+ per game (worst slump)", t[t.form3 <= -10]),
                       ("missed by 7+", t[t.form3 <= -7]),
                       ("beat by 7+", t[t.form3 >= 7]),
                       ("beat by 10+ (hottest)", t[t.form3 >= 10])]:
        print(bet_line(f"  team {label}", sel.ats))


def test_playing_down(tg: pd.DataFrame) -> None:
    section("C. 'Playing down': persistence of underperformance as a favourite")
    fav = tg[tg.spread >= 3].copy()
    print("League-wide, favourite's ATS by size of spread:")
    for lo, hi in [(3, 6.5), (7, 9.5), (10, 13.5), (14, 99)]:
        sel = fav[fav.spread.between(lo, hi)]
        rate, n = cover_rate(sel.ats)
        print(f"  favoured by {lo:>4}-{hi if hi < 99 else '+':<5} mean ATS {sel.ats.mean():+.2f}  "
              f"fav cover {rate:.1%}  n={n}")

    half = fav.assign(half=np.where(fav.week <= 9, "h1", "h2"))
    ts = half.groupby(["season", "team", "half"]).ats.agg(["mean", "size"]).unstack("half")
    ts = ts[(ts[("size", "h1")] >= 3) & (ts[("size", "h2")] >= 3)]
    pairs = pd.DataFrame({"season": ts.index.get_level_values(0),
                          "h1": ts[("mean", "h1")].to_numpy(), "h2": ts[("mean", "h2")].to_numpy()})
    r = pairs.h1.corr(pairs.h2)
    rci = season_bootstrap(pairs, lambda d: d.h1.corr(d.h2), N_BOOT)
    print(f"\nWithin season: corr(fav ATS weeks 1-9, fav ATS weeks 10+): {r:+.3f} {ci(rci)}  "
          f"team-seasons={len(pairs)}")

    fav["prior_fav_ats"] = (fav.groupby(["season", "team"]).ats
                            .transform(lambda s: s.shift(1).expanding(min_periods=3).mean()))
    f = fav.dropna(subset=["prior_fav_ats"])
    b = coef(f, ["prior_fav_ats"], "ats")
    bci = season_bootstrap(f, lambda d: coef(d, ["prior_fav_ats"], "ats"), N_BOOT)
    print(f"Next favoured game ATS ~ mean ATS in this season's earlier favoured games (>=3): "
          f"{b:+.3f} {ci(bci)}  n={len(f)}")
    for label, sel in [("prior fav ATS <= -5 (chronic underperformer)", f[f.prior_fav_ats <= -5]),
                       ("prior fav ATS >= +5", f[f.prior_fav_ats >= 5])]:
        print(bet_line(f"  {label}: bet favourite", sel.ats))

    ys = fav.groupby(["team", "season"]).ats.agg(["mean", "size"]).reset_index()
    ys = ys[ys["size"] >= 4]
    nxt = ys.assign(season=ys.season - 1).rename(columns={"mean": "next", "size": "n_next"})
    yy = ys.merge(nxt, on=["team", "season"])
    rci = season_bootstrap(yy, lambda d: d["mean"].corr(d["next"]), N_BOOT)
    print(f"Across seasons: corr(fav ATS season S, season S+1): {yy['mean'].corr(yy['next']):+.3f} "
          f"{ci(rci)}  pairs={len(yy)}")


def main() -> None:
    hist = load_history(SEASONS)
    print(f"{len(hist)} regular-season games with closing lines, {SEASONS.start}-{SEASONS.stop - 1}")
    tg = team_games(hist)
    test_market_vs_resume(hist)
    test_ats_momentum(hist, tg)
    test_playing_down(tg)


if __name__ == "__main__":
    main()
