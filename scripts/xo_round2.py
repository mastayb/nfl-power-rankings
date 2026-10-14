"""Xs and Os round 2: what explains the market's game-specific premium?

Run: python -u scripts/xo_round2.py

H3  premium_g = closing spread - aggregate-rating prediction, where aggregate
    ratings are fitted to every OTHER game's closing line that season, weighted
    toward nearby weeks (3-week half-life). Regressed on pre-game "mundane"
    factors: backup QB starts, rest difference, divisional game, high wind.
H4  add a coverage-shell matchup score built from PRIOR weeks only:
        bonus(O vs D) = (singlehigh_usage_D - league) x (gapSH_O - gapTH_O) x dropbacks
    gaps = O's situation-adjusted EPA vs single-/two-high minus its own pass EPA,
    shrunk for sample size. Does it explain premium beyond H3?
H5  same score vs (result - closing spread): is it priced?
OQ  opponent-quality sensitivity: per team-season slope of game offensive
    (defensive) EPA/play on opponent defensive (offensive) quality, opponent
    quality excluding games vs the team. Persistence odd vs even games and
    season to season; same for game-to-game consistency (residual SD).

All regressions use season-cluster bootstrap 90% intervals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ratings.backtest import ridge_ratings  # noqa: E402
from ratings.pbp_cache import load_pbp_cached  # noqa: E402

SEASONS = range(2016, 2026)
LEAGUE_SH = 0.60
DROPBACKS = 38
N_BOOT = 500
pd.set_option("display.width", 200)


# --- data --------------------------------------------------------------------------------

def schedules() -> pd.DataFrame:
    import nflreadpy as nfl

    s = nfl.load_schedules(seasons=list(SEASONS)).to_pandas()
    s = s[(s.game_type == "REG") & s.result.notna() & s.spread_line.notna()].copy()
    s["neutral"] = s.location.eq("Neutral")
    return s.reset_index(drop=True)


def local_loo_predictions(s: pd.DataFrame, half_life: float = 3.0) -> pd.Series:
    """Aggregate-skill prediction for each game from all OTHER games' closing lines."""
    out = pd.Series(np.nan, index=s.index)
    for _, g in s.groupby("season"):
        for i in g.index:
            o = g.drop(index=i)
            w = 0.5 ** (np.abs(o.week - g.at[i, "week"]) / half_life)
            r, hfa = ridge_ratings(o, o.spread_line.to_numpy(float), w.to_numpy(float))
            out[i] = (r.get(g.at[i, "home_team"], 0.0) - r.get(g.at[i, "away_team"], 0.0)
                      + (0.0 if g.at[i, "neutral"] else hfa))
    return out


def mundane(s: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=s.index)
    # backup QB: starter differs from the team's most common starter that season
    starts = pd.concat([s[["season", "home_team", "home_qb_id"]].set_axis(["season", "team", "qb"], axis=1),
                        s[["season", "away_team", "away_qb_id"]].set_axis(["season", "team", "qb"], axis=1)])
    mode = starts.groupby(["season", "team"]).qb.agg(lambda q: q.mode().iloc[0] if q.notna().any() else None)
    hb = [q != mode.get((y, t)) for y, t, q in zip(s.season, s.home_team, s.home_qb_id)]
    ab = [q != mode.get((y, t)) for y, t, q in zip(s.season, s.away_team, s.away_qb_id)]
    f["qb_backup_diff"] = np.array(hb, float) - np.array(ab, float)
    f["rest_diff"] = (s.home_rest - s.away_rest).clip(-7, 7).fillna(0)
    f["div_x_pred"] = s.div_game.fillna(0) * s.pred
    f["wind_x_pred"] = (s.wind.fillna(0) >= 15) * s.pred
    return f


# --- shell matchup score from prior weeks -------------------------------------------------

def shell_scores(s: pd.DataFrame) -> pd.Series:
    from xo_explore import adjusted_epa, load

    d = load()
    d["adj"] = adjusted_epa(d)
    p = d[(d["pass"] == 1) & (d.season >= 2018)].copy()
    resid_sd = p.adj.std()
    k = (resid_sd / 0.055) ** 2            # shrinkage: noise var / persistent var (round 1)
    p["sh"] = (p.shell == "single-high").astype(float)
    p["th"] = (p.shell == "two-high").astype(float)
    p["known"] = p.shell.isin(["single-high", "two-high"]).astype(float)

    def cum(team_col, cols):
        g = p.groupby(["season", team_col, "week"])[cols].sum().groupby(level=[0, 1]).cumsum()
        return g.groupby(level=[0, 1]).shift(1).fillna(0)   # strictly prior weeks

    p["adj_sh"] = p.adj * p.sh
    p["adj_th"] = p.adj * p.th
    p["one"] = 1.0
    off = cum("posteam", ["adj", "one", "adj_sh", "sh", "adj_th", "th"])
    dfn = cum("defteam", ["sh", "known"])

    def off_gap(season, team, week):
        try:
            r = off.loc[(season, team, week)]
        except KeyError:
            prior = off.loc[(season, team)] if (season, team) in off.index.droplevel(2) else None
            if prior is None:
                return 0.0
            prior = prior[prior.index < week]
            if prior.empty:
                return 0.0
            r = prior.iloc[-1]
        if r.one == 0:
            return 0.0
        base = r.adj / r.one
        gsh = (r.adj_sh / r.sh - base) * r.sh / (r.sh + k) if r.sh else 0.0
        gth = (r.adj_th / r.th - base) * r.th / (r.th + k) if r.th else 0.0
        return gsh - gth

    def usage(season, team, week, m=50.0):
        try:
            r = dfn.loc[(season, team, week)]
        except KeyError:
            return LEAGUE_SH
        return (r.sh + LEAGUE_SH * m) / (r.known + m)

    out = pd.Series(np.nan, index=s.index)
    for i, g in s[s.season >= 2018].iterrows():
        y, w = g.season, g.week
        hb = (usage(y, g.away_team, w) - LEAGUE_SH) * off_gap(y, g.home_team, w) * DROPBACKS
        ab = (usage(y, g.home_team, w) - LEAGUE_SH) * off_gap(y, g.away_team, w) * DROPBACKS
        out[i] = hb - ab
    return out


# --- regression helpers ---------------------------------------------------------------------

def ols(df, y, xs):
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in xs])
    b = np.linalg.lstsq(X, df[y].to_numpy(float), rcond=None)[0]
    resid = df[y].to_numpy(float) - X @ b
    r2 = 1 - resid.var() / df[y].var()
    return pd.Series(b[1:], index=xs), r2, resid.std()


def boot(df, y, xs, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("season")]
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        draws.append(ols(pd.concat([groups[i] for i in pick]), y, xs)[0])
    return pd.DataFrame(draws).quantile([0.05, 0.95]).T


def report(df, y, xs, title):
    b, r2, sd = ols(df, y, xs)
    ci = boot(df, y, xs)
    print(f"\n{title}  (n={len(df)}, R^2 {r2:.3f}, resid SD {sd:.2f}; SD of {y} {df[y].std():.2f})")
    print(pd.DataFrame({"coef": b, "lo90": ci[0.05], "hi90": ci[0.95]}).round(3).to_string())


# --- opponent-quality sensitivity ------------------------------------------------------------

def opponent_quality(seasons=range(2006, 2026)) -> pd.DataFrame:
    rows = []
    for y in seasons:
        p = load_pbp_cached(y)
        p = p[(p.season_type == "REG") & p.play_type.isin(["run", "pass"]) & p.epa.notna()]
        g = p.groupby(["game_id", "posteam", "defteam"]).epa.agg(["sum", "size"]).reset_index()
        g["season"] = y
        g["week"] = g.game_id.str.slice(5, 7).astype(int)
        rows.append(g)
    g = pd.concat(rows, ignore_index=True)
    # opponent quality excluding games against the team being rated
    dtot = g.groupby(["season", "defteam"])[["sum", "size"]].sum()
    otot = g.groupby(["season", "posteam"])[["sum", "size"]].sum()
    pair = g.groupby(["season", "posteam", "defteam"])[["sum", "size"]].sum()

    def q(tot, season, team, vs, swap):
        key_pair = (season, vs, team) if swap else (season, team, vs)
        t = tot.loc[(season, team)]
        pr = pair.loc[key_pair] if key_pair in pair.index else pd.Series({"sum": 0, "size": 0})
        return (t["sum"] - pr["sum"]) / max(t["size"] - pr["size"], 1)

    g["off_epa"] = g["sum"] / g["size"]
    g["opp_def_q"] = [q(dtot, s_, d_, o_, swap=False) for s_, d_, o_ in zip(g.season, g.defteam, g.posteam)]
    g["opp_off_q"] = [q(otot, s_, o_, d_, swap=False) for s_, o_, d_ in zip(g.season, g.posteam, g.defteam)]
    return g


def slopes(g, team_col, perf, qcol, sign):
    out = []
    for (season, team), t in g.groupby(["season", team_col]):
        t = t.sort_values("week")
        for part, sub in [("odd", t.iloc[0::2]), ("even", t.iloc[1::2]), ("all", t)]:
            x = sub[qcol] - g[qcol].mean()
            yv = sign * sub[perf]
            if len(sub) < 6 or x.std() == 0:
                continue
            b = np.polyfit(x, yv, 1)
            res = yv - np.polyval(b, x)
            out.append({"season": season, "team": team, "part": part, "slope": b[0],
                        "level": yv.mean(), "consistency_sd": res.std(ddof=2)})
    return pd.DataFrame(out)


def persistence_table(sl: pd.DataFrame, label: str) -> pd.DataFrame:
    wide = sl[sl.part.isin(["odd", "even"])].pivot_table(index=["season", "team"], columns="part",
                                                          values=["slope", "level", "consistency_sd"])
    al = sl[sl.part == "all"].set_index(["season", "team"])
    nxt = al.reset_index().assign(season=lambda x: x.season - 1).set_index(["season", "team"])
    yy = al.join(nxt, rsuffix="_next", how="inner")
    rows = []
    for m in ["level", "slope", "consistency_sd"]:
        rows.append({"side": label, "measure": m,
                     "odd_vs_even_r": wide[m]["odd"].corr(wide[m]["even"]),
                     "season_to_season_r": yy[m].corr(yy[f"{m}_next"]),
                     "team_seasons": len(wide)})
    return pd.DataFrame(rows)


def main() -> None:
    s = schedules()
    print(f"{len(s)} games {SEASONS.start}-{SEASONS.stop - 1}; fitting leave-one-out local ratings...", flush=True)
    s["pred"] = local_loo_predictions(s)
    s["premium"] = s.spread_line - s.pred
    s["ats"] = s.result - s.spread_line
    f = mundane(s)
    s = s.join(f)
    print(f"premium: SD {s.premium.std():.2f} pts, mean |premium| {s.premium.abs().mean():.2f}")

    xs3 = ["qb_backup_diff", "rest_diff", "div_x_pred", "wind_x_pred"]
    report(s, "premium", xs3, "=== H3: market premium ~ mundane factors (2016-2025)")

    print("\nbuilding prior-weeks shell matchup scores (2018+)...", flush=True)
    s["shell_score"] = shell_scores(s)
    h = s[(s.season >= 2018) & (s.week >= 5) & s.shell_score.notna()]
    print(f"shell_score: SD {h.shell_score.std():.2f} pts; 90th pct |score| {h.shell_score.abs().quantile(.9):.2f}")
    report(h, "premium", xs3, "=== H4 base: premium ~ mundane (2018+, week >= 5)")
    report(h, "premium", xs3 + ["shell_score"], "=== H4: premium ~ mundane + shell matchup")
    report(h, "ats", ["shell_score"], "=== H5: result - closing spread ~ shell matchup")
    report(h, "ats", xs3, "=== H5 control: result - closing spread ~ mundane (should be ~0 if priced)")

    print("\n=== Opponent-quality sensitivity (2006-2025) ===", flush=True)
    g = opponent_quality()
    off = slopes(g, "posteam", "off_epa", "opp_def_q", +1)
    dfn = slopes(g, "defteam", "off_epa", "opp_off_q", -1)    # defense: higher = better
    tab = pd.concat([persistence_table(off, "offense vs opp defense"),
                     persistence_table(dfn, "defense vs opp offense")])
    print("level = overall efficiency; slope = sensitivity to opponent quality; "
          "consistency_sd = game-to-game scatter around the team's own line")
    print(tab.round(3).to_string(index=False))
    sl = off[off.part == "all"].slope
    print(f"offense slope across team-seasons: mean {sl.mean():.2f} (1 = league-typical), SD {sl.std():.2f}")

    out = ROOT / "data" / "xo_round2.parquet"
    s[["game_id", "season", "week", "home_team", "away_team", "spread_line", "pred", "premium",
       "result", "ats"] + xs3 + ["shell_score"]].to_parquet(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
