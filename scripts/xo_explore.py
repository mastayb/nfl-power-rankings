"""Xs and Os exploration, round 1 (2016-2025 regular seasons).

Run: python -u scripts/xo_explore.py

A. How defenses respond to offensive personnel, and how much teams differ.
H1. League-wide personnel x personnel effects on EPA/play, net of offense and
    defense team-season quality and situation (down x distance, field position,
    win probability). Leave-one-season-out jackknife standard errors.
H2. Do team-specific style skills persist within a season? For each team-season
    and context (e.g. "offense in 12 personnel"), the team's situation-adjusted
    EPA in that context minus its own overall adjusted EPA, first half (weeks
    1-9) vs second half (10+). Benchmark: overall offense/defense EPA
    persistence. If context skills don't persist, a team x context model has
    nothing to estimate.

Caveat for all of it: personnel and coverage are chosen, not assigned, so these
are associations ("when this matchup happens, ...") not causal effects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratings.xo import load_plays  # noqa: E402

pd.set_option("display.width", 200)
OFF = ["11", "12", "13", "21", "22", "10"]
DEF = ["nickel", "base", "dime"]


def load() -> pd.DataFrame:
    path = ROOT / "data" / "cache" / "xo_plays.parquet"
    d = pd.read_parquet(path) if path.exists() else load_plays()
    # two-point tries etc. have no down; drop anything missing the situation controls
    d = d[d.def_pers.isin(DEF) & d.down.notna() & d.ydstogo.notna() & d.yardline_100.notna()
          & d.wp.notna()].copy()
    d["off_g"] = np.where(d.off_pers.isin(OFF), d.off_pers, "other")
    d["dd"] = (d.down.astype(int).astype(str) + "_"
               + pd.cut(d.ydstogo, [0, 2, 5, 10, 99], labels=["s", "m", "l", "xl"]).astype(str))
    d["fp"] = pd.cut(d.yardline_100, [0, 10, 20, 40, 60, 80, 100]).astype(str)
    d["wpb"] = pd.cut(d.wp, [-0.01, 0.1, 0.25, 0.75, 0.9, 1.01]).astype(str)
    d["half"] = np.where(d.week <= 9, 1, 2)
    return d


# --- design matrix helpers ------------------------------------------------------------

def one_hot(col: pd.Series, drop: str | None = None):
    cats = [c for c in pd.unique(col) if c != drop]
    idx = {c: i for i, c in enumerate(cats)}
    rows = np.flatnonzero(col.map(lambda v: v in idx).to_numpy())
    m = sp.csr_matrix((np.ones(len(rows)), (rows, col.iloc[rows].map(idx).to_numpy())),
                      shape=(len(col), len(cats)))
    return m, cats


def fit(d: pd.DataFrame, inter_cols: list[str]):
    """EPA ~ off team-season + def team-season + situation + off_g + def_pers + interactions."""
    blocks, names = [], []
    for col, drop in [("off_ts", None), ("def_ts", d.def_ts.iloc[0]), ("dd", "1_l"), ("fp", None),
                      ("wpb", None), ("off_g", "11"), ("def_pers", "nickel")]:
        m, cats = one_hot(d[col], drop)
        blocks.append(m)
        names += [f"{col}={c}" for c in cats]
    inter = pd.DataFrame({c: (d.off_g + "|" + d.def_pers == c).astype(float) for c in inter_cols})
    blocks.append(sp.csr_matrix(inter.to_numpy()))
    names += [f"x:{c}" for c in inter_cols]
    X = sp.hstack(blocks).tocsr()
    beta = lsqr(X, d.epa.to_numpy(), atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    return pd.Series(beta, index=names), X


def interaction_table(d: pd.DataFrame) -> pd.DataFrame:
    cells = [f"{o}|{p}" for o in OFF if o != "11" for p in DEF if p != "nickel"]
    beta, _ = fit(d, cells)
    full = beta[[f"x:{c}" for c in cells]].rename(lambda s: s[2:])
    jack = []
    seasons = sorted(d.season.unique())
    for s in seasons:
        b, _ = fit(d[d.season != s], cells)
        jack.append(b[[f"x:{c}" for c in cells]].rename(lambda s_: s_[2:]))
    jack = pd.DataFrame(jack)
    n = len(seasons)
    se = np.sqrt((n - 1) / n * ((jack - jack.mean()) ** 2).sum())
    counts = d.groupby(d.off_g + "|" + d.def_pers).size()
    return pd.DataFrame({"plays": counts.reindex(cells), "epa_interaction": full, "se": se,
                         "t": full / se})


def adjusted_epa(d: pd.DataFrame) -> pd.Series:
    """EPA minus the situation-only expectation (league average for down/dist, field, wp)."""
    base = d.groupby(["dd", "fp", "wpb"]).epa.transform("mean")
    return d.epa - base


def persistence(d: pd.DataFrame, side: str, contexts: dict[str, pd.Series], min_n: int = 25) -> pd.DataFrame:
    """Context EPA minus the team's own EPA on comparable plays (pass contexts are
    compared with the team's pass plays, so pass-vs-run strength isn't counted
    as coverage-specific skill)."""
    team = d.off_ts if side == "offense" else d.def_ts
    adj = d.adj
    rows = []
    overall = adj.groupby([team, d.half]).mean().unstack()
    is_pass = d["pass"] == 1
    overall_pass = adj[is_pass].groupby([team[is_pass], d.half[is_pass]]).mean().unstack()
    r_all = overall[1].corr(overall[2])
    rows.append({"context": f"{side}: overall EPA/play", "team_seasons": overall.dropna().shape[0],
                 "median_n_per_half": int(d.groupby([team, d.half]).size().median()),
                 "split_half_r": r_all})
    rows.append({"context": f"{side}: pass vs run gap", "team_seasons": overall_pass.dropna().shape[0],
                 "median_n_per_half": int(is_pass.groupby([team, d.half]).sum().median()),
                 "split_half_r": (overall_pass - overall)[1].corr((overall_pass - overall)[2])})
    for name, mask in contexts.items():
        g = d[mask]
        ref = overall_pass if (mask & ~is_pass).sum() == 0 else overall
        ctx = adj[mask].groupby([team[mask], g.half]).agg(["mean", "size"])
        diff = (ctx["mean"] - ref.stack().reindex(ctx.index)).unstack()
        n = ctx["size"].unstack()
        ok = (n[1] >= min_n) & (n[2] >= min_n)
        diff = diff[ok]
        rows.append({"context": f"{side}: {name} (vs own {'pass plays' if ref is overall_pass else 'overall'})", "team_seasons": int(ok.sum()),
                     "median_n_per_half": int(n[ok].stack().median()) if ok.any() else 0,
                     "split_half_r": diff[1].corr(diff[2])})
    return pd.DataFrame(rows)


def main() -> None:
    d = load()
    d["adj"] = adjusted_epa(d)
    print(f"{len(d):,} plays, {d.season.min()}-{d.season.max()}, {d.off_ts.nunique()} team-seasons")

    # A. defensive response to offensive personnel
    print("\n=== A. How defenses respond (share of plays, 2016-2025) ===")
    ct = pd.crosstab(d.off_g, d.def_pers, normalize="index").reindex(OFF + ["other"])[DEF]
    print(ct.round(3).to_string())
    heavy = d[d.off_g.isin(["12", "13", "22"])]
    ts = heavy.groupby("def_ts").def_pers.apply(lambda s: (s == "base").mean())
    tsn = heavy.groupby("def_ts").size()
    ts = ts[tsn >= 60]
    print(f"\nBase-defense rate vs heavy personnel (12/13/22), by defense team-season: "
          f"median {ts.median():.0%}, 10th-90th pct {ts.quantile(.1):.0%}-{ts.quantile(.9):.0%}")
    by_year = heavy.groupby("season").def_pers.apply(lambda s: (s == "base").mean())
    print("League base rate vs heavy personnel by season:", by_year.round(2).to_dict())
    for t in ("LA_2025", "BUF_2025"):
        if t in ts.index:
            print(f"  {t} defense: base vs heavy {ts[t]:.0%}")

    # H1
    print("\n=== H1. Personnel x personnel interactions (EPA/play beyond team quality + situation) ===")
    print("Baseline: 11 personnel vs nickel. Positive = better for the offense than main effects predict.")
    print(interaction_table(d).round(3).to_string())
    print("\nSame, pass plays only and run plays only (point estimates):")
    cells = [f"{o}|{p}" for o in OFF if o != "11" for p in DEF if p != "nickel"]
    for lab, sub in [("pass", d[d["pass"] == 1]), ("run", d[d["rush"] == 1])]:
        b, _ = fit(sub, cells)
        print(f"  {lab:4}:", {c: round(b[f'x:{c}'], 3) for c in cells})

    # H2
    print("\n=== H2. Do team-specific style skills persist? (split-half correlation) ===")
    pas = d["pass"] == 1
    off_ctx = {
        "in 12 personnel": d.off_g == "12",
        "in heavy (12/13/22)": d.off_g.isin(["12", "13", "22"]),
        "vs base defense": d.def_pers == "base",
        "vs man coverage (pass)": pas & (d.man == True),  # noqa: E712
        "vs two-high (pass)": pas & (d.shell == "two-high"),
        "vs single-high (pass)": pas & (d.shell == "single-high"),
        "under center": d.shotgun == 0,
    }
    def_ctx = {
        "vs 12 personnel": d.off_g == "12",
        "vs heavy (12/13/22)": d.off_g.isin(["12", "13", "22"]),
        "in base": d.def_pers == "base",
        "in man (pass)": pas & (d.man == True),  # noqa: E712
        "in two-high (pass)": pas & (d.shell == "two-high"),
    }
    tab = pd.concat([persistence(d, "offense", off_ctx), persistence(d, "defense", def_ctx)])
    print(tab.round(3).to_string(index=False))
    out = ROOT / "data" / "xo_round1.parquet"
    tab.to_parquet(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
