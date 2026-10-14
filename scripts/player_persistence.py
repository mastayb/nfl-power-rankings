"""Do player-level matchup splits persist? (and: was drift hiding team-level persistence?)

Run: python -u scripts/player_persistence.py

Each measure is a player's (or team's) situation-adjusted efficiency in a
context minus the same player's own efficiency on comparable plays, e.g. a
receiver's EPA/target vs man coverage minus his EPA/target overall. Usage
measures compare a receiver's share of team targets vs man with his overall share.

Persistence is the correlation of the measure across disjoint samples of the
same player-season (or player, season to season):
  odd/even games   interleaved: reliability with drift cancelled (both halves span the season)
  halves           weeks 1-9 vs 10+: forecastability including drift (injuries, wear, scheme)
  season to season same player, consecutive seasons (any team)
plus a lag curve (4-game blocks, correlation by gap) for measures with signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pd.set_option("display.width", 200)


def load() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data" / "cache" / "player_plays.parquet")
    d = d[d.down.notna() & d.ydstogo.notna() & d.yardline_100.notna() & d.wp.notna()].copy()
    dd = d.down.astype(int).astype(str) + pd.cut(d.ydstogo, [0, 2, 5, 10, 99], labels=list("smlx")).astype(str)
    fp = pd.cut(d.yardline_100, [0, 10, 20, 40, 60, 80, 100]).astype(str)
    wpb = pd.cut(d.wp, [-0.01, 0.1, 0.25, 0.75, 0.9, 1.01]).astype(str)
    d["adj"] = d.epa - d.groupby([dd, fp, wpb, d["pass"]]).epa.transform("mean")
    # game order within each team-season, for odd/even and blocks
    order = d[["season", "posteam", "week"]].drop_duplicates().sort_values("week")
    order["gidx"] = order.groupby(["season", "posteam"]).cumcount()
    d = d.merge(order, on=["season", "posteam", "week"])
    d["oddeven"] = np.where(d.gidx % 2 == 0, 1, 2)
    d["halves"] = np.where(d.week <= 9, 1, 2)
    d["block"] = d.gidx // 4
    return d


def split_measure(sub, base, unit, split, min_ctx, min_base):
    """Per unit and split part: mean adj in context - mean adj on comparable plays."""
    c = sub.groupby([unit, split]).adj.agg(["mean", "size"])
    b = base.groupby([unit, split]).adj.agg(["mean", "size"])
    m = (c["mean"] - b["mean"].reindex(c.index)).where((c["size"] >= min_ctx)
                                                       & (b["size"].reindex(c.index) >= min_base))
    return m.unstack()


def r_parts(w: pd.DataFrame) -> tuple[float, int]:
    w = w.dropna()
    return (w.iloc[:, 0].corr(w.iloc[:, 1]) if len(w) > 10 else np.nan), len(w)


def season_to_season(sub, base, unit_id, min_ctx, min_base):
    c = sub.groupby([unit_id, "season"]).adj.agg(["mean", "size"])
    b = base.groupby([unit_id, "season"]).adj.agg(["mean", "size"])
    m = (c["mean"] - b["mean"].reindex(c.index)).where((c["size"] >= 2 * min_ctx)
                                                       & (b["size"].reindex(c.index) >= 2 * min_base))
    m = m.dropna()
    nxt = m.copy()
    nxt.index = pd.MultiIndex.from_arrays([m.index.get_level_values(0), m.index.get_level_values(1) - 1])
    j = pd.concat([m, nxt], axis=1, join="inner").dropna()
    return (j.iloc[:, 0].corr(j.iloc[:, 1]) if len(j) > 10 else np.nan), len(j)


def evaluate(name, d, unit_id, ctx, base_mask, min_ctx, min_base):
    d = d.assign(unit=d[unit_id].astype(str) + "_" + d.season.astype(str))
    sub, base = d[ctx & base_mask], d[base_mask]
    row = {"measure": name}
    for split in ("oddeven", "halves"):
        r, n = r_parts(split_measure(sub, base, "unit", split, min_ctx, min_base))
        row[f"r_{split}"], row[f"n_{split}"] = r, n
    row["r_season_to_season"], row["n_s2s"] = season_to_season(sub, base, unit_id, min_ctx, min_base)
    return row


def overall(name, d, unit_id, mask, min_n):
    """Benchmark: overall adj efficiency level (vs league), same splits."""
    d = d[mask].assign(unit=d[unit_id].astype(str) + "_" + d.season.astype(str))
    row = {"measure": name}
    for split in ("oddeven", "halves"):
        g = d.groupby(["unit", split]).adj.agg(["mean", "size"])
        w = g["mean"].where(g["size"] >= min_n).unstack()
        row[f"r_{split}"], row[f"n_{split}"] = r_parts(w)
    s = d.groupby([unit_id, "season"]).adj.agg(["mean", "size"])
    s = s["mean"].where(s["size"] >= 2 * min_n).dropna()
    nxt = s.copy()
    nxt.index = pd.MultiIndex.from_arrays([s.index.get_level_values(0), s.index.get_level_values(1) - 1])
    j = pd.concat([s, nxt], axis=1, join="inner").dropna()
    row["r_season_to_season"], row["n_s2s"] = j.iloc[:, 0].corr(j.iloc[:, 1]), len(j)
    return row


def target_share(d, split):
    """Receiver share of team targets vs man minus overall share, per player-season split."""
    t = d[(d["pass"] == 1) & d.receiver_player_id.notna()].copy()
    t["unit"] = t.receiver_player_id + "_" + t.season.astype(str)
    games = t.groupby("unit").game_id.apply(set)
    team_t = t.groupby(["game_id", "posteam"]).agg(all=("man", "size"), man=("man", lambda m: (m == True).sum()))  # noqa: E712
    pl = t.groupby(["unit", "game_id", "posteam", split]).agg(all=("man", "size"),
                                                              man=("man", lambda m: (m == True).sum()))  # noqa: E712
    pl = pl.join(team_t, on=["game_id", "posteam"], rsuffix="_team").reset_index()
    agg = pl.groupby(["unit", split])[["all", "man", "all_team", "man_team"]].sum()
    ok = (agg["man"] + 0 >= 0) & (agg["man_team"] >= 30) & (agg["all"] >= 15)
    m = (agg["man"] / agg["man_team"] - agg["all"] / agg["all_team"]).where(ok).unstack()
    del games
    return r_parts(m)


def lag_curve(d, unit_id, ctx, base_mask, min_ctx, min_base, max_lag=3):
    d = d.assign(unit=d[unit_id].astype(str) + "_" + d.season.astype(str))
    w = split_measure(d[ctx & base_mask], d[base_mask], "unit", "block", min_ctx, min_base)
    out = {}
    for lag in range(1, max_lag + 1):
        pairs = [(w[b], w[b + lag]) for b in w.columns if b + lag in w.columns]
        a = pd.concat([p[0] for p in pairs]); c = pd.concat([p[1] for p in pairs])
        j = pd.concat([a.reset_index(drop=True), c.reset_index(drop=True)], axis=1).dropna()
        out[lag] = (round(j.iloc[:, 0].corr(j.iloc[:, 1]), 3), len(j))
    return out


def main() -> None:
    d = load()
    pas = (d["pass"] == 1)
    tgt = pas & d.receiver_player_id.notna()
    db = d.qb_dropback == 1
    run = (d.rush == 1) & d.rusher_player_id.notna()
    man = d.man == True  # noqa: E712
    rows = [
        overall("WR/TE: EPA per target (overall level)", d, "receiver_player_id", tgt, 25),
        evaluate("WR/TE: EPA/target vs man (vs own)", d, "receiver_player_id", man, tgt, 8, 25),
        evaluate("WR/TE: EPA/target vs single-high (vs own)", d, "receiver_player_id", d.shell == "single-high", tgt, 8, 25),
        overall("QB: EPA per dropback (overall level)", d, "passer_player_id", db & d.passer_player_id.notna(), 80),
        evaluate("QB: EPA/dropback vs blitz (vs own)", d, "passer_player_id", d.blitz == 1, db & d.passer_player_id.notna(), 20, 80),
        evaluate("QB: EPA/dropback vs man (vs own)", d, "passer_player_id", man, db & d.passer_player_id.notna(), 20, 80),
        evaluate("QB: EPA/dropback vs single-high (vs own)", d, "passer_player_id", d.shell == "single-high", db & d.passer_player_id.notna(), 30, 80),
        overall("RB: EPA per carry (overall level)", d, "rusher_player_id", run, 40),
        evaluate("RB: EPA/carry vs 8+ box (vs own)", d, "rusher_player_id", d.heavy_box == 1, run, 10, 40),
        overall("Team offense: EPA/play (overall level)", d, "posteam", d.posteam.notna(), 200),
        evaluate("Team offense: vs single-high (vs own pass)", d, "posteam", d.shell == "single-high", pas, 50, 150),
        evaluate("Team offense: vs man (vs own pass)", d, "posteam", man, pas, 40, 150),
        evaluate("Team offense: vs blitz (vs own dropbacks)", d, "posteam", d.blitz == 1, db, 30, 150),
    ]
    tab = pd.DataFrame(rows)
    print("=== Persistence of player and team splits, 2018-2025 ===")
    print("r_oddeven = reliability (drift cancelled); r_halves = forecastability incl. drift; "
          "r_season_to_season = same player/team next year")
    print(tab.round(3).to_string(index=False))

    print("\nUsage: WR/TE share of team targets vs man minus overall share")
    for split in ("oddeven", "halves"):
        r, n = target_share(d, split)
        print(f"  {split:8} r = {r:.3f} (n={n})")

    print("\nLag curves (4-game blocks within a season; correlation by gap in blocks; (r, n)):")
    for name, unit, ctx, base, a, b in [
        ("QB vs blitz", "passer_player_id", d.blitz == 1, db & d.passer_player_id.notna(), 8, 30),
        ("Team offense vs single-high", "posteam", d.shell == "single-high", pas, 20, 60),
        ("WR/TE vs man", "receiver_player_id", man, tgt, 4, 10),
    ]:
        print(f"  {name:28}", lag_curve(d, unit, ctx, base, a, b))
    tab.to_parquet(ROOT / "data" / "player_persistence.parquet", index=False)


if __name__ == "__main__":
    main()
