"""Audit: re-run every persistence measure with season means removed.

Run: python -u scripts/persistence_audit.py

Why: participation data changes source in 2023 (NGS through 2022, FTN Data
from 2023) and some definitions differ between them (e.g. NGS never flags
pressure on sacked plays). Pooling team- or player-seasons across seasons with
different average levels inflates split-sample correlations even when nothing
about the unit persists: both halves share the season's offset. Demeaning each
split by season keeps only within-season differences between units.

Reports raw and season-demeaned odd/even correlations, and demeaned season-to-
season correlations (seasons demeaned before pairing).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from player_persistence import load, split_measure  # noqa: E402

pd.set_option("display.width", 200)


def demeaned_r(w: pd.DataFrame) -> tuple[float, float, int]:
    w = w.dropna().copy()
    w["season"] = w.index.str[-4:]
    raw = w[1].corr(w[2])
    dm = w.groupby("season")[[1, 2]].transform(lambda c: c - c.mean())
    return raw, dm[1].corr(dm[2]), len(w)


def season_values(w_full: pd.Series) -> float:
    """w_full: Series indexed by '<id>_<season>' (full-season measure). Demeaned season-to-season r."""
    s = w_full.dropna()
    df = pd.DataFrame({"v": s.values, "id": s.index.str[:-5], "season": s.index.str[-4:].astype(int)})
    df["v"] = df.v - df.groupby("season").v.transform("mean")
    nxt = df.assign(season=df.season - 1).rename(columns={"v": "v_next"})
    j = df.merge(nxt, on=["id", "season"])
    return j.v.corr(j.v_next) if len(j) > 10 else np.nan


def _order(d: pd.DataFrame) -> pd.DataFrame:
    o = d[["season", "posteam", "week"]].drop_duplicates().sort_values("week")
    o["gidx"] = o.groupby(["season", "posteam"]).cumcount()
    d = d.merge(o, on=["season", "posteam", "week"])
    d["oddeven"] = np.where(d.gidx % 2 == 0, 1, 2)
    d["oddeven_full"] = 1
    return d


def _team_rows(d, group, name, unit, mask=None, base=None, col=None, minn=(50, 150)):
    """Context EPA vs own (mask/base) or a rate (col over base) for team-season units."""
    x = d.assign(unit=d[unit] + "_" + d.season.astype(str))
    if col is None:
        w = split_measure(x[mask & base], x[base], "unit", "oddeven", *minn)
        full = split_measure(x[mask & base], x[base], "unit", "oddeven_full", 2 * minn[0], 2 * minn[1])[1]
    else:
        x = x[base]
        w = x.groupby(["unit", "oddeven"])[col].mean().unstack()
        full = x.groupby("unit")[col].mean()
    raw, dm, n = demeaned_r(w)
    return {"group": group, "measure": name, "r_oddeven_raw": raw, "r_oddeven_demeaned": dm,
            "r_season_to_season_demeaned": season_values(full), "n": n}


def team_extras() -> list[dict]:
    """Personnel (2016-2025, xo_explore) and run-direction (run_direction) measures."""
    from run_direction import load as load_runs
    from xo_explore import load as load_xo

    x = load_xo()
    x["adj"] = x.epa - x.groupby([x.dd, x.fp, x.wpb, x["pass"]]).epa.transform("mean")
    x = _order(x)
    heavy = x.off_g.isin(["12", "13", "22"])
    allp = pd.Series(True, index=x.index)
    x["is_heavy"] = heavy.astype(float)
    x["is_base"] = (x.def_pers == "base").astype(float)
    rows = [
        _team_rows(x, "Tendency", "Offense: heavy-personnel share (12/13/22)", "posteam", base=allp, col="is_heavy"),
        _team_rows(x, "Tendency", "Defense: base rate vs heavy personnel", "defteam", base=heavy, col="is_base"),
        _team_rows(x, "Situational", "Offense: in 12 personnel", "posteam", x.off_g == "12", allp),
        _team_rows(x, "Situational", "Offense: in heavy personnel", "posteam", heavy, allp),
        _team_rows(x, "Situational", "Defense: vs heavy personnel", "defteam", heavy, allp),
        _team_rows(x, "Situational", "Defense: in base", "defteam", x.def_pers == "base", allp),
    ]
    r = _order(load_runs().drop(columns=["oddeven", "halves", "gidx"], errors="ignore"))
    allr = pd.Series(True, index=r.index)
    r["is_out"] = r.outside.astype(float)
    rows += [
        _team_rows(r, "Tendency", "Offense: outside-run share", "posteam", base=allr, col="is_out"),
        _team_rows(r, "Situational", "Offense: outside runs", "posteam", r.outside, allr, minn=(30, 120)),
        _team_rows(r, "Situational", "Defense: vs outside runs", "defteam", r.outside, allr, minn=(30, 120)),
    ]
    return rows


def main() -> None:
    d = load()
    part = pd.concat([pd.read_parquet(ROOT / "data" / "cache" / f"participation_{y}.parquet")
                      [["nflverse_game_id", "play_id", "was_pressure"]] for y in range(2018, 2026)])
    part = part.assign(play_id=part.play_id.astype(float)).rename(columns={"nflverse_game_id": "game_id"})
    d = d.merge(part, on=["game_id", "play_id"], how="left")
    d["oddeven_full"] = 1
    db = (d.qb_dropback == 1) & d.passer_player_id.notna()
    pas = d["pass"] == 1
    tgt = pas & d.receiver_player_id.notna()
    run = (d.rush == 1) & d.rusher_player_id.notna()
    man = d.man == True  # noqa: E712
    one = d.shell == "single-high"
    pr = d.was_pressure == True  # noqa: E712
    known = db & d.was_pressure.notna()

    def keyed(unit):
        return d.assign(unit=d[unit].astype(str) + "_" + d.season.astype(str))

    def ctx(unit, cmask, bmask, a, b):
        x = keyed(unit)
        return (split_measure(x[cmask & bmask], x[bmask], "unit", "oddeven", a, b),
                split_measure(x[cmask & bmask], x[bmask], "unit", "oddeven_full", 2 * a, 2 * b)[1])

    def level(unit, mask, minn):
        x = keyed(unit)[mask]
        g = x.groupby(["unit", "oddeven"]).adj.agg(["mean", "size"])
        f = x.groupby("unit").adj.agg(["mean", "size"])
        return g["mean"].where(g["size"] >= minn).unstack(), f["mean"].where(f["size"] >= 2 * minn)

    def rate(unit, mask, col, minn):
        x = keyed(unit)[mask]
        g = x.groupby(["unit", "oddeven"])[col].agg(["mean", "size"])
        f = x.groupby("unit")[col].agg(["mean", "size"])
        return g["mean"].where(g["size"] >= minn).unstack(), f["mean"].where(f["size"] >= 2 * minn)

    d["pressured"] = d.was_pressure.astype(float)
    d["sack_f"] = d.sack.astype(float)
    d["is_man"] = man.astype(float).where(d.man.notna())
    d["is_one"] = one.astype(float).where(d.shell.isin(["single-high", "two-high"]))
    d["is_blitz"] = d.blitz
    d["defunit"] = d.defteam

    measures = {
        # benchmarks: overall level
        ("Benchmark", "Offense: overall efficiency"): level("posteam", d.posteam.notna(), 200),
        ("Benchmark", "QB: overall efficiency"): level("passer_player_id", db, 80),
        ("Benchmark", "WR/TE: overall efficiency per target"): level("receiver_player_id", tgt, 25),
        # tendencies
        ("Tendency", "Defense: man coverage rate"): rate("defunit", pas & d.man.notna(), "is_man", 60),
        ("Tendency", "Defense: one-high (MOFC) share"): rate("defunit", pas & d.shell.isin(["single-high", "two-high"]), "is_one", 60),
        ("Tendency", "Defense: blitz rate"): rate("defunit", db & d.blitz.notna(), "is_blitz", 60),
        ("Tendency", "QB: pressure rate"): rate("passer_player_id", known, "pressured", 60),
        ("Tendency", "QB: pressure-to-sack rate (2023+)"): rate("passer_player_id", known & pr & (d.season >= 2023), "sack_f", 25),
        # situational skill, relative to own
        ("Situational", "QB: vs one-high (MOFC)"): ctx("passer_player_id", one, db, 30, 80),
        ("Situational", "Offense: vs one-high (MOFC)"): ctx("posteam", one, pas, 50, 150),
        ("Situational", "QB: vs blitz"): ctx("passer_player_id", d.blitz == 1, db, 20, 80),
        ("Situational", "QB: vs man"): ctx("passer_player_id", man, db, 20, 80),
        ("Situational", "QB: under pressure"): ctx("passer_player_id", pr, known, 15, 60),
        ("Situational", "WR/TE: efficiency vs man"): ctx("receiver_player_id", man, tgt, 8, 25),
        ("Situational", "RB: vs 8+ box"): ctx("rusher_player_id", d.heavy_box == 1, run, 10, 40),
        ("Situational", "Offense: vs man"): ctx("posteam", man, pas, 40, 150),
        ("Situational", "Offense: vs blitz"): ctx("posteam", d.blitz == 1, db, 30, 150),
    }
    rows = []
    for (group, name), (w, full) in measures.items():
        raw, dm, n = demeaned_r(w)
        rows.append({"group": group, "measure": name, "r_oddeven_raw": raw, "r_oddeven_demeaned": dm,
                     "r_season_to_season_demeaned": season_values(full), "n": n})

    # receiver usage: target share vs man minus overall share (odd/even), demeaned
    from player_persistence import target_share  # noqa: F401  (raw version, for reference)
    t = d[tgt & d.man.notna()].copy()
    t["unit"] = t.receiver_player_id + "_" + t.season.astype(str)
    team = t.groupby(["game_id", "posteam"]).agg(all_t=("man", "size"), man_t=("man", lambda m: (m == True).sum()))  # noqa: E712
    pl = t.groupby(["unit", "game_id", "posteam", "oddeven"]).agg(all_p=("man", "size"), man_p=("man", lambda m: (m == True).sum()))  # noqa: E712
    pl = pl.join(team, on=["game_id", "posteam"]).reset_index()
    a = pl.groupby(["unit", "oddeven"])[["all_p", "man_p", "all_t", "man_t"]].sum()
    ok = (a.man_t >= 30) & (a.all_p >= 15)
    w = (a.man_p / a.man_t - a.all_p / a.all_t).where(ok).unstack()
    raw, dm, n = demeaned_r(w)
    rows.append({"group": "Situational (usage)", "measure": "WR/TE: share of targets vs man", "r_oddeven_raw": raw,
                 "r_oddeven_demeaned": dm, "r_season_to_season_demeaned": np.nan, "n": n})

    rows += team_extras()
    tab = pd.DataFrame(rows)
    print(tab.round(3).to_string(index=False))
    tab.to_parquet(ROOT / "data" / "persistence_audit.parquet", index=False)


if __name__ == "__main__":
    main()
