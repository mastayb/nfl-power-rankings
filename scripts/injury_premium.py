"""How much of the market's game-specific premium do injuries explain?

Run: python -u scripts/injury_premium.py

premium_g = closing spread - leave-one-out local aggregate rating prediction
(same definition as scripts/xo_round2.py), 2013-2025.

Injury features per team-game, from nflverse weekly injury reports + snap counts:
  role       a player's mean snap share in the team's EARLIER games that season
             (known before kickoff); offense or defense share, whichever is larger
  new_out    listed Out/Doubtful and played in the team's previous game
  still_out  listed Out/Doubtful and did not play the previous game (already absent,
             so partly reflected in nearby weeks' lines)
  quest      listed Questionable
Summed by position group (role-weighted), home minus away, then regressed on the
premium: coefficients are points per full-time player of that group.
Players on injured reserve aren't on weekly reports; long absences are absorbed by
the local ratings, which is why "new" absences are the main signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from xo_round2 import boot, local_loo_predictions, mundane, ols  # noqa: E402

SEASONS = range(2013, 2026)
CACHE = ROOT / "data" / "cache"
GROUPS = {"QB": ["QB"], "RB": ["RB", "FB"], "WR": ["WR"], "TE": ["TE"],
          "OL": ["T", "G", "C", "OT", "OG", "OL", "G/T"], "DL": ["DE", "DT", "NT", "DL"],
          "LB": ["LB", "ILB", "OLB", "MLB"], "CB": ["CB", "DB"], "S": ["S", "FS", "SS", "SAF"]}
POS2GROUP = {p: g for g, ps in GROUPS.items() for p in ps}
pd.set_option("display.width", 200)


def cached(name, fn):
    path = CACHE / f"{name}.parquet"
    if not path.exists():
        fn().to_parquet(path, index=False)
    return pd.read_parquet(path)


def load():
    import nflreadpy as nfl

    yrs = list(SEASONS)
    sched = cached("sched_2013_2025", lambda: nfl.load_schedules(seasons=yrs).to_pandas())
    inj = cached("injuries_2013_2025", lambda: nfl.load_injuries(seasons=yrs).to_pandas())
    snaps = cached("snaps_2013_2025", lambda: nfl.load_snap_counts(seasons=yrs).to_pandas())
    # nflverse's master players table maps PFR -> GSIS ids for ~100% of snap rows at every
    # position (the weekly rosters table covers only 30-70%, and almost no linemen)
    ids = cached("player_ids", lambda: nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]]
                 .dropna().drop_duplicates("pfr_id"))
    return sched, inj, snaps, ids


def team_game_injuries(sched, inj, snaps, ids) -> pd.DataFrame:
    snaps = snaps[snaps.game_type == "REG"].copy()
    snaps["role"] = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    snaps["gsis_id"] = snaps.pfr_player_id.map(ids.set_index("pfr_id").gsis_id)
    snaps = snaps.dropna(subset=["gsis_id"])
    played = snaps.groupby(["season", "team", "gsis_id"]).week.apply(set)
    roles = snaps.sort_values("week").groupby(["season", "team", "gsis_id"])[["week", "role"]].apply(
        lambda d: list(zip(d.week, d.role)))
    team_weeks = snaps.groupby(["season", "team"]).week.apply(lambda w: sorted(set(w)))

    inj = inj[(inj.game_type == "REG") & inj.report_status.isin(["Out", "Doubtful", "Questionable"])].copy()
    inj["group"] = inj.position.map(POS2GROUP)
    inj = inj.dropna(subset=["group"])
    rows = []
    for r in inj.itertuples():
        key = (r.season, r.team, r.gsis_id)
        hist = [x for x in roles.get(key, []) if x[0] < r.week]
        if not hist:
            continue
        role = float(np.mean([x[1] for x in hist]))
        tw = [w for w in team_weeks.get((r.season, r.team), []) if w < r.week]
        prev = tw[-1] if tw else None
        played_prev = prev is not None and prev in played.get(key, set())
        kind = "quest" if r.report_status == "Questionable" else ("new_out" if played_prev else "still_out")
        rows.append({"season": r.season, "week": r.week, "team": r.team, "group": r.group,
                     "kind": kind, "role": role})
    f = pd.DataFrame(rows)
    return f.pivot_table(index=["season", "week", "team"], columns=["kind", "group"], values="role",
                         aggfunc="sum", fill_value=0.0)


def main() -> None:
    sched, inj, snaps, ids = load()
    s = sched[(sched.game_type == "REG") & sched.result.notna() & sched.spread_line.notna()].copy()
    s["neutral"] = s.location.eq("Neutral")
    s = s.reset_index(drop=True)
    cache = CACHE / "premium_2013_2025.parquet"
    if cache.exists():
        s["pred"] = pd.read_parquet(cache).set_index("game_id").pred.reindex(s.game_id).to_numpy()
    else:
        print("fitting leave-one-out local ratings for", len(s), "games...", flush=True)
        s["pred"] = local_loo_predictions(s)
        s[["game_id", "pred"]].to_parquet(cache, index=False)
    s["premium"] = s.spread_line - s.pred
    s["ats"] = s.result - s.spread_line
    s = s.join(mundane(s))

    print("building injury features...", flush=True)
    tg = team_game_injuries(sched, inj, snaps, ids)
    tg.columns = [f"{k}_{g}" for k, g in tg.columns]
    home = tg.reindex(pd.MultiIndex.from_arrays([s.season, s.week, s.home_team])).fillna(0).to_numpy()
    away = tg.reindex(pd.MultiIndex.from_arrays([s.season, s.week, s.away_team])).fillna(0).to_numpy()
    feats = pd.DataFrame(home - away, columns=tg.columns, index=s.index)
    s = s.join(feats)
    s = s[s.week >= 2]                    # need a prior game for roles
    newcols = [c for c in feats.columns if c.startswith("new_out_")]
    stillcols = [c for c in feats.columns if c.startswith("still_out_")]
    qcols = [c for c in feats.columns if c.startswith("quest_")]
    base = ["rest_diff", "div_x_pred", "wind_x_pred"]

    def fitprint(xs, label):
        b, r2, sd = ols(s, "premium", xs)
        print(f"{label:58} R^2 {r2:.3f}  resid SD {sd:.2f}")
        return b

    print(f"\n{len(s)} games, 2013-2025 (week >= 2); premium SD {s.premium.std():.2f} pts")
    fitprint(base + ["qb_backup_diff"], "A. mundane (rest, division, wind, crude backup-QB flag)")
    fitprint(base + newcols, "B. mundane + newly-out players by position")
    fitprint(base + newcols + stillcols + qcols, "C. + still-out + questionable")
    fitprint(base + newcols + stillcols + qcols + ["qb_backup_diff"], "D. C + crude backup-QB flag")

    xs = base + newcols + stillcols + qcols
    b, r2, _ = ols(s, "premium", xs)
    ci = boot(s, "premium", xs, n=300)
    tab = pd.DataFrame({"coef": b, "lo90": ci[0.05], "hi90": ci[0.95]})
    out = []
    for kind in ("new_out", "still_out", "quest"):
        for g in GROUPS:
            c = f"{kind}_{g}"
            if c in tab.index:
                n = (s[c] != 0).sum()
                out.append({"status": kind, "group": g, "pts_per_fulltime_player": -tab.loc[c, "coef"],
                            "lo90": -tab.loc[c, "hi90"], "hi90": -tab.loc[c, "lo90"], "games_with": n})
    t = pd.DataFrame(out)
    print("\nPoints the line moves AGAINST a team per full-time player listed (model C, 90% season bootstrap):")
    print(t.pivot_table(index="group", columns="status", values="pts_per_fulltime_player")
          .reindex(list(GROUPS)).round(2).to_string())
    print("\nnewly-out detail:")
    print(t[t.status == "new_out"].set_index("group").round(2).to_string())

    # does the market under/over-react? residual of result vs close on injury features
    print("\nDoes the close fully price injuries? (result - closing spread) ~ newly-out by position")
    b2, r22, _ = ols(s, "ats", newcols)
    ci2 = boot(s, "ats", newcols, n=300)
    print((-pd.DataFrame({"coef": b2, "lo90": ci2[0.95], "hi90": ci2[0.05]})).round(2)
          .rename(columns={"coef": "extra pts vs close per player (neg = underpriced)"}).to_string())
    print(f"R^2 {r22:.4f}")
    s[["game_id", "season", "week", "premium", "ats"] + xs].to_parquet(ROOT / "data" / "injury_premium.parquet", index=False)


if __name__ == "__main__":
    main()
